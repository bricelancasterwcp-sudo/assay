"""assay CLI (spec §9, plan Task 11).

``assay probe URL --model NAME`` runs the full suite and prints the
human table (``--json PATH`` writes the profile document). Subcommands
``geometry | ceiling | envelope | codecs`` run one family and print its
slice as JSON.

Exit codes (the robigo taxonomy, minus model-outcome codes):
  0  profile/slice produced, whatever it says
  2  budget exhausted before ANY family completed
  4  infrastructure failure before any measurement

The CLI supplies documented budget defaults (the default full mode: 500
calls / 1M prompt tokens; quick: 110 / 200k); the library requires an
explicit Budget — consent to burn GPU time is never implicit.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from assay.backends import detect_backend
from assay.budget import Budget, BudgetMeter
from assay.codecs import CodecDirectives
from assay.ceiling import calibrate, probe_ceiling
from assay.codecs import probe_codecs
from assay.envelope import probe_envelope
from assay.errors import BudgetExhausted, InfrastructureError
from assay.geometry import free_vram_mib, plan_window
from assay.profile import render_table
from assay.replay import CallRecorder
from assay.run import MODE_PARAMS, ceiling_cap_for, probe

# Defaults must cover the WORST-case suite, not the clean one — a
# default below the suite's own call count exhausts mid-family on every
# run (this bit once at 60 and nearly again at 80). Quick: 2 calibration
# + 5 ladder + ~7 bisection + 9 shape probes + 10 envelope + 45 codecs +
# 9 loop + 2 speed ≈ 89. Full is now sequential, so its worst case IS
# thorough's old worst case (no cell decides early and every one runs to
# the 35-sample cap): 2 calibration + ~12 ladder + 9 shapes + 30
# envelope + up to 315 codec + 15 loop + 4 speed ≈ 387. A typical run
# stops well short of that — the budget covers the case where nothing
# decides.
DEFAULT_BUDGETS = {
    "quick": Budget(max_calls=110, max_prompt_tokens=200_000),
    "full": Budget(max_calls=500, max_prompt_tokens=1_000_000),
    "thorough": Budget(max_calls=500, max_prompt_tokens=1_000_000),
}

_COMMANDS = ("probe", "geometry", "ceiling", "envelope", "codecs")
_REPORT_COMMAND = "report"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assay",
        description="Probe a locally-served LLM endpoint's real capabilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in _COMMANDS:
        sub = subparsers.add_parser(
            name,
            help=(
                "run the full probe suite"
                if name == "probe"
                else f"run only the {name} family and print its slice"
            ),
        )
        sub.add_argument("url", help="endpoint base URL")
        sub.add_argument("--model", required=True, help="model name to probe")
        mode = sub.add_mutually_exclusive_group()
        mode.add_argument(
            "--quick", dest="mode", action="store_const", const="quick",
            help="short ladder, reduced probe counts "
                 "(fixed n=5, fixed-n lens)",
        )
        mode.add_argument(
            "--full", dest="mode", action="store_const", const="full",
            help="full seeds, full ladder, full probe counts (default; "
                 "sequential codec sampling, stops at the first decided "
                 "look)",
        )
        mode.add_argument(
            "--thorough", dest="mode", action="store_const", const="thorough",
            help="alias of --full (its old fixed n=35 is subsumed by the "
                 "sequential cap)",
        )
        sub.set_defaults(mode="full")
        sub.add_argument(
            "--backend", choices=("ollama", "openai"),
            help="force the backend kind instead of auto-detecting",
        )
        sub.add_argument(
            "--json", dest="json_path", type=Path,
            help="write the result document to this path",
        )
        sub.add_argument(
            "--record", type=Path,
            help="record every model call to a JSONL transcript",
        )
        sub.add_argument("--max-calls", type=int, help="budget: max model calls")
        sub.add_argument(
            "--max-prompt-tokens", type=int,
            help="budget: max total estimated prompt tokens",
        )
        sub.add_argument(
            "--window-cap", type=int,
            help="user context cap: bounds geometry and the ceiling ladder",
        )
        marking = sub.add_mutually_exclusive_group()
        marking.add_argument(
            "--emulated", dest="emulated", action="store_const", const=True,
            help="mark this profile as measured on EMULATED tier hardware",
        )
        marking.add_argument(
            "--real-hardware", dest="emulated", action="store_const", const=False,
            help="mark this profile as measured on real tier hardware",
        )
        sub.set_defaults(emulated=None)
        sub.add_argument(
            "--tier", metavar="NAME",
            help="operator-declared hardware tier (e.g. average-gamer-8gb); "
                 "REQUIRES --emulated or --real-hardware — an unmarked "
                 "emulated number could masquerade as real hardware",
        )
        sub.add_argument(
            "--directives", type=Path, metavar="JSON",
            help="consumer-supplied codec presentation: a JSON object with "
                 "search_replace/whole_file/json_object directive strings; "
                 "the profile's lens records presentation=custom",
        )
    report = subparsers.add_parser(
        _REPORT_COMMAND,
        help="render one self-contained HTML report from N profile JSONs "
             "(the capability matrix)")
    report.add_argument("profiles", type=Path, nargs="+",
                        help="profile JSON files (assay probe --json output)")
    report.add_argument("--out", type=Path, default=Path("assay-report.html"))
    return parser


def _run_report(args: argparse.Namespace) -> int:
    from assay.report import render_report

    docs = [json.loads(p.read_text(encoding="utf-8")) for p in args.profiles]
    args.out.write_text(render_report(docs), encoding="utf-8")
    print(f"wrote {args.out} ({len(docs)} profile(s))")
    return 0


def _load_directives(path: Path | None) -> CodecDirectives | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = {"search_replace", "whole_file", "json_object"} - set(payload)
    if missing:
        raise SystemExit(
            f"--directives file is missing keys: {', '.join(sorted(missing))}"
        )
    return CodecDirectives(
        search_replace=payload["search_replace"],
        whole_file=payload["whole_file"],
        json_object=payload["json_object"],
    )


def _budget_for(args: argparse.Namespace) -> Budget:
    budget = DEFAULT_BUDGETS[args.mode]
    if args.max_calls is not None:
        budget = dataclasses.replace(budget, max_calls=args.max_calls)
    if args.max_prompt_tokens is not None:
        budget = dataclasses.replace(
            budget, max_prompt_tokens=args.max_prompt_tokens
        )
    return budget


def _slice_payload(value) -> object:
    if value is None:
        return None
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    # codecs matrix: codec -> grade -> Landing
    return {
        codec: {grade: dataclasses.asdict(cell) for grade, cell in grades.items()}
        for codec, grades in value.items()
    }


def _run_family(args: argparse.Namespace, budget: Budget) -> int:
    backend = detect_backend(args.url, args.model, forced=args.backend)
    if args.record is not None:
        backend = CallRecorder(backend, args.record)
    meter = BudgetMeter(budget)
    params = MODE_PARAMS[args.mode]

    if args.command == "geometry":
        info = backend.model_info()
        result = plan_window(
            info, vram_free_mib=free_vram_mib(), user_cap=args.window_cap
        )
    elif args.command == "ceiling":
        info = backend.model_info()
        calibration = calibrate(backend, meter, seed=params.seeds[0])
        result = probe_ceiling(
            backend,
            meter,
            cap_tokens=ceiling_cap_for(args.mode, info.training_ctx, args.window_cap),
            seeds=params.seeds,
            calibration=calibration,
        )
    elif args.command == "envelope":
        result = probe_envelope(backend, meter, n=params.envelope_n)
    else:  # codecs
        # The mode's stopping rule travels with the mode: a family run
        # must sample exactly as the probe command would, or --full here
        # would silently mean 315 fixed calls instead of the sequential
        # matrix (controller ruling, v1.5).
        result = probe_codecs(backend, meter,
                              n_per_cell=params.codecs_n_per_cell,
                              look_schedule=params.codec_look_schedule)

    text = json.dumps({args.command: _slice_payload(result)}, indent=2)
    print(text)
    if args.json_path is not None:
        args.json_path.write_text(text + "\n", encoding="utf-8")
    return 0


def _run_probe(args: argparse.Namespace, budget: Budget) -> int:
    profile = probe(
        args.url,
        args.model,
        budget=budget,
        mode=args.mode,
        backend=args.backend,
        record=args.record,
        window_cap=args.window_cap,
        directives=_load_directives(args.directives),
        tier=args.tier,
        emulated=args.emulated,
    )
    print(render_table(profile))
    if args.json_path is not None:
        args.json_path.write_text(profile.to_json() + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    budget = _budget_for(args) if args.command != "report" else None
    try:
        if args.command == "report":
            return _run_report(args)
        if args.command == "probe":
            return _run_probe(args, budget)
        return _run_family(args, budget)
    except BudgetExhausted as error:
        print(
            f"assay: budget exhausted before any probe family completed: {error}",
            file=sys.stderr,
        )
        return 2
    except InfrastructureError as error:
        print(f"assay: infrastructure failure: {error}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
