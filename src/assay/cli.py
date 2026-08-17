"""assay CLI (spec §9, plan Task 11).

``assay probe URL --model NAME`` runs the full suite and prints the
human table (``--json PATH`` writes the profile document). Subcommands
``geometry | ceiling | envelope | codecs`` run one family and print its
slice as JSON. ``assay report`` renders N profiles as one HTML matrix,
and ``assay diff OLD.json NEW.json`` compares two profile documents
without touching an endpoint at all.

Exit codes are PER-SUBCOMMAND. The measuring commands (probe, the
family slices, report) use the robigo taxonomy minus model-outcome
codes:

  0  profile/slice/report produced, whatever it says
  2  budget exhausted before ANY family completed
  4  infrastructure failure before any measurement

``diff`` measures nothing, so 2 means something else there and 1 —
unused by every other subcommand — carries its answer:

  0  comparable, and nothing moved beyond noise (with ``--gate``:
     nothing moved in the regression direction)
  1  drift found (with ``--gate``: a REGRESSION was found; an
     improvement alone still exits 0)
  2  not comparable — a different model, quant, weight size, or
     hardware tier, so nothing was subtracted and nothing is reported
  4  a profile file could not be read or parsed. Never 1: exit 1 from
     this command claims a measured change, and an unreadable file
     measured nothing.

The CLI supplies documented budget defaults (the default full mode: 500
calls / 1M prompt tokens; quick: 130 / 220k); the library requires an
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
from assay.diff import DiffResult, diff_profiles, render_diff
from assay.envelope import probe_envelope
from assay.errors import BudgetExhausted, InfrastructureError
from assay.geometry import free_vram_mib, plan_window
from assay.profile import render_table
from assay.replay import CallRecorder
from assay.run import MODE_PARAMS, ceiling_cap_for, probe

# Defaults must cover the WORST-case suite, not the clean one — a
# default below the suite's own call count exhausts mid-family on every
# run (this bit once at 60 and nearly again at 80).
#
# Quick (v1.6): 2 calibration + 5 ladder + ~7 bisection + 9 shape probes
# + 10 envelope + 45 codecs + 15 loop + 2 speed + 4 long-output rungs +
# 10 tools ≈ 109. THE QUICK CALL BUDGET IS RAISED 110 -> 130 HERE, and
# the reason is exactly the failure this comment block exists to
# prevent: v1.6 added the loop's error script (+2 per run, so 9 -> 15)
# and the tools family (+10), which put the worst case at 109 of 110 —
# one call from a mid-family death, on the mode an operator reaches for
# when they are in a hurry. 130 restores the headroom the 110 was chosen
# for. Full is sequential, so its worst case IS thorough's old worst
# case (no cell decides early and every one runs to the 35-sample cap):
# 2 calibration + ~12 ladder + 9 shapes + 30 envelope + up to 315 codec
# + 25 loop + 4 speed + 4 long-output rungs + 40 tools ≈ 441 of 500 —
# still comfortable, so it does not move. The tools term is 40 rather
# than v1.6's 10 because full now samples that family sequentially too
# (v1.7): 20 tasks x 2 turns is the cap a pool that never decides runs
# to. A typical run stops well short of either: the budget covers the
# case where nothing decides.
#
# Token side: the long-output ladder is the one family whose charge is
# dominated by GENERATION, not prompt — 4 rungs at 512/1024/2048/4096
# charge 7,832 tokens, because a 4096-token generation shares the window
# with its prompt and must not be priced like a 512-token one. Measured
# on the scripted suite (re-measured v1.7), a clean quick run spends 102
# calls and 78,832 of 220,000 prompt tokens, and a clean full run 441
# calls and 226,009 of 1,000,000; the worst case (a failing ceiling adds
# its bisection calls) stays inside both. Quick's token ceiling rises
# 200k -> 220k with its call ceiling so the two stay proportionate — a
# call budget that outruns its token budget just moves the mid-family
# death to the other meter.
DEFAULT_BUDGETS = {
    "quick": Budget(max_calls=130, max_prompt_tokens=220_000),
    "full": Budget(max_calls=500, max_prompt_tokens=1_000_000),
    "thorough": Budget(max_calls=500, max_prompt_tokens=1_000_000),
}

#: The commands that spend GPU time, and so take a budget.
_COMMANDS = ("probe", "geometry", "ceiling", "envelope", "codecs")
_REPORT_COMMAND = "report"
_DIFF_COMMAND = "diff"
#: The key every profile schema has carried since v1. A document
#: without it is not a profile, whatever else it parses as.
PROFILE_VERSION_KEY = "assay_profile_version"


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
    diff = subparsers.add_parser(
        _DIFF_COMMAND,
        help="compare two profile JSONs: what moved beyond noise "
             "(exit 1 = drift, 2 = not comparable)")
    diff.add_argument("old", type=Path, help="the earlier profile JSON")
    diff.add_argument("new", type=Path, help="the later profile JSON")
    diff.add_argument(
        "--gate", action="store_true",
        help="CI mode: exit 1 only for REGRESSIONS, not for any change "
             "(a model that got faster should not fail a build)",
    )
    diff.add_argument(
        "--json", dest="json_path", type=Path,
        help="write the full diff result to this path",
    )
    return parser


def _run_report(args: argparse.Namespace) -> int:
    from assay.report import render_report

    # Same gate as ``diff``, for the same reason: a matrix row asserts
    # that a model was measured. An unreadable file used to traceback
    # here, and ``{}`` — an object with no version key — rendered a row
    # of "unmeasured" badges under whatever name the file had, which is
    # a published capability claim for a run nobody made. Exit 4 (see
    # ``_load_profile``), and write no report at all.
    docs = [_load_profile(path) for path in args.profiles]
    args.out.write_text(render_report(docs), encoding="utf-8")
    print(f"wrote {args.out} ({len(docs)} profile(s))")
    return 0


def _load_profile(path: Path) -> dict:
    """Read one profile document as the RAW dict ``diff`` wants.

    Every way this can fail is an ``InfrastructureError`` (exit 4), and
    the reason is deliberate: ``diff``'s exit 1 asserts that a measured
    number moved. A path that does not exist, a truncated file, a JSON
    array where a profile belongs — none of them measured anything, and
    a CI gate that read them as 1 would report drift nobody observed.

    Being an object is NOT enough, and this is the failure worth
    naming. ``{}`` — or a saved ``{"error": ...}`` reply — parses,
    passes any isinstance check, and then sails through the whole
    comparator: the identity gate reads ``None == None`` on every
    field and calls it comparable, no family finds a cell, and the
    command exits **0** on "no drift beyond noise". That is a green CI
    check for a file nobody measured, and unlike a loud exit 1 nobody
    ever investigates it. So a profile must SAY it is one:
    ``assay_profile_version`` is the key every schema this project has
    written carries, v1 included.

    The version key alone is not enough either, and the gap is the same
    one: ``{"assay_profile_version": 5}`` passes the key check and then
    reads ``None == None`` on the model name, so the identity gate calls
    it comparable and a self-diff still exits 0. A document that names
    no model has not said WHICH model was measured — the one fact every
    verdict, matrix row and gate result is about — so ``model.name``
    must be there and must not be blank.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InfrastructureError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise InfrastructureError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise InfrastructureError(
            f"{path} is not a profile document: found a JSON "
            f"{type(payload).__name__}, not an object"
        )
    if PROFILE_VERSION_KEY not in payload:
        raise InfrastructureError(
            f"{path} is not a profile document: no {PROFILE_VERSION_KEY} key "
            f"(an object without it compares clean against anything, which "
            f"would exit 0 for a file that measured nothing)"
        )
    model = payload.get("model")
    name = model.get("name") if isinstance(model, dict) else None
    if not (isinstance(name, str) and name.strip()):
        raise InfrastructureError(
            f"{path} is not a profile document: no model.name "
            f"(a document that names no model compares clean against "
            f"anything, which would exit 0 for a file that measured nothing)"
        )
    return payload


def _diff_exit_code(result: DiffResult, *, gate: bool) -> int:
    """Not comparable outranks everything: it is not a clean run and it
    is not a regression, it is the absence of a comparison."""
    if not result.comparable:
        return 2
    if gate:
        return 1 if any(change.direction == "regression"
                        for change in result.changes) else 0
    return 1 if result.changes else 0


def _run_diff(args: argparse.Namespace) -> int:
    result = diff_profiles(_load_profile(args.old), _load_profile(args.new))
    print(render_diff(result))
    if args.json_path is not None:
        # The whole result, including the cells that were checked and
        # found clean: a machine reader needs to know what was compared,
        # not only what moved.
        args.json_path.write_text(
            json.dumps(dataclasses.asdict(result), indent=2) + "\n",
            encoding="utf-8",
        )
    return _diff_exit_code(result, gate=args.gate)


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
    # Only the measuring commands take a budget; report and diff read
    # files that already exist and never touch an endpoint.
    budget = _budget_for(args) if args.command in _COMMANDS else None
    try:
        if args.command == _REPORT_COMMAND:
            return _run_report(args)
        if args.command == _DIFF_COMMAND:
            return _run_diff(args)
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
