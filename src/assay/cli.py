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
  3  incomplete, for either of two reasons — both mean part of the
     comparison never happened, and outrank 1: neither is a measured
     move.
       - at least one cell was measured on exactly one side. A
         cross-schema pair reads 3 whenever the newer schema actually
         measured a cell the older one lacks — not merely because the
         schemas differ — which is the instrument-changed rule
         enforcing itself; a budget-mode profile against a full one
         reads 3 under the same rule, whenever the full run measured a
         cell the budget run skipped
       - at least one cell was measured on BOTH sides, but under two
         different rules (a registered ``SEMANTIC_BREAKS`` entry the
         pair straddles) — both instruments spoke, and not about the
         same thing, which no amount of re-running the older document
         fixes
  4  a profile file could not be read or parsed. Never 1: exit 1 from
     this command claims a measured change, and an unreadable file
     measured nothing

The CLI supplies documented budget defaults (the default full mode: 610
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
from assay.cover import _cover_exit_code, cover_profiles, render_cover
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
# Every term below is DERIVED, not counted by hand: ``run.worst_case_calls``
# prices each family from the constants that family's probe consumes, and
# tests/test_run.py sums that table against the numbers in this block. A
# family that grows re-prices itself there and fails the test here.
#
# Quick (v1.7): 2 calibration + 5 ladder + 9 shape probes + 10 envelope
# + 60 codecs + 15 loop + 2 speed + 4 long-output rungs + 10 tools = 117,
# which is exactly what a clean quick run spends. Add at most 4 bisection
# calls (``ceiling.bisection_worst_case_steps(16384)`` x 1 seed) and the
# worst case is 121 of 130 — an upper bound twice over, because a ladder
# only bisects when a rung FAILS and a failing ladder stopped early, so
# no run pays the whole 5 rungs AND the 4. THE QUICK CALL BUDGET WAS
# RAISED 110 -> 130 in v1.6, and the reason is exactly the failure this
# comment block exists to prevent: v1.6 added the loop's error script
# (+2 per run, so 9 -> 15) and the tools family (+10), which put the
# worst case at 109 of 110 — one call from a mid-family death, on the
# mode an operator reaches for when they are in a hurry. v1.7's three
# deeper json grades take the codec term 45 -> 60, so quick's worst case
# is now 121 of 130: inside, but with nine calls of headroom rather than
# twenty-one.
#
# Full is sequential, so its worst case IS thorough's old worst case (no
# cell decides early and every one runs to the 35-sample cap):
# 2 calibration + 12 ladder + 9 shapes + 30 envelope + up to 420 codec
# + 25 loop + 4 speed + 6 parallel lanes + 4 long-output rungs + 40 tools
# = 552. THE FULL AND THOROUGH CALL BUDGETS WERE RAISED 500 -> 600 in
# v1.7, for the same reason quick's went up in v1.6: the deep json grades
# take the codec term 315 -> 420 (json_object gained
# nested/tabular/constrained — codecs.GRADES_FOR) on top of the tools
# family's 10 -> 40 (full samples that pool sequentially too since v1.7:
# 20 tasks x 2 turns is the cap a pool that never decides runs to), and
# at 500 a clean full run died with the long-output ladder and the whole
# tools family named in `dropped`.
#
# 600 -> 610 HERE, and it is the same rule applied to a new measurement
# rather than a fresh guess: the parallel family (v1.7) adds six lanes to
# a full run — k=2 and k=4, charged in full before either launches — so
# the measured clean run went 546 -> 552 and 600 left 48 calls of
# headroom where the derivation asks for ~10%. The DEFAULT FOLLOWS THE
# MEASUREMENT (the alternative was moving the acceptance threshold to fit
# the number, which is the one thing a measurement instrument may not
# do): 552 x 1.1 = 607, rounded up to 610. The first claim on that
# headroom is the one term the per-family numbers do not carry: a failing
# ceiling ladder bisects, at most 8 calls
# (``ceiling.bisection_worst_case_steps(32768)`` x 2 seeds), so the full
# worst case is 552 + 8 = 560 of 610 and the remaining 50 is headroom
# rather than an unnamed family. A typical run stops well short: the
# budget covers the case where nothing decides.
#
# Token side: the long-output ladder is the one family whose charge is
# dominated by GENERATION, not prompt — 4 rungs at 512/1024/2048/4096
# charge 7,832 tokens, because a 4096-token generation shares the window
# with its prompt and must not be priced like a 512-token one. Measured
# on the scripted suite (re-measured v1.7, 2026-08-17), a clean quick
# run spends 117 calls and 79,420 of 220,000 prompt tokens, and a clean
# full run 552 calls and 230,293 of 1,000,000 — both now inside their
# defaults on BOTH meters, with nothing dropped. Quick's ceilings stay
# 130 / 220k (worst case 121 — quick does not measure concurrency) and
# the token ceilings do not move: a call budget that outruns its token
# budget just relocates the mid-family death to the other meter, and 1M
# is nowhere near 230k.
#
# BUDGET MODE (v1.7) is the consumer's own ceiling, so its defaults are
# only what the flags do not say. ``--budget-calls N`` replaces the call
# ceiling and ``--budget-seconds S`` adds a wall-clock one; whatever is
# left over comes from quick's documented numbers, because budget mode
# runs quick-shaped families (run.MODE_PARAMS). A consumer who names
# only seconds still gets a call ceiling — an unbounded one would be a
# budget nobody granted.
DEFAULT_BUDGETS = {
    "quick": Budget(max_calls=130, max_prompt_tokens=220_000),
    "full": Budget(max_calls=610, max_prompt_tokens=1_000_000),
    "thorough": Budget(max_calls=610, max_prompt_tokens=1_000_000),
    "budget": Budget(max_calls=130, max_prompt_tokens=220_000),
}

#: The commands that spend GPU time, and so take a budget.
_COMMANDS = ("probe", "geometry", "ceiling", "envelope", "codecs")
_REPORT_COMMAND = "report"
_DIFF_COMMAND = "diff"
_COVER_COMMAND = "cover"
#: The key every profile schema has carried since v1. A document
#: without it is not a profile, whatever else it parses as.
PROFILE_VERSION_KEY = "assay_profile_version"


class _ModeFlag(argparse.Action):
    """``--quick``/``--full``/``--thorough``, and a record that one was TYPED.

    The default mode is full, so ``args.mode`` alone cannot tell a run
    that asked for full from one that asked for nothing — and budget
    mode has to know the difference, because ``--budget-calls`` selects
    a mode and combining two mode selections is an error rather than a
    silent precedence rule.
    """

    def __init__(self, option_strings, dest, const, **kwargs) -> None:
        super().__init__(option_strings, dest, nargs=0, const=const, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, self.const)
        namespace.mode_flag_given = True


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
            "--quick", dest="mode", action=_ModeFlag, const="quick",
            help="short ladder, reduced probe counts "
                 "(fixed n=5, fixed-n lens)",
        )
        mode.add_argument(
            "--full", dest="mode", action=_ModeFlag, const="full",
            help="full seeds, full ladder, full probe counts (default; "
                 "sequential codec sampling, stops at the first decided "
                 "look)",
        )
        mode.add_argument(
            "--thorough", dest="mode", action=_ModeFlag, const="thorough",
            help="alias of --full (its old fixed n=35 is subsumed by the "
                 "sequential cap)",
        )
        sub.set_defaults(mode="full", mode_flag_given=False)
        if name == "probe":
            # PROBE ONLY, and that is the whole point of the mode: the
            # budget flags promise a priority-ordered run that drops
            # whole families by name, and only ``probe`` runs that
            # orchestrator. On a family subcommand the same flag would
            # charge calibration and then truncate the one family the
            # command exists to run — the started-and-truncated family
            # budget mode was written to forbid, under a flag whose help
            # says otherwise. Unknown here, so argparse refuses it.
            sub.add_argument(
                "--budget-calls", type=int, metavar="N",
                help="BUDGET MODE: measure the pre-registered family "
                     "priority under a ceiling of N model calls, dropping "
                     "by name every family that does not fit (implies "
                     "budget mode)",
            )
            sub.add_argument(
                "--budget-seconds", type=float, metavar="S",
                help="BUDGET MODE: stop starting families after S "
                     "wall-clock seconds; checked between calls, never "
                     "mid-call (implies budget mode; combines with "
                     "--budget-calls)",
            )
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
             "(exit 1 = drift, 2 = not comparable, 3 = incomplete)")
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
    cover = subparsers.add_parser(
        _COVER_COMMAND,
        help="does a candidate profile cover a floor profile? "
             "one-directional; crossed models allowed, crossed "
             "instruments refused (exit 1 = not covered, "
             "2 = refused, 3 = incomplete)")
    cover.add_argument("floor", type=Path,
                       help="the floor profile JSON — the requirement")
    cover.add_argument("candidate", type=Path,
                       help="the candidate profile JSON")
    cover.add_argument(
        "--json", dest="json_path", type=Path,
        help="write the full cover result to this path",
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
    is not a regression, it is the absence of a comparison.

    Incomplete (3) is the PARTIAL form of that same fact and outranks
    the measured classes for the same reason. A pair where five
    families vanish and nothing else moves is not "no drift beyond
    noise" — it is a comparison that mostly did not happen, and a
    consumer reading exit codes alone was being told the opposite.
    Measured live by bloomery's drift watch against a v8-vs-v4 pair,
    which exited 0 under --gate while long_output, tool_calling and
    three deep json cells went unmeasured on one side.

    Two DIFFERENT reasons land on 3: `dropped` (a cell measured on
    exactly one side) and `incomparable` (a cell both sides measured,
    under two different rules — v1.10). Neither is a measured move, so
    neither can ride on 1: a vanished family and a cell the two
    instruments defined differently are both an incomplete comparison,
    not a clean one and not a scored one.

    Precedence 2 > 3 > 1 > 0. Exit 1 keeps its precise claim — a
    measured number moved — which is why a vanished family or a
    straddled rule could never ride on it.
    """
    if not result.comparable:
        return 2
    if result.dropped or result.incomparable:
        return 3
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


def _run_cover(args: argparse.Namespace) -> int:
    result = cover_profiles(_load_profile(args.floor),
                            _load_profile(args.candidate))
    print(render_cover(result))
    if args.json_path is not None:
        # The whole result, covered cells included: a machine reader
        # needs to know what was compared, not only what failed.
        args.json_path.write_text(
            json.dumps(dataclasses.asdict(result), indent=2) + "\n",
            encoding="utf-8",
        )
    return _cover_exit_code(result)


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


def _apply_budget_mode(parser: argparse.ArgumentParser,
                       args: argparse.Namespace) -> None:
    """Either budget flag selects budget mode; a second mode is an error.

    The flags ARE the mode (spec §4): a consumer says what it can spend,
    not which sampling table to use. Combining them with --quick/--full/
    --thorough would leave the profile's own provenance unable to say
    which mode measured it, so argparse refuses the invocation the way
    it refuses any other mutually exclusive pair.

    Only ``probe`` defines the flags at all (``_build_parser``), so this
    is only ever asked about the command that implements the mode.
    """
    if args.budget_calls is None and args.budget_seconds is None:
        return
    if args.mode_flag_given:
        parser.error(
            "--budget-calls/--budget-seconds cannot be combined with "
            "--quick/--full/--thorough: a budget IS a mode")
    args.mode = "budget"


def _budget_for(args: argparse.Namespace) -> Budget:
    budget = DEFAULT_BUDGETS[args.mode]
    if args.max_calls is not None:
        budget = dataclasses.replace(budget, max_calls=args.max_calls)
    if args.max_prompt_tokens is not None:
        budget = dataclasses.replace(
            budget, max_prompt_tokens=args.max_prompt_tokens
        )
    # The budget flags are the mode's own ceilings, so they are applied
    # last: a run that names both --max-calls and --budget-calls asked
    # for budget mode, and budget mode's ceiling is the one it gets.
    # ``getattr``: only the probe subparser defines them, because only
    # probe runs the orchestrator that honours them.
    budget_calls = getattr(args, "budget_calls", None)
    budget_seconds = getattr(args, "budget_seconds", None)
    if budget_calls is not None:
        budget = dataclasses.replace(budget, max_calls=budget_calls)
    if budget_seconds is not None:
        budget = dataclasses.replace(budget, max_seconds=budget_seconds)
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
    parser = _build_parser()
    args = parser.parse_args(argv)
    # Only the measuring commands take a budget; report, diff and cover
    # read files that already exist and never touch an endpoint.
    if args.command == "probe":
        _apply_budget_mode(parser, args)
    budget = _budget_for(args) if args.command in _COMMANDS else None
    try:
        if args.command == _REPORT_COMMAND:
            return _run_report(args)
        if args.command == _DIFF_COMMAND:
            return _run_diff(args)
        if args.command == _COVER_COMMAND:
            return _run_cover(args)
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
