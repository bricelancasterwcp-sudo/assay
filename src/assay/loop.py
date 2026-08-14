"""Mini-loop discipline probe (v1.4): the missing middle.

The 2026-08-14 pair of measurements that forced this probe: the same
model landed 97% of single-call codec probes and scored 0/940 in a real
multi-turn repair loop. Single-call probes structurally cannot see loop
failure — turn discipline, repetition, anchor violations, knowing when
to stop. This probe is a SCRIPTED three-turn repair conversation: the
environment's side of the transcript is canned (the golden read → patch
→ done path), the model's replies are scored per turn, and no branching
happens on what the model actually says. That makes it an instrument
with a named shape ("scripted-loop-v1" in the lens), deterministic,
replayable, and ~10-15 calls — not a benchmark. robigo's frozen corpus
remains the real measurement; this predicts cheaply whether it is worth
running.

Scored per run:
- action fidelity — each reply parses as exactly one valid action
  (``read <file>`` / ``patch <file>`` + payload / ``done <note>``);
- patch landing — turn 2's payload applies to the fixture and parses as
  Python (the applies-and-parses lens, same as codecs);
- finishing — turn 3, after being told tests pass, is ``done``;
- repeats — a reply verbatim-identical to an earlier reply in the run;
- anchor violations — any attempt to patch the declared read-only test
  file, the cardinal robigo sin.
"""

from __future__ import annotations

from dataclasses import dataclass

from assay import fixtures
from assay.backends.base import Backend
from assay.budget import BudgetMeter
from assay.codecs import _parses_as_python, apply_search_replace
from assay.errors import BudgetExhausted

LOOP_INSTRUMENT = "scripted-loop-v1"
_SEED_BASE = 800
_MAX_TOKENS = 512

_ACTIONS = ("read", "patch", "done")
_SOURCE = "tiny.py"
_TESTS = "test_tiny.py"


def _fixture():
    entry = [e for e in fixtures.EXPECTED
             if e[0] == "tiny" and e[1] == "dropped_return"][0]
    return entry[3], entry[4], entry[5]  # instruction, original, expected


def _envelope(instruction: str) -> str:
    return (
        "You are repairing one bug. Reply with exactly ONE action line, "
        "optionally followed by a payload in a fenced block.\n"
        "Actions:\n"
        "  read <file>\n"
        "  patch <file>   (payload: one SEARCH/REPLACE block; SEARCH must "
        "match the file exactly, character for character)\n"
        "  done <note>\n"
        f"Files: {_SOURCE} (source), {_TESTS} (READ-ONLY tests — never "
        "patch it).\n"
        f"Failing behavior: {instruction}\n"
    )


def turn_prompts() -> tuple[str, str, str]:
    """The canned environment transcript, deterministic by design."""
    instruction, original, _ = _fixture()
    t1 = _envelope(instruction) + "\nYour first action:"
    t2 = (_envelope(instruction)
          + f"\n> read {_SOURCE}\n\nContents of `{_SOURCE}`:\n{original}\n"
          + "\nYour next action:")
    t3 = (_envelope(instruction)
          + f"\n> patch {_SOURCE}\n\nThe patch applied cleanly and every "
          + "test passes.\n\nYour next action:")
    return t1, t2, t3


@dataclass(frozen=True)
class Loop:
    action_fidelity: float | None   # valid action lines / scored turns
    patch_rate: float | None        # turn-2 payload applies AND parses
    finish_rate: float | None       # turn-3 reply is `done`
    repeat_rate: float | None       # verbatim repeats / scored turns
    anchor_violations: int
    n_runs: int
    n_turns: int


def _parse_action(reply: str) -> tuple[str | None, str]:
    lines = [line for line in reply.strip().splitlines() if line.strip()]
    if not lines:
        return None, ""
    first = lines[0].strip().strip("`")
    parts = first.split(None, 1)
    verb = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""
    if verb not in _ACTIONS:
        return None, ""
    return verb, arg


def probe_loop(
    backend: Backend,
    meter: BudgetMeter,
    *,
    runs: int = 3,
    seed_base: int = _SEED_BASE,
) -> Loop:
    _, original, _ = _fixture()
    prompts = turn_prompts()

    valid = 0
    scored_turns = 0
    patched = 0
    finished = 0
    repeats = 0
    violations = 0
    completed_runs = 0

    for run in range(runs):
        seen: list[str] = []
        run_complete = True
        for turn, prompt in enumerate(prompts):
            try:
                meter.charge(max(1, len(prompt) // 4))
            except BudgetExhausted:
                run_complete = False
                break
            reply = backend.generate(
                prompt, seed=seed_base + run, max_tokens=_MAX_TOKENS
            )
            text = reply.text
            scored_turns += 1
            if text in seen:
                repeats += 1
            seen.append(text)

            verb, arg = _parse_action(text)
            if verb is not None:
                valid += 1
            if verb == "patch" and arg.startswith(_TESTS):
                violations += 1
            if turn == 1 and verb == "patch" and arg.startswith(_SOURCE):
                applied = apply_search_replace(original, text)
                if applied is not None and _parses_as_python(applied):
                    patched += 1
            if turn == 2 and verb == "done":
                finished += 1
        if run_complete:
            completed_runs += 1
        else:
            break

    if scored_turns == 0:
        return Loop(action_fidelity=None, patch_rate=None, finish_rate=None,
                    repeat_rate=None, anchor_violations=0, n_runs=0, n_turns=0)
    return Loop(
        action_fidelity=valid / scored_turns,
        patch_rate=(patched / completed_runs) if completed_runs else None,
        finish_rate=(finished / completed_runs) if completed_runs else None,
        repeat_rate=repeats / scored_turns,
        anchor_violations=violations,
        n_runs=completed_runs,
        n_turns=scored_turns,
    )
