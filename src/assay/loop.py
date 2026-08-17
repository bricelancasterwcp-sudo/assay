"""Mini-loop discipline probe (v1.4): the missing middle.

The 2026-08-14 pair of measurements that forced this probe: the same
model landed 97% of single-call codec probes and scored 0/940 in a real
multi-turn repair loop. Single-call probes structurally cannot see loop
failure — turn discipline, repetition, anchor violations, knowing when
to stop. This probe is a SCRIPTED three-turn repair conversation: the
environment's side of the transcript is canned (the golden read → patch
→ done path), the model's replies are scored per turn, and no branching
happens on what the model actually says. That makes it an instrument
with a named shape ("scripted-loop-v2" in the lens), deterministic,
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

v1.6 — the ERROR script (the lens becomes ``scripted-loop-v2``). The
golden path only ever asks what a model does when everything works. The
failure that actually ended robigo runs was the other one: a patch comes
back "SEARCH text not found", and the model re-emits the same block,
turn after turn. So every run now also plays a second, two-turn script:
the golden first turn, then a canned transcript in which the model's
patch has FAILED to apply — the canned failure being the measured qwen
signature, the right target line with its indentation stripped — and the
file is shown unchanged. Two more scored rates:

- recovery — the next reply is a ``patch tiny.py`` that applies and
  parses (the same applies-and-parses lens as everywhere else);
- doom loop — the next reply re-emits the SEARCH it was just shown
  failing, whitespace-normalized (see ``_is_doom_loop``).

A reply can be neither: reading the file again is no recovery, and it is
no doom loop either. Both rates are None when the error script never ran
— unmeasured is not zero. Every turn of BOTH scripts is a scored turn of
the one instrument: they share the action-fidelity, repeat and anchor
denominators, which is why ``n_turns`` counts five turns per run.
"""

from __future__ import annotations

from dataclasses import dataclass

from assay import fixtures
from assay.backends.base import Backend
from assay.budget import BudgetMeter
from assay.codecs import _parse_blocks, _parses_as_python, apply_search_replace
from assay.errors import BudgetExhausted

LOOP_INSTRUMENT = "scripted-loop-v2"
_SEED_BASE = 800
#: Error runs are seeded ``seed_base + 50 + run`` so they can never
#: collide with the golden runs' ``seed_base + run``: the error script is
#: a second measurement, not a re-roll of the first.
_ERROR_SEED_OFFSET = 50
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


def broken_patch() -> str:
    """The canned FAILED patch: the real target line, indentation stripped.

    The measured qwen signature — the model finds the right line and
    emits it flush left, so SEARCH matches nothing and the patch is
    rejected. Built from the fixture's OWN lines rather than hand-typed,
    so the canned failure cannot drift away from the file the probe
    ships (a canned "failure" that would in fact apply would make the
    whole error script a lie; a test pins that it really does not).
    """
    _, original, expected = _fixture()
    old, new = original.split("\n"), expected.split("\n")
    at = next(i for i, (a, b) in enumerate(zip(old, new)) if a != b)
    return ("```\n<<<<<<< SEARCH\n" + old[at].lstrip()
            + "\n=======\n" + new[at].lstrip() + "\n>>>>>>> REPLACE\n```")


def error_turn_prompts() -> tuple[str, str]:
    """The error script: the golden first turn, then a failed patch.

    T1 is the golden T1 character for character — the model has been
    told nothing yet, so there is nothing to differ about — and T2'
    hands back the environment's side of the failure: the patch it
    "sent", the rejection, and the file unchanged.
    """
    instruction, original, _ = _fixture()
    t1 = turn_prompts()[0]  # taken from the golden script, never re-typed
    t2 = (_envelope(instruction)
          + f"\n> read {_SOURCE}\n\nContents of `{_SOURCE}`:\n{original}\n"
          + f"\n> patch {_SOURCE}\n{broken_patch()}\n"
          + "\nThe patch FAILED to apply: SEARCH text not found in "
          + f"`{_SOURCE}`.\n"
          + f"\nContents of `{_SOURCE}` (unchanged):\n{original}\n"
          + "\nYour next action:")
    return t1, t2


@dataclass(frozen=True)
class Loop:
    action_fidelity: float | None   # valid action lines / scored turns
    patch_rate: float | None        # turn-2 payload applies AND parses
    finish_rate: float | None       # turn-3 reply is `done`
    repeat_rate: float | None       # verbatim repeats / scored turns
    anchor_violations: int
    n_runs: int                     # completed GOLDEN runs (patch/finish's n)
    n_turns: int                    # every scored turn, both scripts
    # v1.6. Defaulted to None so a profile written before the error
    # script existed still constructs (`Loop(**payload)` in
    # profile.from_json): "this schema had no such field" and "the field
    # measured zero" are different facts and must not collapse.
    recovery_rate: float | None = None    # error T2' patch applies AND parses
    doom_loop_rate: float | None = None   # error T2' re-emits the failure


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


def _landed(applied: str | None) -> bool:
    """The applies-and-parses lens, spelled once for both scripts."""
    return applied is not None and _parses_as_python(applied)


def _normalized(lines: list[str]) -> list[str]:
    return [" ".join(line.split()) for line in lines]


def _is_doom_loop(reply: str, *, applied: str | None) -> bool:
    """The reply re-emits the SEARCH it was just shown failing.

    Whitespace-normalized per line, so a re-emission with cosmetically
    different spacing is still the same doom loop. That normalization
    also erases the leading indentation which is the ONLY difference
    between the canned broken block and a correct one — so whether the
    block APPLIED is the discriminator: the failure being repeated here
    is "SEARCH text not found", and a SEARCH that matched this time is
    not that failure, whatever else is wrong with the reply.

    ``applied`` (the result of applying the reply to the file, None when
    it does not apply) is therefore a REQUIRED argument rather than a
    caller-side gate. The predicate was gate-free once, and the gate
    promptly went missing on one path: a correct fix wrapped in prose
    scored as a doom loop, because "did not apply" and "was not offered
    as a patch action" had been collapsed into one None.
    """
    if applied is not None:
        return False
    blocks = _parse_blocks(reply)
    if len(blocks) != 1:  # same one-block discipline as the codec lens
        return False
    canned = _parse_blocks(broken_patch())[0][0]
    return _normalized(blocks[0][0]) == _normalized(canned)


@dataclass
class _Tally:
    """The counters every scored turn of every script contributes to."""

    valid: int = 0
    turns: int = 0
    repeats: int = 0
    violations: int = 0

    def score(self, text: str, seen: list[str]) -> tuple[str | None, str]:
        self.turns += 1
        if text in seen:
            self.repeats += 1
        seen.append(text)
        verb, arg = _parse_action(text)
        if verb is not None:
            self.valid += 1
        if verb == "patch" and arg.startswith(_TESTS):
            self.violations += 1
        return verb, arg


def _ask(
    backend: Backend, meter: BudgetMeter, prompt: str, seed: int
) -> str | None:
    """One charged turn, or None when the budget refused it."""
    try:
        meter.charge(max(1, len(prompt) // 4))
    except BudgetExhausted:
        return None
    return backend.generate(prompt, seed=seed, max_tokens=_MAX_TOKENS).text


def _golden_run(
    backend: Backend, meter: BudgetMeter, tally: _Tally, *,
    seed: int, original: str,
) -> tuple[bool, bool, bool]:
    """(completed, patch landed, finished) for one golden run."""
    seen: list[str] = []
    landed = finished = False
    for turn, prompt in enumerate(turn_prompts()):
        text = _ask(backend, meter, prompt, seed)
        if text is None:
            return False, landed, finished
        verb, arg = tally.score(text, seen)
        if turn == 1 and verb == "patch" and arg.startswith(_SOURCE):
            landed = _landed(apply_search_replace(original, text))
        if turn == 2 and verb == "done":
            finished = True
    return True, landed, finished


def _error_run(
    backend: Backend, meter: BudgetMeter, tally: _Tally, *,
    seed: int, original: str,
) -> tuple[bool, bool, bool]:
    """(completed, recovered, doom-looped) for one error-script run."""
    seen: list[str] = []
    first, second = error_turn_prompts()
    text = _ask(backend, meter, first, seed)
    if text is None:
        return False, False, False
    tally.score(text, seen)

    text = _ask(backend, meter, second, seed)
    if text is None:
        return False, False, False
    verb, arg = tally.score(text, seen)
    # `applied` is computed from the reply ALONE, never inside the
    # action-line branch: "this payload does not apply to the file" and
    # "this reply was not framed as a patch action" are different facts,
    # and collapsing them into one None makes a correct fix offered in
    # prose read as a re-emission of the failed SEARCH. Recovery needs
    # both facts; the doom lens needs only the first.
    applied = apply_search_replace(original, text)
    targeted = verb == "patch" and arg.startswith(_SOURCE)
    recovered = targeted and _landed(applied)
    return True, recovered, _is_doom_loop(text, applied=applied)


def probe_loop(
    backend: Backend,
    meter: BudgetMeter,
    *,
    runs: int = 3,
    seed_base: int = _SEED_BASE,
) -> Loop:
    _, original, _ = _fixture()
    tally = _Tally()

    golden_runs = patched = finished = 0
    for run in range(runs):
        complete, landed, done = _golden_run(
            backend, meter, tally, seed=seed_base + run, original=original)
        if not complete:
            break
        golden_runs += 1
        patched += 1 if landed else 0
        finished += 1 if done else 0

    error_runs = recovered = doomed = 0
    # A golden run cut short means the meter cannot pay for a golden
    # turn, and T2' — the file twice over plus the failed patch — is the
    # longest prompt in the instrument, so no error run could reach the
    # turn it exists to score. Starting one anyway would spend calls
    # (its T1 is the SHORT golden T1 and might well be admitted) to
    # measure nothing, so the script is not run at all and its rates
    # stay None: unmeasured, never zero.
    if golden_runs == runs:
        for run in range(runs):
            complete, ok, doom = _error_run(
                backend, meter, tally,
                seed=seed_base + _ERROR_SEED_OFFSET + run, original=original)
            if not complete:
                break
            error_runs += 1
            recovered += 1 if ok else 0
            doomed += 1 if doom else 0

    if tally.turns == 0:
        return Loop(action_fidelity=None, patch_rate=None, finish_rate=None,
                    repeat_rate=None, anchor_violations=0, n_runs=0, n_turns=0)
    return Loop(
        action_fidelity=tally.valid / tally.turns,
        patch_rate=(patched / golden_runs) if golden_runs else None,
        finish_rate=(finished / golden_runs) if golden_runs else None,
        repeat_rate=tally.repeats / tally.turns,
        anchor_violations=tally.violations,
        n_runs=golden_runs,
        n_turns=tally.turns,
        recovery_rate=(recovered / error_runs) if error_runs else None,
        doom_loop_rate=(doomed / error_runs) if error_runs else None,
    )
