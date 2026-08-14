"""Envelope fidelity probe (spec §6).

N seeded probes instruct an exact one-line reply from a tiny verb menu;
fidelity is the fraction of replies that are exactly format-valid.
Invalid replies are classified — extra prose, wrong shape, refusal —
because a 60%-fidelity-from-chattiness model and a 60%-from-refusals
model need different application responses. No grammar forcing (§0).

None-vs-zero (spec §8): zero completed probes means fidelity was never
measured — ``None``, never ``0.0``.
"""

import math
from dataclasses import dataclass

from assay.backends.base import Backend
from assay.budget import BudgetMeter
from assay.errors import BudgetExhausted

_VERBS = ("ALPHA", "BRAVO", "CHARLIE")
_REFUSAL_MARKERS = ("sorry", "can't", "cannot")
_PROMPT_TEMPLATE = (
    "Reply with exactly one line of the form: VERB ARG\n"
    "VERB must be one of: ALPHA, BRAVO, CHARLIE. ARG must be the number {i}.\n"
    "Example of a valid reply: ALPHA 7\n"
    "Reply with the single line only. Verb to use: {verb}"
)
# One short line expected back; generous headroom without inviting essays.
_MAX_REPLY_TOKENS = 32
# Conservative sizing for budget accounting ONLY (same seed value as the
# ceiling calibration, spec §5). Never reported anywhere a measurement
# could be read from it.
_EST_CHARS_PER_TOKEN = 3.0


@dataclass(frozen=True)
class Envelope:
    fidelity: float | None  # None only when n == 0 (budget died first)
    n: int  # completed probes
    failures: dict[str, int]  # {"prose": _, "shape": _, "refusal": _}


def _estimated_prompt_tokens(prompt: str) -> int:
    return max(1, math.ceil(len(prompt) / _EST_CHARS_PER_TOKEN))


def _classify(text: str, expected: str) -> str:
    """Classify one reply against the expected exact line.

    Valid = stripped reply is exactly the expected line. Otherwise:
    the correct line present as its own line among other content is
    "prose" (the answer is extractable chattiness); refusal markers
    with no correct line is "refusal"; anything else is "shape".
    """
    if text.strip() == expected:
        return "valid"
    if expected in (line.strip() for line in text.splitlines()):
        return "prose"
    lowered = text.lower()
    # A refusal marker only counts when the reply contains NO attempt at
    # the verb menu — "I can't fit ALPHA 7 on one line" is a shape
    # failure by a model that tried, not a refusal (external review,
    # 2026-08-13: substring matching misfiled shape as refusal).
    attempted = any(verb in text for verb in _VERBS)
    if not attempted and any(m in lowered for m in _REFUSAL_MARKERS):
        return "refusal"
    return "shape"


def probe_envelope(
    backend: Backend, meter: BudgetMeter, *, n: int, seed_base: int = 100
) -> Envelope:
    """Run n seeded one-line probes; report fidelity and failure classes.

    Budget exhaustion mid-run reports what was measured (``n`` =
    completed probes); infrastructure errors propagate (spec §3).
    """
    failures = {"prose": 0, "shape": 0, "refusal": 0}
    valid = 0
    completed = 0
    for i in range(n):
        verb = _VERBS[i % len(_VERBS)]
        prompt = _PROMPT_TEMPLATE.format(i=i, verb=verb)
        try:
            meter.charge(_estimated_prompt_tokens(prompt))
        except BudgetExhausted:
            break
        reply = backend.generate(
            prompt, seed=seed_base + i, max_tokens=_MAX_REPLY_TOKENS
        )
        outcome = _classify(reply.text, f"{verb} {i}")
        completed += 1
        if outcome == "valid":
            valid += 1
        else:
            failures[outcome] += 1
    fidelity = None if completed == 0 else valid / completed
    return Envelope(fidelity=fidelity, n=completed, failures=failures)
