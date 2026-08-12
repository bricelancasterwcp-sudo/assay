"""Empirical ceiling probe (spec §5).

Finds the largest prompt the daemon's serving path handles correctly
and names the failure mode past it — automating the manual discovery
of robigo's Ollama ceiling (≤ ~11.5k fine; past it, stats-free 200s).

The canary instruction rides at the FRONT of every probe prompt
because Ollama front-truncates: truncation eats the canary, and a
reply that does not begin with it is the detection signal.

Model behavior (a dropped canary with counts proving the prompt
arrived whole) is DATA — ``attention_loss`` is evidence, never a
failure. Infrastructure signals (``hard_error``, ``missing_stats``,
``silent_truncation``, ``canary_loss``) are failures and drive the
ladder + bisection.
"""

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from assay.backends.base import Backend, Reply
from assay.budget import BudgetMeter
from assay.errors import BudgetExhausted, ContractViolation, InfrastructureError

# Unmeasured chars-per-token falls back to this conservative sizing
# value; it is NEVER reported as a measurement (Calibration keeps None).
FALLBACK_CHARS_PER_TOKEN = 3.0

_CALIBRATION_EST_TOKENS = 500
_LADDER_START = 1024
_PROBE_MAX_TOKENS = 32
_TRUNCATION_FRACTION = 0.8

# Signals that fail a size and stop/steer the ladder. attention_loss
# is deliberately absent: it is a model result, not a ceiling.
_FAIL_SIGNALS = frozenset(
    {"hard_error", "missing_stats", "silent_truncation", "canary_loss"}
)
# Decision-table order, for failure_mode tie-breaks.
_MODE_ORDER = ("missing_stats", "hard_error", "silent_truncation", "canary_loss")

# Fixed naturalistic word list for filler (mixed lengths — one repeated
# token tokenizes unrepresentatively).
_FILLER_WORDS = (
    "time", "year", "people", "way", "day", "man", "thing", "woman",
    "life", "child", "world", "school", "state", "family", "student",
    "group", "country", "problem", "hand", "part", "place", "case",
    "week", "company", "system", "program", "question", "work",
    "government", "number", "night", "point", "home", "water", "room",
    "mother", "area", "money", "story", "fact", "month", "lot",
    "right", "study", "book", "eye", "job", "word", "business",
    "issue", "side", "kind", "head", "house", "service", "friend",
    "father", "power", "hour", "game", "line", "end", "member", "law",
    "car", "city", "community", "name", "president", "team", "minute",
    "idea", "body", "information", "back", "parent", "face", "others",
    "level", "office", "door", "health", "person", "art", "war",
    "history", "party", "result", "change", "morning", "reason",
    "research", "girl", "guy", "moment", "air", "teacher", "force",
    "education", "foot", "boy", "age", "policy", "process", "music",
    "market", "sense", "nation", "plan", "college", "interest",
    "death", "experience", "effect", "use", "class", "control",
    "care", "field", "development", "role", "effort", "rate", "heart",
    "drug", "show", "leader", "light", "voice", "wife", "whole",
    "police", "mind", "finally", "return", "free", "military",
    "price", "report", "less", "according", "decision", "explain",
    "son", "hope", "even", "develop", "view", "relationship",
    "carry", "town", "road", "drive", "arm", "true", "federal",
    "break", "better", "difference", "thanks", "possible", "fine",
    "certainly", "material", "particular", "evidence", "presence",
    "umbrella", "gigantic", "wonderful", "mysterious", "elaborate",
    "fundamental", "considerable", "afternoon", "mountain", "river",
    "garden", "window", "kitchen", "journey", "silence", "pattern",
    "texture", "harbor", "signal", "measure", "balance", "current",
    "shadow", "letter", "bridge", "castle", "engine", "fabric",
    "island", "lantern", "meadow", "needle", "orchard", "pebble",
    "quarry", "ribbon", "saddle", "timber", "valley", "willow",
    "anchor", "basket", "candle", "drawer",
)


@dataclass(frozen=True)
class Calibration:
    chars_per_token: float | None  # None = counts unavailable; NEVER a guess
    counts_available: bool
    deterministic: bool | None  # two identical seeded calls, same text?


@dataclass(frozen=True)
class CallEvidence:
    est_tokens: int
    seed: int
    signal: str
    detail: str


@dataclass(frozen=True)
class Ceiling:
    max_verified: int | None
    first_failure: int | None
    # hard_error|missing_stats|silent_truncation|canary_loss|none_up_to_cap
    # ("budget" when the meter died before any failure was found — an
    # unreached cap is never claimed as "none_up_to_cap").
    failure_mode: str
    counts_available: bool
    evidence: tuple[CallEvidence, ...]


def _canary_word(seed: int) -> str:
    return f"ASSAY-{seed}"


def _canary_prompt(seed: int, filler: str) -> str:
    # The instruction sits at the FRONT: Ollama front-truncates, so
    # truncation eats the canary — that absence is the signal.
    return (
        f"Begin your reply with exactly the word ASSAY-{seed}. That word must be\n"
        f"the very first token of your reply.\n\n"
        f"{filler}"
    )


def build_filler(rng: random.Random, est_tokens: int, chars_per_token: float) -> str:
    """Seeded naturalistic filler sized to ``est_tokens * chars_per_token`` chars."""
    target_chars = int(est_tokens * chars_per_token)
    if target_chars <= 0:
        return ""
    parts: list[str] = []
    length = 0
    while length < target_chars:
        word = rng.choice(_FILLER_WORDS)
        length += len(word) if not parts else len(word) + 1
        parts.append(word)
    return " ".join(parts)


def classify_call(
    outcome: Reply | Exception,
    *,
    sent_est: int,
    canary: str,
    counts_available: bool,
) -> str:
    """Spec §5 decision table, applied strictly in order."""
    if isinstance(outcome, ContractViolation):
        return "missing_stats"
    if isinstance(outcome, InfrastructureError):
        return "hard_error"
    reply = outcome
    tokens_in = reply.tokens_in
    canary_present = reply.text.lstrip().startswith(canary)
    if tokens_in is not None and tokens_in < _TRUNCATION_FRACTION * sent_est:
        return "silent_truncation"
    if (
        not canary_present
        and tokens_in is not None
        and tokens_in >= _TRUNCATION_FRACTION * sent_est
    ):
        # The prompt demonstrably arrived whole; the MODEL dropped the
        # instruction. Evidence, not a failure.
        return "attention_loss"
    if not canary_present and not counts_available:
        return "canary_loss"
    return "ok"


def calibrate(backend: Backend, meter: BudgetMeter, *, seed: int) -> Calibration:
    """One ~500-token probe for chars_per_token, one repeat for determinism.

    Counts missing → ``chars_per_token`` is None (the 3.0 fallback is
    sizing-only, never reported). Either call failing → the affected
    measurements are None, not guessed.
    """
    filler = build_filler(
        random.Random(seed), _CALIBRATION_EST_TOKENS, FALLBACK_CHARS_PER_TOKEN
    )
    prompt = _canary_prompt(seed, filler)
    meter.charge(_CALIBRATION_EST_TOKENS)
    try:
        first = backend.generate(prompt, seed=seed, max_tokens=_PROBE_MAX_TOKENS)
    except InfrastructureError:
        return Calibration(
            chars_per_token=None, counts_available=False, deterministic=None
        )
    counts_available = first.tokens_in is not None
    chars_per_token = (
        len(prompt) / first.tokens_in
        if first.tokens_in is not None and first.tokens_in > 0
        else None
    )
    meter.charge(_CALIBRATION_EST_TOKENS)
    try:
        second = backend.generate(prompt, seed=seed, max_tokens=_PROBE_MAX_TOKENS)
    except InfrastructureError:
        return Calibration(
            chars_per_token=chars_per_token,
            counts_available=counts_available,
            deterministic=None,
        )
    return Calibration(
        chars_per_token=chars_per_token,
        counts_available=counts_available,
        deterministic=first.text == second.text,
    )


class _MeterDry(Exception):
    """Internal: the budget died mid-probe after evidence existed."""


def _majority_failure(signals: Sequence[str]) -> str:
    """Majority failing classification; ties break toward table order."""
    counts = Counter(s for s in signals if s in _FAIL_SIGNALS)
    return max(counts, key=lambda s: (counts[s], -_MODE_ORDER.index(s)))


def probe_ceiling(
    backend: Backend,
    meter: BudgetMeter,
    *,
    cap_tokens: int,
    seeds: Sequence[int] = (0, 1),
    calibration: Calibration,
) -> Ceiling:
    """Canary ladder (1024 doubling to cap) then bisection to ~10%."""
    chars_per_token = (
        calibration.chars_per_token
        if calibration.chars_per_token is not None
        else FALLBACK_CHARS_PER_TOKEN
    )
    evidence: list[CallEvidence] = []
    signals_at: dict[int, list[str]] = {}

    def run_call(size: int, seed: int) -> None:
        try:
            meter.charge(size)
        except BudgetExhausted as exc:
            if not evidence:
                raise  # nothing measured yet: the caller must know
            evidence.append(CallEvidence(size, seed, "budget", str(exc)))
            raise _MeterDry() from exc
        prompt = _canary_prompt(seed, build_filler(random.Random(seed), size, chars_per_token))
        outcome: Reply | Exception
        try:
            outcome = backend.generate(prompt, seed=seed, max_tokens=_PROBE_MAX_TOKENS)
        except InfrastructureError as exc:
            outcome = exc
        signal = classify_call(
            outcome,
            sent_est=size,
            canary=_canary_word(seed),
            counts_available=calibration.counts_available,
        )
        if isinstance(outcome, Reply):
            detail = f"tokens_in={outcome.tokens_in} stop_reason={outcome.stop_reason}"
        else:
            detail = f"{type(outcome).__name__}: {outcome}"
        evidence.append(CallEvidence(size, seed, signal, detail))
        signals_at.setdefault(size, []).append(signal)

    def size_fails(size: int) -> bool:
        for seed in seeds:
            run_call(size, seed)
        return any(s in _FAIL_SIGNALS for s in signals_at[size])

    lo = 0  # largest fully-verified size; 0 = nothing verified (reported None)
    hi: int | None = None  # smallest fully-measured failing size
    budget_died = False
    try:
        size = _LADDER_START
        while size <= cap_tokens:
            if size_fails(size):
                hi = size
                break
            lo = size
            size *= 2
        if hi is not None:
            while hi - lo > max(lo // 10, 256):
                mid = (lo + hi) // 2
                if size_fails(mid):
                    hi = mid
                else:
                    lo = mid
    except _MeterDry:
        budget_died = True

    max_verified = lo if lo > 0 else None
    if hi is None:
        first_failure = None
        # An unreached cap is never claimed as clean.
        failure_mode = "budget" if budget_died else "none_up_to_cap"
    else:
        first_failure = hi
        failure_mode = _majority_failure(signals_at[hi])
    return Ceiling(
        max_verified=max_verified,
        first_failure=first_failure,
        failure_mode=failure_mode,
        counts_available=calibration.counts_available,
        evidence=tuple(evidence),
    )
