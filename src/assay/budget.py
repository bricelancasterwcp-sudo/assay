"""Probe budget accounting (spec §9).

Probes burn the user's GPU time; every call is charged against an
explicit `Budget` before it is made. The meter never records a charge
it refuses: a crossing charge raises `BudgetExhausted` first, so
`spent` only ever shows what was actually admitted.

Three limits, one rule. Calls and prompt tokens are counted; wall-clock
seconds are read from an injected clock. Because charging precedes
calling everywhere in this codebase, the seconds check lands BETWEEN
calls — a run that runs out of time is stopped before the next call,
never cut off mid-call with a half-measurement to show for it.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

from assay.errors import BudgetExhausted


@dataclass(frozen=True)
class Budget:
    max_calls: int
    max_prompt_tokens: int
    # None is the default and means no wall-clock ceiling at all — not a
    # very large one. The meter reads no clock it does not need.
    max_seconds: float | None = None


@dataclass
class Spend:
    calls: int = 0
    prompt_tokens: int = 0
    # Elapsed wall-clock at the moment of the last admitted charge. Stays
    # 0.0 when the budget carries no `max_seconds`: with nothing to
    # compare against, the meter never asks the clock what time it is,
    # and a duration nobody measured is not reported as one.
    seconds: float = 0.0


class BudgetMeter:
    """Charges calls against a `Budget`; `spent` is the live view."""

    def __init__(
        self, budget: Budget, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.budget = budget
        self.spent = Spend()
        self._clock = clock
        # Read once, unconditionally: the meter does not yet know whether
        # anyone will ask about elapsed time, and a start time recorded
        # after the first charge would under-report the run.
        self._started = clock()

    def would_exceed(self, prompt_tokens: int) -> bool:
        """True if charging `prompt_tokens` now would cross any limit.

        Asking is free: the clock may be read, but nothing is recorded.
        """
        return self._past_ceiling(self._elapsed()) or self._counts_would_exceed(
            prompt_tokens
        )

    def charge(self, prompt_tokens: int) -> None:
        """Record one call of `prompt_tokens`, or raise `BudgetExhausted`.

        The crossing charge raises BEFORE being recorded; `spent` keeps
        its pre-raise value.
        """
        elapsed = self._elapsed()
        if self._past_ceiling(elapsed):
            raise BudgetExhausted("budget exhausted: seconds")
        if self._counts_would_exceed(prompt_tokens):
            raise BudgetExhausted(
                f"charge of 1 call / {prompt_tokens} prompt tokens would exceed "
                f"budget (max_calls={self.budget.max_calls}, "
                f"max_prompt_tokens={self.budget.max_prompt_tokens}); "
                f"spent so far: {self.spent.calls} calls / "
                f"{self.spent.prompt_tokens} prompt tokens"
            )
        self.spent.calls += 1
        self.spent.prompt_tokens += prompt_tokens
        if elapsed is not None:
            self.spent.seconds = elapsed

    def _counts_would_exceed(self, prompt_tokens: int) -> bool:
        return (
            self.spent.calls + 1 > self.budget.max_calls
            or self.spent.prompt_tokens + prompt_tokens > self.budget.max_prompt_tokens
        )

    def _elapsed(self) -> float | None:
        """Seconds since construction, or None when no ceiling is set.

        The None arm is what keeps the clock-free path clock-free: one
        reading per charge when a ceiling exists, zero when it does not.
        """
        if self.budget.max_seconds is None:
            return None
        return self._clock() - self._started

    def _past_ceiling(self, elapsed: float | None) -> bool:
        """True when `elapsed` is beyond the ceiling.

        Takes the reading rather than taking it, so `charge` can ask this
        question about the same reading it will later record — one clock
        read per charge, and one place where the comparison lives.
        """
        ceiling = self.budget.max_seconds
        if elapsed is None or ceiling is None:  # no ceiling: nothing to cross
            return False
        return elapsed > ceiling  # reaching the ceiling is not crossing it
