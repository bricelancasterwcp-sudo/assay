"""Probe budget accounting (spec §9).

Probes burn the user's GPU time; every call is charged against an
explicit `Budget` before it is made. The meter never records a charge
it refuses: a crossing charge raises `BudgetExhausted` first, so
`spent` only ever shows what was actually admitted.
"""

from dataclasses import dataclass

from assay.errors import BudgetExhausted


@dataclass(frozen=True)
class Budget:
    max_calls: int
    max_prompt_tokens: int


@dataclass
class Spend:
    calls: int = 0
    prompt_tokens: int = 0


class BudgetMeter:
    """Charges calls against a `Budget`; `spent` is the live view."""

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.spent = Spend()

    def would_exceed(self, prompt_tokens: int) -> bool:
        """True if charging `prompt_tokens` now would cross either limit.

        Asking is free: nothing is recorded.
        """
        return (
            self.spent.calls + 1 > self.budget.max_calls
            or self.spent.prompt_tokens + prompt_tokens > self.budget.max_prompt_tokens
        )

    def charge(self, prompt_tokens: int) -> None:
        """Record one call of `prompt_tokens`, or raise `BudgetExhausted`.

        The crossing charge raises BEFORE being recorded; `spent` keeps
        its pre-raise value.
        """
        if self.would_exceed(prompt_tokens):
            raise BudgetExhausted(
                f"charge of 1 call / {prompt_tokens} prompt tokens would exceed "
                f"budget (max_calls={self.budget.max_calls}, "
                f"max_prompt_tokens={self.budget.max_prompt_tokens}); "
                f"spent so far: {self.spent.calls} calls / "
                f"{self.spent.prompt_tokens} prompt tokens"
            )
        self.spent.calls += 1
        self.spent.prompt_tokens += prompt_tokens
