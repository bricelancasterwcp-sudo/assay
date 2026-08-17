"""Budget meter invariants (plan Task 1; seconds ceiling, Task 7).

The invariant under test everywhere: a charge that would cross any
limit raises BudgetExhausted BEFORE being recorded — `spent` never
shows a value that was not actually admitted (a value that looks like
a measurement but is not).
"""

from dataclasses import replace

import pytest

from assay.budget import Budget, BudgetMeter, Spend
from assay.errors import AssayError, BudgetExhausted


class _Clock:
    """An injected clock: hands back the scripted readings in order and
    refuses to be read more often than the script allows.

    Both halves matter. The readings make elapsed time exact instead of
    ambient, and the refusal lets a test pin *how many times* the meter
    looked at the time — which is the only way to prove the no-ceiling
    path never looks at all.
    """

    def __init__(self, readings: list[float]) -> None:
        self._readings = list(readings)
        self.reads = 0

    def __call__(self) -> float:
        if self.reads >= len(self._readings):
            raise AssertionError(
                f"clock read {self.reads + 1} times; the script allows "
                f"{len(self._readings)}"
            )
        value = self._readings[self.reads]
        self.reads += 1
        return value


def test_charge_accumulates_calls_and_tokens():
    meter = BudgetMeter(Budget(max_calls=10, max_prompt_tokens=1000))

    meter.charge(120)
    meter.charge(80)

    assert meter.spent == Spend(calls=2, prompt_tokens=200)


@pytest.mark.parametrize(
    ("budget", "charges", "over_charge"),
    [
        pytest.param(
            Budget(max_calls=2, max_prompt_tokens=10_000),
            [10, 10],
            10,
            id="call_limit",
        ),
        pytest.param(
            Budget(max_calls=100, max_prompt_tokens=100),
            [60],
            50,
            id="token_limit",
        ),
    ],
)
def test_charge_past_either_limit_raises_budget_exhausted(budget, charges, over_charge):
    meter = BudgetMeter(budget)
    for tokens in charges:
        meter.charge(tokens)
    before = Spend(calls=meter.spent.calls, prompt_tokens=meter.spent.prompt_tokens)

    with pytest.raises(BudgetExhausted):
        meter.charge(over_charge)

    # The crossing charge is raised BEFORE being recorded.
    assert meter.spent == before


def test_would_exceed_predicts_without_charging():
    meter = BudgetMeter(Budget(max_calls=10, max_prompt_tokens=100))
    meter.charge(90)

    assert meter.would_exceed(20) is True
    assert meter.would_exceed(10) is False

    # Asking is free: spent unchanged by either question.
    assert meter.spent == Spend(calls=1, prompt_tokens=90)


def test_budget_exhausted_is_an_assay_error():
    assert issubclass(BudgetExhausted, AssayError)


# --- the seconds ceiling ---------------------------------------------


def test_charges_inside_the_window_pass_and_spent_seconds_advances():
    clock = _Clock([100.0, 100.5, 101.25])  # construction, charge, charge
    meter = BudgetMeter(
        Budget(max_calls=10, max_prompt_tokens=1000, max_seconds=30.0), clock=clock
    )

    meter.charge(10)
    assert meter.spent.seconds == 0.5

    meter.charge(10)
    assert meter.spent.seconds == 1.25

    # The wall clock is a THIRD limit, not a replacement for the other
    # two: calls and tokens went on being counted.
    assert meter.spent.calls == 2
    assert meter.spent.prompt_tokens == 20


def test_a_charge_exactly_at_the_ceiling_is_still_admitted():
    # The ceiling is crossed by exceeding it, not by reaching it — a run
    # that lands on its own limit spent what it was allowed to spend.
    clock = _Clock([0.0, 30.0])
    meter = BudgetMeter(
        Budget(max_calls=10, max_prompt_tokens=1000, max_seconds=30.0), clock=clock
    )

    meter.charge(10)

    assert meter.spent == Spend(calls=1, prompt_tokens=10, seconds=30.0)


def test_the_first_charge_past_the_window_raises_and_registers_nothing():
    clock = _Clock([0.0, 5.0, 30.001])
    meter = BudgetMeter(
        Budget(max_calls=10, max_prompt_tokens=1000, max_seconds=30.0), clock=clock
    )
    meter.charge(10)
    before = replace(meter.spent)

    with pytest.raises(BudgetExhausted) as excinfo:
        meter.charge(10)

    # The message names the limit that ran out, because a caller that
    # dies on the clock did not die on tokens and should not be told so.
    assert str(excinfo.value) == "budget exhausted: seconds"
    # The check precedes the call: the crossing charge is refused BEFORE
    # anything is recorded, so no call is ever cut mid-flight and the
    # seconds reading of a refused charge never lands in `spent`.
    assert meter.spent == before
    assert meter.spent == Spend(calls=1, prompt_tokens=10, seconds=5.0)


def test_no_seconds_ceiling_never_consults_the_clock():
    # EXACTLY ONE read is allowed, and that one is the construction read:
    # BudgetMeter records `_started` unconditionally, before it knows
    # whether anyone will ask about elapsed time. Every read after that
    # is a bug on this path — with no ceiling there is nothing to compare
    # against, so the meter must not ask the clock what time it is.
    clock = _Clock([0.0])
    meter = BudgetMeter(Budget(max_calls=10, max_prompt_tokens=1000), clock=clock)

    meter.charge(10)
    meter.charge(10)
    assert meter.would_exceed(10) is False

    assert clock.reads == 1
    # And `seconds` stays at its floor rather than reporting a duration
    # nobody measured.
    assert meter.spent == Spend(calls=2, prompt_tokens=20, seconds=0.0)


def test_max_seconds_defaults_to_no_ceiling():
    assert Budget(max_calls=1, max_prompt_tokens=1).max_seconds is None


def test_would_exceed_reports_the_seconds_ceiling_without_charging():
    clock = _Clock([0.0, 60.001, 60.002])
    meter = BudgetMeter(
        Budget(max_calls=10, max_prompt_tokens=1000, max_seconds=60.0), clock=clock
    )

    # Nothing has been charged, and the counts are nowhere near their
    # limits — the clock alone is what closes the meter.
    assert meter.would_exceed(1) is True
    assert meter.would_exceed(0) is True
    # Asking is free: no charge, and no seconds reading recorded either.
    assert meter.spent == Spend()


def test_the_counts_limits_are_unchanged_by_a_seconds_ceiling():
    clock = _Clock([0.0, 1.0, 2.0, 3.0])
    meter = BudgetMeter(
        Budget(max_calls=2, max_prompt_tokens=10_000, max_seconds=10_000.0), clock=clock
    )
    meter.charge(10)
    meter.charge(10)

    with pytest.raises(BudgetExhausted) as excinfo:
        meter.charge(10)

    # A run that dies on calls is told it died on calls, ceiling or no.
    assert "max_calls=2" in str(excinfo.value)
    assert meter.spent == Spend(calls=2, prompt_tokens=20, seconds=2.0)
