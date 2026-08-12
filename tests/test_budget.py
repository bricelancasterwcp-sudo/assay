"""Budget meter invariants (plan Task 1).

The invariant under test everywhere: a charge that would cross either
limit raises BudgetExhausted BEFORE being recorded — `spent` never
shows a value that was not actually admitted (a value that looks like
a measurement but is not).
"""

import pytest

from assay.budget import Budget, BudgetMeter, Spend
from assay.errors import AssayError, BudgetExhausted


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
