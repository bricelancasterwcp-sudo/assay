"""Fixed-request-shape ceiling matrix (v1.4).

Origin: the 2026-08-14 robigo subject row — a daemon that serves 14B-Q4
to 16k when num_ctx is right-sized ERRORED above ~1.8k at a fixed
num_ctx=8192, turning a benchmark row to 0/940. Applications pin
num_ctx once; this probe measures the pinned shapes.
"""

from assay.backends.base import BackendCaps, Reply
from assay.budget import Budget, BudgetMeter
from assay.ceiling import Calibration, ShapeCeiling, probe_fixed_shapes
from assay.errors import InfrastructureError

CAL = Calibration(chars_per_token=3.0, counts_available=True,
                  deterministic=True)


def meter():
    return BudgetMeter(Budget(max_calls=99, max_prompt_tokens=10**9))


class ShapeFake:
    """Errors on prompts above `breaks_at` tokens ONLY when num_ctx is
    pinned at or above `when_shape` — the 14B fingerprint."""

    caps = BackendCaps(reports_counts=True, per_request_ctx=True,
                       truncate_control=True, metadata_access=True)

    def __init__(self, breaks_at=1800, when_shape=8192):
        self.breaks_at = breaks_at
        self.when_shape = when_shape

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None):
        est = len(prompt) // 3
        if num_ctx is not None and num_ctx >= self.when_shape and est > self.breaks_at:
            raise InfrastructureError("daemon refused this request shape")
        canary = prompt.split()[6].rstrip(".")  # "the word ASSAY-<seed>."
        return Reply(text=f"{canary} ok", tokens_in=est, tokens_out=2,
                     stop_reason="stop", raw={})


def test_healthy_shapes_read_ok_to_shape():
    results = probe_fixed_shapes(ShapeFake(breaks_at=10**9), meter(),
                                 calibration=CAL, shapes=(2048, 4096))
    assert all(r.failure_mode == "ok_to_shape" for r in results)
    assert results[0].max_verified is not None


def test_the_14b_fingerprint_is_caught_per_shape():
    # Small shapes fine; the 8192 shape errors above ~1.8k — exactly
    # what a right-sized ladder cannot see.
    results = probe_fixed_shapes(ShapeFake(breaks_at=3000), meter(),
                                 calibration=CAL, shapes=(2048, 4096, 8192))
    by_shape = {r.shape: r for r in results}
    assert by_shape[2048].failure_mode == "ok_to_shape"
    assert by_shape[4096].failure_mode == "ok_to_shape"
    assert by_shape[8192].failure_mode == "hard_error"
    assert by_shape[8192].max_verified == 8192 // 4  # quarter passed, half broke


def test_budget_death_reports_unmeasured_never_ok():
    dead = BudgetMeter(Budget(max_calls=0, max_prompt_tokens=1))
    results = probe_fixed_shapes(ShapeFake(), dead, calibration=CAL,
                                 shapes=(2048,))
    assert results[0].failure_mode == "unmeasured"
    assert results[0].max_verified is None
