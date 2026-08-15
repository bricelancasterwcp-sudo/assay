"""Verdict arithmetic: Wilson intervals, the rung ladder, the stop test.

A leaf module on purpose — it imports nothing from ``assay``. Both
``codecs`` (which stops sampling sequentially) and ``profile`` (which
grades the result) need this arithmetic, and ``profile`` already
imports ``codecs``; keeping the numbers here is what stops that from
becoming a cycle.
"""

READY_THRESHOLD = 0.9
RISKY_THRESHOLD = 0.6
# Interim looks for sequential sampling (spec §1). A cell is examined
# after each look and stops as soon as the interval decides a rung; 35
# is the terminal look, not a promise that 35 calls will be spent.
LOOK_SCHEDULE = (5, 10, 20, 35)


def wilson95(passes: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion. Reported so
    a verdict near a threshold SAYS so: at n=5, 5/5 spans ~[0.57, 1.0] —
    ready and risky are indistinguishable, and pretending otherwise is
    the point-estimate overclaim this project bans elsewhere (external
    review, 2026-08-13)."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    phat = passes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * ((phat * (1 - phat) + z * z / (4 * n)) / n) ** 0.5
    return (max(0.0, (centre - margin) / denom),
            min(1.0, (centre + margin) / denom))


def ladder(lands: float | None, *, ready_blocked: bool = False) -> str:
    if lands is None:
        return "unmeasured"
    if lands >= READY_THRESHOLD and not ready_blocked:
        return "ready"
    if lands >= RISKY_THRESHOLD:
        return "risky"
    return "unusable"


def decided(passes: int, n: int) -> bool:
    """Sequential stop test (spec §1, amended form): the Wilson-95
    interval endpoints ladder to the SAME rung. Exactly the negation
    of the provisional condition."""
    lo, hi = wilson95(passes, n)
    return ladder(lo) == ladder(hi)
