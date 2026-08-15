"""Task 1 tests: the stats leaf module — Wilson, ladder, sequential stop.

``decided`` is the sequential stopping test (spec §1, amended form): a
cell stops early only when BOTH Wilson-95 endpoints ladder to the same
rung, i.e. exactly when the profile layer would NOT call the verdict
provisional. Every interval pinned below was computed with this
module's own ``wilson95`` before being written down.
"""

from assay.stats import (
    LOOK_SCHEDULE,
    READY_THRESHOLD,
    RISKY_THRESHOLD,
    decided,
    ladder,
    wilson95,
)


def test_look_schedule_is_registered():
    assert LOOK_SCHEDULE == (5, 10, 20, 35)


def test_thresholds_match_the_profile_ladder():
    assert (READY_THRESHOLD, RISKY_THRESHOLD) == (0.9, 0.6)


def test_wilson_of_zero_trials_is_the_whole_unit_interval():
    assert wilson95(0, 0) == (0.0, 1.0)


def test_ladder_rungs():
    assert ladder(None) == "unmeasured"
    assert ladder(0.95) == "ready"
    assert ladder(0.9) == "ready"
    assert ladder(0.75) == "risky"
    assert ladder(0.6) == "risky"
    assert ladder(0.2) == "unusable"
    assert ladder(0.95, ready_blocked=True) == "risky"


def test_perfect_5_is_not_decided():
    # wilson95(5,5) == [0.5655, 1.0]: lo ladders unusable, hi ready —
    # three rungs are still live, so five perfect calls decide nothing.
    lo, hi = wilson95(5, 5)
    assert (ladder(lo), ladder(hi)) == ("unusable", "ready")
    assert decided(5, 5) is False


def test_zero_of_5_is_decided_unusable():
    # wilson95(0,5) == [0.0, 0.4345]: both endpoints ladder unusable.
    lo, hi = wilson95(0, 5)
    assert ladder(lo) == ladder(hi) == "unusable"
    assert decided(0, 5) is True


def test_35_of_35_is_decided_ready():
    lo, _ = wilson95(35, 35)
    assert lo >= 0.9  # the documented 0.9011 property
    assert decided(35, 35) is True


def test_decided_risky_stops_too():
    # The amendment's addition: an interval wholly inside [0.6, 0.9)
    # decides risky. 15/20 -> [0.5313, 0.8881]: NOT decided (lo is
    # unusable-side). 26/35 -> [0.5793, 0.8584]: still straddles.
    # 27/35 -> [0.6098, 0.8793]: both endpoints risky, so it lands.
    lo, hi = wilson95(27, 35)
    assert ladder(lo) == ladder(hi) == "risky"
    assert decided(27, 35) is True


def test_straddling_interval_does_not_stop():
    # 26/35 -> [0.5793, 0.8584] straddles the risky threshold: the data
    # cannot tell unusable from risky, so the cell must keep sampling.
    assert decided(26, 35) is False


def test_decided_is_the_negation_of_provisional():
    # profile.py calls a verdict provisional exactly when the endpoints
    # disagree; decided() must be that same predicate, inverted.
    for passes, n in [(0, 5), (3, 5), (5, 5), (7, 10), (10, 10),
                      (15, 20), (20, 20), (26, 35), (27, 35), (35, 35)]:
        lo, hi = wilson95(passes, n)
        provisional = ladder(lo) != ladder(hi)
        assert decided(passes, n) is (not provisional), (passes, n)


def test_profile_reexports_wilson95():
    from assay.profile import wilson95 as w
    assert w(0, 0) == (0.0, 1.0)


def test_profile_ladder_alias_is_the_stats_ladder():
    from assay.profile import _ladder
    assert _ladder is ladder


def test_stats_is_a_leaf_module():
    # codecs.py will import stats while profile.py imports codecs; if
    # stats ever imports back into assay the cycle returns.
    import inspect

    import assay.stats

    source = inspect.getsource(assay.stats)
    assert "import assay" not in source
    assert "from assay" not in source
