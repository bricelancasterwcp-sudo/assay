"""Geometry invariants (plan Task 6, spec §4).

The window law: usable_window = min(training_ctx, kv_fit, user_cap)
over the terms that are measurable, reported with which term bound it.
The named review item everywhere: a value that looks like a measurement
but is not — unmeasured is None and drops out, never a guess.
"""

import pytest

from assay.backends.base import ModelInfo
from assay.geometry import Geometry, free_vram_mib, kv_bytes_per_token, plan_window

MIB = 1024 * 1024


def make_info(**overrides) -> ModelInfo:
    """A fully-measured qwen2.5-coder-7b-shaped ModelInfo; override to break parts."""
    fields = dict(
        name="qwen2.5-coder:7b-instruct-q8_0",
        quant="q8_0",
        weights_bytes=8_100 * MIB,
        training_ctx=32_768,
        block_count=28,
        kv_head_count=4,
        head_dim=128,
        loaded=False,
        source="api_show",
    )
    fields.update(overrides)
    return ModelInfo(**fields)


# --- kv arithmetic ---------------------------------------------------------


def test_qwen_arithmetic_is_56_kib_per_token():
    # 2 (K+V) * 28 blocks * 4 kv-heads * 128 head-dim * 2 bytes = 57344.
    info = make_info()

    assert kv_bytes_per_token(info) == 57_344
    assert kv_bytes_per_token(info) // 1024 == 56


def test_kv8_halves_it():
    assert kv_bytes_per_token(make_info(), kv_bits=8) == 28_672


@pytest.mark.parametrize("missing", ["block_count", "kv_head_count", "head_dim"])
def test_kv_bytes_is_none_when_any_part_is_none(missing):
    assert kv_bytes_per_token(make_info(**{missing: None})) is None


# --- the residency rule ----------------------------------------------------


def test_loaded_model_does_not_double_subtract_weights():
    # A loaded model's weights are already outside "free": subtracting
    # them again double-counts. Constructed so vram binds in both cases.
    common = dict(vram_free_mib=10_000, user_cap=None)

    loaded = plan_window(make_info(training_ctx=1_000_000, loaded=True), **common)
    unloaded = plan_window(make_info(training_ctx=1_000_000, loaded=False), **common)

    assert loaded.limited_by == "vram"
    assert unloaded.limited_by == "vram"
    assert loaded.usable_window > unloaded.usable_window


# --- limited_by names the binding term ------------------------------------


@pytest.mark.parametrize(
    ("info", "vram_free_mib", "user_cap", "expect_window", "expect_term"),
    [
        pytest.param(
            make_info(training_ctx=8_192, loaded=True),
            100_000,
            1_000_000,
            8_192,
            "training_ctx",
            id="training_ctx_binds",
        ),
        pytest.param(
            # loaded, 1024 MiB free - 512 overhead = 512 MiB for kv:
            # 512 * 1024 KiB / 56 KiB = 9362 tokens < ctx < cap.
            make_info(training_ctx=32_768, loaded=True),
            1_024,
            1_000_000,
            9_362,
            "vram",
            id="vram_binds",
        ),
        pytest.param(
            make_info(training_ctx=32_768, loaded=True),
            100_000,
            4_096,
            4_096,
            "user_cap",
            id="user_cap_binds",
        ),
    ],
)
def test_limited_by_names_the_actual_binding_term(
    info, vram_free_mib, user_cap, expect_window, expect_term
):
    geometry = plan_window(info, vram_free_mib=vram_free_mib, user_cap=user_cap)

    assert geometry.usable_window == expect_window
    assert geometry.limited_by == expect_term


def test_tie_breaks_toward_earlier_term():
    # training_ctx == user_cap, both binding: training_ctx names the tie.
    geometry = plan_window(
        make_info(training_ctx=8_192, loaded=True),
        vram_free_mib=100_000,
        user_cap=8_192,
    )

    assert geometry.usable_window == 8_192
    assert geometry.limited_by == "training_ctx"


# --- None-vs-zero ----------------------------------------------------------


def test_unmeasurable_parts_yield_none_not_guesses():
    # Missing kv arithmetic => the whole geometry is unmeasurable.
    assert plan_window(make_info(kv_head_count=None), vram_free_mib=8_000, user_cap=None) is None
    # Missing training_ctx => unmeasurable too.
    assert plan_window(make_info(training_ctx=None), vram_free_mib=8_000, user_cap=None) is None

    # Missing VRAM reading only => geometry exists, vram is recorded as
    # None and is NOT a candidate term.
    geometry = plan_window(make_info(training_ctx=8_192), vram_free_mib=None, user_cap=100_000)

    assert isinstance(geometry, Geometry)
    assert geometry.vram_free_mib is None
    assert geometry.usable_window == 8_192
    assert geometry.limited_by == "training_ctx"
    assert geometry.source == "api_show"


def test_unloaded_model_with_unknown_weights_drops_vram_term():
    # loaded=False and weights_bytes=None: the vram term cannot be
    # computed without guessing the weights — it drops out rather than
    # being treated as zero-weight (which would inflate kv_fit).
    geometry = plan_window(
        make_info(weights_bytes=None, loaded=False, training_ctx=32_768),
        vram_free_mib=1_024,  # would bind hard if the term were computed
        user_cap=None,
    )

    assert geometry.usable_window == 32_768
    assert geometry.limited_by == "training_ctx"


# --- free VRAM reading -----------------------------------------------------


class FakeCompleted:
    def __init__(self, returncode: int, stdout: str):
        self.returncode = returncode
        self.stdout = stdout


def test_free_vram_parses_nvidia_smi_output():
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return FakeCompleted(0, "8192\n")

    assert free_vram_mib(run=fake_run) == 8_192
    assert seen["argv"][0] == "nvidia-smi"
    assert "--query-gpu=memory.free" in seen["argv"]


@pytest.mark.parametrize(
    "fake_run",
    [
        pytest.param(lambda argv, **kw: (_ for _ in ()).throw(FileNotFoundError()), id="no_binary"),
        pytest.param(lambda argv, **kw: FakeCompleted(1, ""), id="nonzero_exit"),
        pytest.param(lambda argv, **kw: FakeCompleted(0, "not a number\n"), id="garbage"),
        pytest.param(lambda argv, **kw: FakeCompleted(0, ""), id="empty"),
    ],
)
def test_free_vram_is_none_on_any_failure(fake_run):
    # None, never 0: an unreadable GPU is unmeasured, not empty.
    assert free_vram_mib(run=fake_run) is None
