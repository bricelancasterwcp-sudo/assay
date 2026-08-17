"""Geometry invariants (plan Task 6, spec §4).

The window law: usable_window = min(training_ctx, kv_fit, user_cap)
over the terms that are measurable, reported with which term bound it.
The named review item everywhere: a value that looks like a measurement
but is not — unmeasured is None and drops out, never a guess.
"""

import json
import pathlib
from dataclasses import replace

import pytest

from assay.backends.base import ModelInfo
from assay.backends.ollama import OllamaNative
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


def make_moe_info(**overrides) -> ModelInfo:
    """The same attention geometry, plus MoE routing metadata."""
    return make_info(**{"expert_count": 128, "expert_used_count": 8, **overrides})


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


def test_kv_arithmetic_is_expert_invariant():
    # The formula is unchanged for MoE BY DESIGN, not by omission: K/V
    # heads are dense in these architectures — the experts live in the
    # FFN weights, which the kv cache never holds. Identical attention
    # geometry must cost identically per token whether the model routes
    # or not; scaling this by expert count would invent a cost the
    # hardware does not pay.
    assert kv_bytes_per_token(make_moe_info()) == kv_bytes_per_token(make_info())
    assert kv_bytes_per_token(make_moe_info(expert_used_count=1)) == 57_344


# --- MoE metadata rides along with the geometry ----------------------------


def test_plan_window_carries_the_expert_metadata_it_measured():
    geometry = plan_window(
        make_moe_info(loaded=True), vram_free_mib=100_000, user_cap=None
    )

    assert geometry.expert_count == 128
    assert geometry.expert_used_count == 8
    # ...and the window law is untouched by the routing metadata.
    assert geometry.usable_window == 32_768
    assert geometry.limited_by == "training_ctx"


def test_dense_model_geometry_keeps_the_expert_fields_none():
    # None-vs-zero: a dense model is not a 0-expert MoE.
    geometry = plan_window(
        make_info(loaded=True), vram_free_mib=100_000, user_cap=None
    )

    assert geometry.expert_count is None
    assert geometry.expert_used_count is None


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


# --- the live anchor (plan Task 10) ----------------------------------------
#
# The expert fields and the `key_length` head_dim are claims about what a
# real daemon reports, so they are pinned to what one really did: the
# verbatim /api/show bodies captured 2026-08-16 from ollama 0.32.13, and the
# /api/tags entries read in the same breath, committed under
# docs/superpowers/evidence/tools-anchor/. No daemon, no GPU, no network —
# the committed bytes drive the real extraction.

ANCHOR = pathlib.Path(__file__).resolve().parents[1] / (
    "docs/superpowers/evidence/tools-anchor")


def anchor_moe() -> list[dict]:
    results = json.loads((ANCHOR / "results.json").read_text())
    return results["moe"]["models"]


def anchor_backend(record: dict) -> OllamaNative:
    """`OllamaNative` served the committed bodies for this model.

    /api/show is the capture, verbatim. /api/tags replays the entry read
    from the same daemon at the same time (recorded in results.json), and
    /api/ps replays the residency the run recorded. Nothing here is
    invented: every byte was read off the live endpoint, and the wire
    SHAPES are pinned independently in tests/test_backend_ollama.py.
    """
    show = json.loads((ANCHOR / record["show"]).read_text())
    loaded = record["measured_geometry"]["loaded"]
    entry = record["tags_entry"]

    def http_post(url, payload):
        assert url.endswith("/api/show") and payload == {"model": record["model"]}
        return 200, show

    def http_get(url):
        if url.endswith("/api/tags"):
            return 200, {"models": [entry]}
        assert url.endswith("/api/ps")
        return 200, {"models": [entry] if loaded else []}

    return OllamaNative(
        "http://anchor", record["model"], http_post=http_post, http_get=http_get
    )


@pytest.mark.parametrize("record", anchor_moe(), ids=lambda r: r["model"])
def test_the_committed_show_body_extracts_what_the_anchor_recorded(record):
    info = anchor_backend(record).model_info()

    assert info.quant == record["quant"]
    assert info.training_ctx == record["training_ctx"]
    assert info.block_count == record["block_count"]
    assert info.kv_head_count == record["kv_head_count"]
    assert info.head_dim == record["head_dim"]
    assert info.weights_bytes == record["tags_entry"]["size"]
    assert info.loaded is record["measured_geometry"]["loaded"]
    # None-vs-zero, off real metadata: the MoE reports its routing, the
    # model with no expert keys reports None for both — never 0, which
    # downstream would read as a measured routing fact.
    assert info.expert_count == record["expert_count"]
    assert info.expert_used_count == record["expert_used_count"]
    assert kv_bytes_per_token(info) == record["kv_bytes_per_token"]


def test_the_anchor_covers_a_real_moe_and_a_real_non_moe():
    # An anchor that only ever saw one of the two would pin nothing about
    # the distinction it exists to check.
    records = anchor_moe()
    assert [r["is_moe"] for r in records] == [True, False]

    moe = records[0]
    assert moe["architecture"] == "deepseek2"
    assert moe["expert_count"] == 64 and moe["expert_used_count"] == 6
    show = json.loads((ANCHOR / moe["show"]).read_text())
    reported = sorted(k for k in show["model_info"] if "expert" in k)
    assert reported == moe["expert_keys_reported"]
    # The extraction takes the two it names and is not confused by the
    # three siblings sitting beside them.
    assert len(reported) == 5

    dense = records[1]
    assert dense["architecture"] == "qwen35"
    other = json.loads((ANCHOR / dense["show"]).read_text())
    assert [k for k in other["model_info"] if "expert" in k] == []


@pytest.mark.parametrize("record", anchor_moe(), ids=lambda r: r["model"])
def test_the_measured_geometry_replays_from_the_committed_metadata(record):
    measured = record["measured_geometry"]
    geometry = plan_window(
        anchor_backend(record).model_info(),
        vram_free_mib=measured["vram_free_mib"],
        user_cap=None,
    )

    assert geometry.kv_kib_per_token == record["kv_kib_per_token"]
    assert geometry.usable_window == measured["usable_window"]
    assert geometry.limited_by == measured["limited_by"]
    # The routing metadata rides through the window law untouched.
    assert geometry.expert_count == record["expert_count"]
    assert geometry.expert_used_count == record["expert_used_count"]


@pytest.mark.parametrize("record", anchor_moe(), ids=lambda r: r["model"])
def test_the_v14_kv_numbers_were_the_derived_head_dim_and_are_now_superseded(
    record
):
    """The erratum, held to arithmetic rather than to a footnote.

    Both committed v1.4 profiles report 216 KiB/token. That is not a
    property of the models — it is the signature of the DERIVATION v1.4
    used (`embedding_length // head_count`), which the v1.6 reading
    replaces with the STATED `attention.key_length`. Feeding the derived
    head_dim back through today's window law reproduces each v1.4
    profile's `usable_window` exactly, which is what pins the head_dim
    source as the only thing that changed — and what stops anyone
    "fixing" the discrepancy by editing committed evidence.
    """
    info = anchor_backend(record).model_info()
    v14 = record["v14_profile"]
    recomputed = record["v16_recomputed_under_v14_conditions"]
    conditions = dict(vram_free_mib=v14["vram_free_mib"], user_cap=None)

    # The v1.4 profile as committed, reproduced from the derived head_dim.
    derived = plan_window(
        replace(info, head_dim=record["head_dim_if_derived"], loaded=True),
        **conditions,
    )
    assert derived.kv_kib_per_token == v14["kv_kib_per_token"] == 216
    assert derived.usable_window == v14["usable_window"]
    assert derived.limited_by == v14["limited_by"]

    # ...and the stated key_length costs MORE cache per token, so the
    # window v1.4 promised was optimistic. The old profile stands as
    # written: evidence is not rewritten to suit a later fix.
    current = plan_window(replace(info, loaded=True), **conditions)
    assert current.kv_kib_per_token == recomputed["kv_kib_per_token"]
    assert current.usable_window == recomputed["usable_window"]
    assert current.kv_kib_per_token > v14["kv_kib_per_token"]
    assert current.usable_window < v14["usable_window"]
