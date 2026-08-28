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
from assay.geometry import (
    Geometry,
    attention_layer_count,
    free_vram_mib,
    kv_bytes_per_token,
    plan_window,
    recurrent_state_bytes,
    serving_block_count,
)

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


# --- R9: MLA, separate widths (H-b) -----------------------------------------
#
# Phase 1 measured H-b directly: 276,480 B/token at all three ctx points for
# the deepseek2 artifact under ollama 0.32.13 (llama runner) — the runtime
# caches K at key_length (head_dim) width and V at value_length width, so
# the K+V-doubling factor in R2/R3 is wrong whenever the two widths differ.


def test_kv_mla_separate_widths():
    # R9: 27 attention layers * 16 kv-heads * (192 key + 128 value) * 2
    # bytes = 276480 — the measured figure, not the dense 2x(192) guess.
    info = make_info(block_count=27, kv_head_count=16, head_dim=192, value_length=128)

    assert kv_bytes_per_token(info) == 276_480


def test_kv_equal_widths_unchanged():
    # v == k is NOT MLA: a stated value_length equal to head_dim must stay
    # on the dense R2/R3 path by identity, not fall into the R9 branch.
    info = make_info(block_count=28, kv_head_count=8, head_dim=128, value_length=128)

    assert kv_bytes_per_token(info) == 2 * 28 * 8 * 128 * 2


def test_kv_dense_path_unchanged():
    # Regression: an unstated value_length is not a claim of equal
    # widths — it is the pre-R9 reading, which the dense formula already
    # covers untouched.
    info = make_info(block_count=28, kv_head_count=8, head_dim=128)

    assert kv_bytes_per_token(info) == 2 * 28 * 8 * 128 * 2


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


# --- hybrid architectures: R3, R4, R6 --------------------------------------
#
# The shape below is the REAP-48 Qwen3.6-35B-A3B one, whose values are
# hardware-verified on two boots in bloomery's turn-5 evidence and frozen
# as the gguf-geometry vectors `qwen3.6-35b-a3b-reap48-ours-q4km` (the
# patched artifact) and `-mtp-trap` (as converted) — sealed in v1 and
# carried byte-identical into the vendored v2 set. The conformance
# module drives those vectors end to end through the extractor; these
# tests pin the arithmetic each rule does, so a break names the rule
# instead of only reddening a vector.

#: The artifact's own `<arch>.ssm.*` dimensions. `time_step_rank` is
#: stated too and is deliberately NOT an input: llama.cpp's state is
#: `n_embd_r + n_embd_s`, which these four size completely.
REAP48_SSM = dict(conv_kernel=4, state_size=128, group_count=16, inner_size=4096)


def make_hybrid_info(**overrides) -> ModelInfo:
    """40 serving blocks, 1 in 4 an attention layer, 30 recurrent."""
    return make_info(**{
        "block_count": 40,
        "kv_head_count": 2,
        "head_dim": 256,
        "training_ctx": 262_144,
        "attention_layer_count": 10,
        "mtp_layer_count": 0,
        "recurrent_state_bytes": 65_863_680,
        **overrides,
    })


def test_kv_charges_the_attention_layers_not_every_block():
    # R3: 2 (K+V) * 10 attention layers * 2 kv-heads * 256 head-dim * 2
    # bytes = 20480 — the figure llama.cpp itself allocates for this
    # artifact (`10 layers`, 2110.00 MiB over 108,032 cells).
    assert kv_bytes_per_token(make_hybrid_info()) == 20_480
    # ...and never the all-40-blocks figure, which is precisely the 4.00x
    # over-charge bloomery measured and fixed in turn 5.
    assert kv_bytes_per_token(make_hybrid_info()) != 81_920


def test_kv_falls_back_to_the_block_count_when_no_attention_count_is_reported():
    # The dense identity AND the compatibility path in one: a ModelInfo
    # built before this field existed leaves it None, every block owns a
    # cache, and the qwen2.5-coder-7b number is unmoved at 2 * 28 * 4 *
    # 128 * 2 = 57344.
    dense = make_info()

    assert dense.attention_layer_count is None
    assert kv_bytes_per_token(dense) == 57_344


@pytest.mark.parametrize(
    ("serving", "interval", "expected"),
    [
        pytest.param(40, 4, 10, id="reap48_hybrid"),
        pytest.param(64, 4, 16, id="one_in_four_of_64"),
        pytest.param(28, None, 28, id="dense_states_no_interval"),
        pytest.param(None, 4, None, id="no_block_count_read"),
    ],
)
def test_attention_layer_count_divides_the_serving_blocks(
    serving, interval, expected
):
    # R3. The dense case is the identity, not a special case.
    assert attention_layer_count(serving, interval) == expected


@pytest.mark.parametrize("interval", [0, -1])
def test_attention_layer_count_refuses_a_nonsense_interval(interval):
    # R8: an interval that cannot be applied is refused, not divided by
    # (ZeroDivisionError) and not silently replaced with the raw count.
    assert attention_layer_count(40, interval) is None


def test_attention_layer_count_refuses_an_interval_larger_than_the_model():
    # 2 // 4 == 0 full-attention layers. Zero is not a smaller answer
    # here, it is a DIFFERENT claim — "this model holds no kv cache" —
    # which downstream reads as an unbounded window or divides by zero.
    assert attention_layer_count(2, 4) is None


@pytest.mark.parametrize(
    ("blocks", "mtp", "expected"),
    [
        pytest.param(41, 1, 40, id="the_mtp_trap"),
        pytest.param(40, 0, 40, id="key_present_and_zero"),
        pytest.param(28, None, 28, id="key_absent"),
        pytest.param(None, 1, None, id="no_block_count_read"),
    ],
)
def test_serving_block_count_excludes_the_mtp_layers(blocks, mtp, expected):
    # R6: `block_count = 40 + 1 = 41` is what convert_hf_to_gguf wrote
    # for a checkpoint carrying 40 blocks of tensors; the 41st is an MTP
    # layer and is not a serving layer.
    assert serving_block_count(blocks, mtp) == expected


def test_recurrent_state_bytes_from_the_ssm_dimensions():
    # R4, llama.cpp's `n_embd_r + n_embd_s` in f32, per recurrent layer:
    #   (conv_kernel-1) * (inner_size + 2*group_count*state_size)
    #     + state_size * inner_size
    #   = 3 * 8192 + 524288 = 548864 elements * 4 = 2195456 B
    # over 30 recurrent layers (40 serving - 10 attention) = 65863680,
    # which is the `RS buffer size = 62.81 MiB` line llama.cpp printed on
    # both hardware-verified boots.
    assert recurrent_state_bytes(30, **REAP48_SSM) == 65_863_680
    assert 65_863_680 == 62.8125 * 1024 * 1024


def test_recurrent_state_is_zero_only_when_the_architecture_states_no_ssm():
    # An architecture with no ssm keys has no recurrent layers to charge:
    # 0 is the measured answer there, and only there.
    assert recurrent_state_bytes(
        0, conv_kernel=None, state_size=None, group_count=None, inner_size=None
    ) == 0


def test_a_partial_ssm_set_is_unmeasured_rather_than_zero():
    # R4: zero ONLY when there are no recurrent layers, never as a
    # default. A file stating SOME ssm keys has recurrent layers whose
    # size this implementation cannot compute; 0 would charge nothing for
    # them and publish that as a measurement.
    partial = dict(REAP48_SSM, inner_size=None)

    assert recurrent_state_bytes(30, **partial) is None


def test_recurrent_state_is_unmeasured_when_the_layer_count_is_unknown():
    assert recurrent_state_bytes(None, **REAP48_SSM) is None


def test_the_recurrent_state_is_charged_against_the_budget():
    # R7: recurrent_state_bytes is a fixed per-context term, taken off
    # the accelerator budget once, before the kv division.
    common = dict(vram_free_mib=2_048, user_cap=None)
    info = make_hybrid_info(loaded=True, training_ctx=1_000_000)

    charged = plan_window(info, **common)
    uncharged = plan_window(replace(info, recurrent_state_bytes=0), **common)

    budget = (2_048 - 512) * MIB
    assert charged.limited_by == uncharged.limited_by == "vram"
    assert uncharged.usable_window == budget // 20_480
    assert charged.usable_window == (budget - 65_863_680) // 20_480
    assert charged.usable_window < uncharged.usable_window


def test_unreported_recurrent_state_drops_out_rather_than_charging_zero():
    # A ModelInfo from a backend that never derived the term — or one
    # built before the field existed — leaves it None. The term drops out
    # of the arithmetic (R7: an unmeasured term is never guessed), and
    # the geometry REPORTS None, so an uncharged window is not readable
    # as a model measured to have no recurrent state.
    info = make_hybrid_info(
        loaded=True, training_ctx=1_000_000, recurrent_state_bytes=None
    )

    geometry = plan_window(info, vram_free_mib=2_048, user_cap=None)

    assert geometry.recurrent_state_bytes is None
    assert geometry.usable_window == ((2_048 - 512) * MIB) // 20_480


def test_the_geometry_reports_the_hybrid_terms_it_measured():
    geometry = plan_window(
        make_hybrid_info(loaded=True), vram_free_mib=100_000, user_cap=None
    )

    assert geometry.attention_layer_count == 10
    assert geometry.serving_block_count == 40
    assert geometry.recurrent_state_bytes == 65_863_680
    assert geometry.kv_kib_per_token == 20


def test_the_mtp_layer_is_not_a_serving_block_in_the_reported_geometry():
    # The trap state as converted: block_count 41, nextn_predict_layers 1.
    geometry = plan_window(
        make_hybrid_info(loaded=True, block_count=41, mtp_layer_count=1),
        vram_free_mib=100_000,
        user_cap=None,
    )

    assert geometry.serving_block_count == 40
    assert geometry.serving_block_count != 41


def test_a_dense_geometry_reports_no_hybrid_terms():
    geometry = plan_window(
        make_info(loaded=True), vram_free_mib=100_000, user_cap=None
    )

    # Nothing was derived about attention intervals or ssm state, so
    # neither is reported as a number.
    assert geometry.attention_layer_count is None
    assert geometry.recurrent_state_bytes is None
    # The serving count is not a hybrid quantity: with no MTP key, every
    # block the file states is a serving block.
    assert geometry.serving_block_count == 28


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


def as_pre_hybrid_fix(info: ModelInfo) -> ModelInfo:
    """``info`` read the way the probe that wrote these figures read it.

    Every kv number committed under `tools-anchor/` and `tier-enthusiast*/`
    predates the 2026-08-27 hybrid-geometry fix (R3/R4/R6): the probe
    charged every RAW block as an attention layer and charged nothing for
    recurrent state. On a model that states no `full_attention_interval`
    and no `ssm.*` keys that is exactly today's derivation, so this is
    the identity there — which is why it is applied to every row rather
    than to a hand-picked one.

    On `qwen3.8:27b` it is not the identity: 65 raw blocks, a stated
    interval of 4 and an MTP layer make 16 attention layers, so the
    published figures carry a 4.0625x attention over-charge on top of the
    E1 head_dim correction, and no recurrent term at all. That second
    defect is filed as an erratum beside the profiles
    (`tier-enthusiast-2026-08/ERRATA.md`, 2026-08-27) and pinned as
    arithmetic in `test_the_anchors_hybrid_row_carries_the_overcharge`.

    On the deepseek2 row (`deepseek-coder-v2:16b-lite-instruct-q5_K_M`)
    there is a third, narrower exception, dated 2026-08-28: its
    committed `geometry.kv_kib_per_token` figure is itself an R9
    re-pin (the measured 270 KiB/token, `docs/superpowers/evidence/
    mla-kv-2026-08-27/`), not the era probe's R2-arithmetic figure
    (324 KiB/token, erratum E3) — so for that row this helper replays
    today's MLA width rule rather than reproducing the figure the era
    probe actually wrote.

    The committed evidence stands as written; what this names is the
    derivation that reproduces it.
    """
    return replace(
        info, attention_layer_count=info.block_count, recurrent_state_bytes=0
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
    # Under the layer derivation that recorded it — see as_pre_hybrid_fix.
    assert kv_bytes_per_token(as_pre_hybrid_fix(info)) == record[
        "kv_bytes_per_token"]


def test_the_anchors_hybrid_row_carries_the_overcharge():
    """The second defect in the anchor's qwen3.8 figures, as arithmetic.

    The anchor was captured to pin the expert fields and the `key_length`
    head_dim, and it does. But one of its two models is a HYBRID, and
    every kv number recorded for it charges all 65 raw blocks — the same
    defect class R3 was written against. Pinned here so the correction is
    a checked number rather than a claim in an errata file, and so a
    future re-measurement has something to disagree with.
    """
    record = next(r for r in anchor_moe() if r["model"] == "qwen3.8:27b")
    info = anchor_backend(record).model_info()

    # What the file states, and what the rules make of it.
    assert info.block_count == record["block_count"] == 65
    assert info.mtp_layer_count == 1          # R6: 65 - 1 = 64 serving
    assert info.attention_layer_count == 16   # R3: 64 // 4
    assert info.recurrent_state_bytes == 156_893_184  # R4: 48 recurrent layers

    # The recorded figure is the all-blocks one; the conforming figure is
    # 4.0625x smaller, and R4's term is missing from the record entirely.
    assert kv_bytes_per_token(as_pre_hybrid_fix(info)) == 266_240
    assert kv_bytes_per_token(info) == 65_536
    assert record["kv_bytes_per_token"] / kv_bytes_per_token(info) == 4.0625
    assert "recurrent_state_bytes" not in record


def test_the_anchor_covers_a_real_moe_and_a_real_non_moe():
    # An anchor that only ever saw one of the two would pin nothing about
    # the distinction it exists to check.
    records = anchor_moe()
    # The flag is `moe_metadata_reported`, not `is_moe`: the daemon told
    # us whether expert keys are PRESENT, which for the second model is
    # "unreported", not "measured dense". `expert_keys_reported` carries
    # the raw fact and the two must agree.
    assert sorted(r["moe_metadata_reported"] for r in records) == [False, True]
    for record in records:
        assert record["moe_metadata_reported"] is bool(
            record["expert_keys_reported"])
    moe = next(r for r in records if r["moe_metadata_reported"])
    unreported = next(r for r in records if not r["moe_metadata_reported"])

    assert moe["architecture"] == "deepseek2"
    assert moe["expert_count"] == 64 and moe["expert_used_count"] == 6
    show = json.loads((ANCHOR / moe["show"]).read_text())
    reported = sorted(k for k in show["model_info"] if "expert" in k)
    assert reported == moe["expert_keys_reported"]
    # The extraction takes the two it names and is not confused by the
    # three siblings sitting beside them.
    assert len(reported) == 5

    # The other model was ASSUMED to be the box's MoE and is not: its
    # metadata carries no expert key at all, which is what makes it the
    # right partner here — unreported, not measured dense.
    assert unreported["architecture"] == "qwen35"
    other = json.loads((ANCHOR / unreported["show"]).read_text())
    assert [k for k in other["model_info"] if "expert" in k] == []


@pytest.mark.parametrize("record", anchor_moe(), ids=lambda r: r["model"])
def test_the_measured_geometry_replays_from_the_committed_metadata(record):
    measured = record["measured_geometry"]
    geometry = plan_window(
        # Under the layer derivation that recorded it — as_pre_hybrid_fix.
        as_pre_hybrid_fix(anchor_backend(record).model_info()),
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

    Every replay below reads the layer geometry the way the probe that
    wrote these profiles did (see `as_pre_hybrid_fix`), so head_dim
    remains the ONE variable this test moves — which is the whole point
    of it, and would be lost if a second correction were folded in
    silently.
    """
    info = as_pre_hybrid_fix(anchor_backend(record).model_info())
    v14 = record["v14_profile"]
    recomputed = record["v16_recomputed_under_v14_conditions"]
    conditions = dict(vram_free_mib=v14["vram_free_mib"], user_cap=None)

    # Against the PROFILE ITSELF, not a copy of its numbers: the erratum
    # is a claim about a committed file, so the file is what it is
    # checked against. Transcribing the figures into results.json and
    # comparing to those would only prove the transcription.
    profile = json.loads((ANCHOR / v14["file"]).read_text())
    assert profile["model"]["name"] == record["model"]
    assert profile["assay_profile_version"] == 4
    for field in ("kv_kib_per_token", "vram_free_mib", "usable_window",
                  "limited_by"):
        assert profile["geometry"][field] == v14[field], field
    # v1.4's schema had no expert keys at all — which is why the erratum
    # is about kv arithmetic and not about MoE metadata going missing.
    assert "expert_count" not in profile["geometry"]

    # `loaded=True` is FORCED by the conditions, not fitted to make the
    # arithmetic land: a v1.4 profile is written mid-run with its own
    # model resident, and unloaded there is nothing to reproduce — the
    # weights alone exceed the free VRAM the profile recorded.
    assert plan_window(replace(info, loaded=False), **conditions
                       ).usable_window == 0

    # The v1.4 profile as committed, reproduced from the derived head_dim.
    derived = plan_window(
        # era-faithful: the pre-R9 ModelInfo this replay simulates had no
        # value_length field at all.
        replace(info, head_dim=record["head_dim_if_derived"], value_length=None,
                loaded=True),
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


@pytest.mark.parametrize("record", anchor_moe(), ids=lambda r: r["model"])
def test_the_erratum_percentages_are_on_the_bases_they_name(record):
    """Two ratios, two bases, and the identity that makes them confusable.

    The published pair is the **window shortfall of the v1.4 promise**
    and the **kv excess over v1.4**. They are not independent: where the
    VRAM term binds, `usable_window` is inversely proportional to kv
    bytes per token, so the shortfall stated the other way round — as
    excess over the TRUE window — is identically the kv excess. Quoting
    one number from each pair is the mistake this test exists to make
    impossible, so both are recomputed here from the integers rather
    than trusted as prose.
    """
    v14 = record["v14_profile"]
    now = record["v16_recomputed_under_v14_conditions"]
    old_bytes = now["kv_bytes_per_token_if_derived"]
    new_bytes = record["kv_bytes_per_token"]

    # The derived-head_dim kv figure is arithmetic, not a second reading.
    assert old_bytes == (
        4 * record["block_count"] * record["kv_head_count"]
        * record["head_dim_if_derived"]
    )

    shortfall = 100 * (v14["usable_window"] - now["usable_window"]) / v14["usable_window"]
    excess = 100 * (new_bytes - old_bytes) / old_bytes
    assert round(shortfall, 1) == now["window_shortfall_pct_of_v14_promise"]
    assert round(excess, 1) == now["kv_excess_pct_over_v14"]
    # The two bases are genuinely different numbers...
    assert shortfall < excess
    # ...and the identity that links them, which is why they get mixed up.
    over_true = 100 * (v14["usable_window"] - now["usable_window"]) / now["usable_window"]
    assert round(over_true, 1) == pytest.approx(round(excess, 1), abs=0.1)


def test_the_erratum_is_filed_beside_the_profiles_it_corrects():
    """Reachability: the correction must be where the wrong numbers are.

    A reader opening `tier-enthusiast/` for a kv or window figure has no
    reason to know the tools anchor exists, so the erratum is filed in
    that directory and names every profile it applies to.
    """
    errata = (ANCHOR / "../tier-enthusiast/ERRATA.md").resolve()
    assert errata.exists()
    text = errata.read_text()
    for record in anchor_moe():
        name = pathlib.Path(record["v14_profile"]["file"]).name
        assert name in text, name
        assert (errata.parent / name).exists()
        now = record["v16_recomputed_under_v14_conditions"]
        assert str(now["usable_window"]) in text
        assert str(now["kv_kib_per_token"]) in text
