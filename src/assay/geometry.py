"""Geometry: kv-cache arithmetic and the window law (spec §4).

`usable_window = min(training_ctx, kv_fit, user_cap)` over the terms
that are measurable, reported with which term bound it. Unmeasured
parts are None and drop out — never a guess that looks like a
measurement. The residency rule: a loaded model's weights are already
outside "free" VRAM; subtracting them again would double-count.

Hybrid architectures (2026-08-27) add three rules to the same
arithmetic, named here as they are named in the gguf-geometry contract
(SPEC.md R3/R4/R6, vectors vendored under tests/data/gguf_geometry_v1):
an MTP layer counted into the block count does not serve (R6), a stated
`full_attention_interval` means only a fraction of the serving layers
owns a kv cache (R3), and the layers that do not are recurrent and
charge a fixed per-context state (R4). Each is a pure function of the
integers the metadata states, so the backend that reads the keys and
this module that spends them cannot drift apart.
"""

import subprocess
from dataclasses import dataclass

from assay.backends.base import ModelInfo

_KIB = 1024
_MIB = 1024 * 1024
#: llama.cpp holds recurrent (conv + ssm) state in f32, whatever the
#: weights are quantized to — the kv cache's own dtype does not apply.
_RECURRENT_STATE_BYTES_PER_ELEMENT = 4
_NVIDIA_SMI_ARGV = [
    "nvidia-smi",
    "--query-gpu=memory.free",
    "--format=csv,noheader,nounits",
]
_NVIDIA_SMI_TIMEOUT_S = 10


@dataclass(frozen=True)
class Geometry:
    kv_kib_per_token: int
    vram_free_mib: int | None  # the reading; None = unmeasured, vram term dropped
    usable_window: int
    limited_by: str  # "training_ctx" | "vram" | "user_cap"
    source: str  # ModelInfo.source
    # MoE routing as reported by the metadata; None on a dense model
    # (which is NOT a 0-expert MoE) and on any backend that cannot read
    # architecture. Defaulted so a geometry written before v1.6 — which
    # has no expert keys at all — still parses, as None, not as a claim.
    expert_count: int | None = None
    expert_used_count: int | None = None
    # Hybrid-architecture terms (R3/R4/R6), reported so a window number
    # can be read back to the layer counts that produced it. None means
    # the run did not derive them — a dense model on a backend that
    # reads metadata still reports its serving block count, and 0
    # recurrent bytes only where the architecture states no ssm keys.
    attention_layer_count: int | None = None
    serving_block_count: int | None = None
    recurrent_state_bytes: int | None = None


def serving_block_count(
    block_count: int | None, mtp_layer_count: int | None
) -> int | None:
    """R6: the blocks that serve tokens, MTP layers excluded.

    A multi-token-prediction layer is counted into ``block_count`` by
    ``convert_hf_to_gguf`` (the REAP-48 prune left
    ``mtp_num_hidden_layers: 1`` in the HF config, so a 40-block
    checkpoint was sized 41) but predicts ahead rather than serving, and
    charging it inflates every downstream layer count. An absent or zero
    key subtracts nothing: absence is not a measurement of zero MTP
    layers, and both read the same way here because there is nothing to
    take off either way.
    """
    if block_count is None:
        return None
    if not mtp_layer_count:
        return block_count
    return block_count - mtp_layer_count


def attention_layer_count(
    serving_blocks: int | None, full_attention_interval: int | None
) -> int | None:
    """R3: the serving layers that own a kv cache.

    An architecture that interleaves attention and recurrent layers
    states its period as ``<arch>.full_attention_interval`` and llama.cpp
    makes layer *i* a full-attention layer iff ``(i+1) % interval == 0``.
    Charging the raw count instead is the defect this rule exists
    against: bloomery charged all 40 blocks of Qwen3.6-35B-A3B where 10
    are attention layers, a measured 4.00x kv over-charge.

    No interval stated is the DENSE IDENTITY, not a special case: every
    serving block owns a cache. An interval that cannot be applied — not
    positive, or larger than the model — returns None rather than 0:
    zero attention layers is not a smaller answer, it is the claim that
    the model holds no kv cache at all, which downstream reads as an
    unbounded window or divides by zero (R8: refuse, never guess).
    """
    if serving_blocks is None:
        return None
    if full_attention_interval is None:
        return serving_blocks
    if full_attention_interval <= 0:
        return None
    layers = serving_blocks // full_attention_interval
    if serving_blocks > 0 and layers == 0:
        return None
    return layers


def recurrent_state_bytes(
    recurrent_layer_count: int | None,
    *,
    conv_kernel: int | None,
    state_size: int | None,
    group_count: int | None,
    inner_size: int | None,
) -> int | None:
    """R4: the recurrent layers' per-context state, in bytes.

    llama.cpp's ``n_embd_r + n_embd_s`` per recurrent layer, held in f32:

        (conv_kernel - 1) * (inner_size + 2 * group_count * state_size)
            + state_size * inner_size

    Independent of the window — it is charged once per context, not per
    token — so it belongs to the budget, never to ``kv_bytes_per_token``.
    (Arithmetic ported from bloomery ``crates/bloomery-core/src/gguf.rs``
    ``resolve_recurrent_state_bytes``, whose figure was verified against
    llama.cpp's own ``RS buffer size = 62.81 MiB`` allocation line.)

    Zero ONLY where the architecture states no ssm keys at all: it has no
    recurrent layers, and that is a measurement. A PARTIAL ssm set
    returns None — the file has recurrent layers whose size cannot be
    computed here — which is the one place this diverges from bloomery,
    which returns 0 for a partial set. 0 there charges nothing for layers
    that exist and publishes it as a measured term; None says so.
    """
    dimensions = (conv_kernel, state_size, group_count, inner_size)
    if recurrent_layer_count is None:
        return None
    if all(dimension is None for dimension in dimensions):
        return 0
    if any(dimension is None for dimension in dimensions):
        return None
    conv_state = (conv_kernel - 1) * (inner_size + 2 * group_count * state_size)
    per_layer = conv_state + state_size * inner_size
    return recurrent_layer_count * per_layer * _RECURRENT_STATE_BYTES_PER_ELEMENT


def kv_bytes_per_token(info: ModelInfo, *, kv_bits: int = 16) -> int | None:
    """2 (K+V) x attention layers x kv_head_count x head_dim x bytes/element.

    None if any architectural part is unreported — kv arithmetic is
    never guessed from model size or name.

    The layer term is ``attention_layer_count`` (R3), falling back to
    ``block_count`` when no attention count was derived: with no
    ``full_attention_interval`` stated, every block owns a cache, so the
    fallback IS the dense answer, and it keeps every ModelInfo built
    before that field existed arithmetically unchanged. A backend that
    reads metadata and cannot apply a stated interval reports neither
    count, so the fallback cannot be reached with a raw hybrid count.

    Expert-invariant BY DESIGN, not by omission: K/V heads are dense in
    MoE architectures — the experts live in the FFN weights, which the
    kv cache never holds — so a routed model pays exactly this per
    token, and the formula takes no expert term. The MoE metadata rides
    in ``Geometry`` because it explains the WEIGHTS footprint, not the
    cache one.
    """
    layers = (
        info.block_count
        if info.attention_layer_count is None
        else info.attention_layer_count
    )
    parts = (layers, info.kv_head_count, info.head_dim)
    if any(part is None for part in parts):
        return None
    return 2 * layers * info.kv_head_count * info.head_dim * (kv_bits // 8)


def free_vram_mib(run=subprocess.run) -> int | None:
    """Free VRAM of GPU 0 in MiB via nvidia-smi; None on ANY failure.

    None means unmeasured (no binary, non-NVIDIA box, parse failure) —
    never 0, which would read as an empty GPU.
    """
    try:
        result = run(
            _NVIDIA_SMI_ARGV,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().splitlines()
    if not lines:
        return None
    try:
        return int(lines[0].strip())
    except ValueError:
        return None


def plan_window(
    info: ModelInfo,
    *,
    vram_free_mib: int | None,
    user_cap: int | None,
    kv_bits: int = 16,
    overhead_mib: int = 512,
) -> Geometry | None:
    """The window law with the residency rule.

    Returns None (geometry unmeasurable; caller records why) when the
    kv arithmetic or training_ctx is unavailable. A missing VRAM
    reading or user_cap just drops that term. The vram term also drops
    when the model is not known-loaded and weights_bytes is None: the
    weights subtraction cannot be computed without guessing.

    ``recurrent_state_bytes`` (R4) is a fixed per-context charge, so it
    comes off the budget once, beside the weights, rather than scaling
    with the window. An unreported one drops out of that arithmetic (R7)
    and is reported as None, never as a measured 0.
    """
    kv_bytes = kv_bytes_per_token(info, kv_bits=kv_bits)
    if kv_bytes is None or info.training_ctx is None:
        return None

    candidates: list[tuple[str, int]] = [("training_ctx", info.training_ctx)]

    if vram_free_mib is not None:
        # Residency rule: loaded weights are already outside "free".
        if info.loaded:
            weights_bytes = 0
        else:
            weights_bytes = info.weights_bytes  # None => term unmeasurable
        if weights_bytes is not None:
            recurrent_bytes = info.recurrent_state_bytes or 0
            kv_budget_bytes = (
                (vram_free_mib - overhead_mib) * _MIB
                - weights_bytes
                - recurrent_bytes
            )
            kv_fit = max(0, kv_budget_bytes // kv_bytes)
            candidates.append(("vram", kv_fit))

    if user_cap is not None:
        candidates.append(("user_cap", user_cap))

    # min over measurable terms; ties break toward the earlier name.
    limited_by, usable_window = min(candidates, key=lambda term: term[1])

    return Geometry(
        kv_kib_per_token=kv_bytes // _KIB,
        vram_free_mib=vram_free_mib,
        usable_window=usable_window,
        limited_by=limited_by,
        source=info.source,
        expert_count=info.expert_count,
        expert_used_count=info.expert_used_count,
        attention_layer_count=info.attention_layer_count,
        serving_block_count=serving_block_count(
            info.block_count, info.mtp_layer_count
        ),
        recurrent_state_bytes=info.recurrent_state_bytes,
    )
