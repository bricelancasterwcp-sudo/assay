"""Geometry: kv-cache arithmetic and the window law (spec §4).

`usable_window = min(training_ctx, kv_fit, user_cap)` over the terms
that are measurable, reported with which term bound it. Unmeasured
parts are None and drop out — never a guess that looks like a
measurement. The residency rule: a loaded model's weights are already
outside "free" VRAM; subtracting them again would double-count.
"""

import subprocess
from dataclasses import dataclass

from assay.backends.base import ModelInfo

_KIB = 1024
_MIB = 1024 * 1024
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


def kv_bytes_per_token(info: ModelInfo, *, kv_bits: int = 16) -> int | None:
    """2 (K+V) x block_count x kv_head_count x head_dim x bytes/element.

    None if any architectural part is unreported — kv arithmetic is
    never guessed from model size or name.
    """
    parts = (info.block_count, info.kv_head_count, info.head_dim)
    if any(part is None for part in parts):
        return None
    return 2 * info.block_count * info.kv_head_count * info.head_dim * (kv_bits // 8)


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
            kv_budget_bytes = (vram_free_mib - overhead_mib) * _MIB - weights_bytes
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
    )
