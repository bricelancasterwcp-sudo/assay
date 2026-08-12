"""assay backends: one protocol, two implementations (spec §3)."""

from assay.backends.base import (
    Backend,
    BackendCaps,
    ModelInfo,
    Reply,
    validate_reply,
)

__all__ = [
    "Backend",
    "BackendCaps",
    "ModelInfo",
    "Reply",
    "validate_reply",
]
