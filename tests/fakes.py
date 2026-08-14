"""Scripted fake backends for orchestrator and CLI tests.

No sockets anywhere: every "endpoint" here is a pure in-process script
that answers the real probe prompts (canary, envelope, codecs, JSON)
well enough to drive the full pipeline deterministically.
"""

from assay import fixtures
from assay.backends.base import BackendCaps, ModelInfo, Reply
from assay.codecs import JSON_DIRECTIVE
from assay.errors import InfrastructureError

# The fake's fixed "tokenizer" rate; calibration should measure ~this.
import random as _random

from assay.ceiling import build_filler as _build_filler

# The speed prefill probe sends bare filler built from seed 900; its word
# sequence is deterministic even though its length varies with the
# calibrated chars-per-token, so the first words identify it.
_SPEED_FILLER_PREFIX = _build_filler(_random.Random(900), 8, 3.0)[:20]

CHARS_PER_TOKEN = 4

_FULL_CAPS = BackendCaps(
    reports_counts=True,
    per_request_ctx=True,
    truncate_control=True,
    metadata_access=True,
)


def _search_replace_block(original: str, expected: str) -> str:
    """A correct single-line SEARCH/REPLACE block for a fixture."""
    for old, new in zip(original.split("\n"), expected.split("\n")):
        if old != new:
            return (
                "<<<<<<< SEARCH\n"
                f"{old}\n"
                "=======\n"
                f"{new}\n"
                ">>>>>>> REPLACE"
            )
    raise AssertionError("fixture original and expected do not differ")


class ScriptedBackend:
    """A well-behaved endpoint: counts reported, every probe answered."""

    caps = _FULL_CAPS

    def __init__(self, model: str = "fake-model") -> None:
        self.model = model
        self.calls = 0

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model,
            quant="q8_0",
            weights_bytes=8_100_000_000,
            training_ctx=32768,
            block_count=28,
            kv_head_count=4,
            head_dim=128,
            loaded=True,
            source="api_show",
        )

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        max_tokens: int,
        num_ctx: int | None = None,
    ) -> Reply:
        self.calls += 1
        text = self._reply_text(prompt, seed)
        return Reply(
            text=text,
            tokens_in=max(1, len(prompt) // CHARS_PER_TOKEN),
            tokens_out=max(1, len(text) // CHARS_PER_TOKEN),
            stop_reason="stop",
            raw={
                "scripted": True,
                # Deterministic server timings for the speed probes:
                # decode 16 tok/s, prefill 1024 tok/s.
                "eval_count": 64, "eval_duration": 4_000_000_000,
                "prompt_eval_count": 2048,
                "prompt_eval_duration": 2_000_000_000,
            },
        )

    def _reply_text(self, prompt: str, seed: int) -> str:
        if prompt.startswith("You are repairing one bug"):
            if "Contents of `tiny.py`" in prompt:
                from assay.loop import _fixture
                _, original, expected = _fixture()
                o, e = original.split("\n"), expected.split("\n")
                at = next(i for i, (a, b) in enumerate(zip(o, e)) if a != b)
                return ("patch tiny.py\n```\n<<<<<<< SEARCH\n" + o[at]
                        + "\n=======\n" + e[at] + "\n>>>>>>> REPLACE\n```")
            if "every test passes" in prompt:
                return "done the defect is fixed"
            return "read tiny.py"
        if prompt.startswith("Count upward from one"):
            return "1\n2\n3\n4"
        if prompt.startswith(_SPEED_FILLER_PREFIX):
            return "ok"
        if prompt.startswith("Begin your reply with exactly the word ASSAY-"):
            return f"ASSAY-{seed} acknowledged."
        if prompt.startswith(JSON_DIRECTIVE):
            return '{"name": "apples", "count": 3, "tags": ["fruit"]}'
        if "VERB must be one of" in prompt:
            verb = prompt.rsplit("Verb to use: ", 1)[1].strip()
            number = prompt.split("ARG must be the number ", 1)[1].split(".", 1)[0]
            return f"{verb} {number}"
        for _, _, _, _, original, expected in fixtures.EXPECTED:
            if original in prompt:
                if "<<<<<<< SEARCH" in prompt:
                    return _search_replace_block(original, expected)
                return expected
        raise AssertionError(
            f"scripted backend got an unexpected prompt: {prompt[:80]!r}"
        )


class MetadataFreeBackend(ScriptedBackend):
    """Answers probes but reports no architecture metadata (geometry None)."""

    caps = BackendCaps(
        reports_counts=None,
        per_request_ctx=False,
        truncate_control=False,
        metadata_access=False,
    )

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model,
            quant=None,
            weights_bytes=None,
            training_ctx=None,
            block_count=None,
            kv_head_count=None,
            head_dim=None,
            loaded=None,
            source="openai_models",
        )


class UnreachableBackend:
    """Every touch fails at the transport layer."""

    caps = _FULL_CAPS
    model = "fake-model"

    def model_info(self) -> ModelInfo:
        raise InfrastructureError("transport failure: connection refused (scripted)")

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        max_tokens: int,
        num_ctx: int | None = None,
    ) -> Reply:
        raise InfrastructureError("transport failure: connection refused (scripted)")
