"""Scripted fake backends for orchestrator and CLI tests.

No sockets anywhere: every "endpoint" here is a pure in-process script
that answers the real probe prompts (canary, envelope, codecs, JSON)
well enough to drive the full pipeline deterministically.
"""

from assay import fixtures
from assay.backends.base import (
    BackendCaps,
    ModelInfo,
    Reply,
    ToolCall,
    ToolReply,
    ToolsUnsupported,
)
from assay.codecs import JSON_DIRECTIVE
from assay.errors import InfrastructureError
from assay.tools import TASKS as _TOOL_TASKS

# The fake's fixed "tokenizer" rate; calibration should measure ~this.
import random as _random

from assay.ceiling import build_filler as _build_filler

# The speed prefill probe sends bare filler built from seed 900; its word
# sequence is deterministic even though its length varies with the
# calibrated chars-per-token, so the first words identify it.
_SPEED_FILLER_PREFIX = _build_filler(_random.Random(900), 8, 3.0)[:20]

CHARS_PER_TOKEN = 4

# The long-output probe (v1.5) asks for an enumeration; a well-behaved
# endpoint answers with varied, non-repeating lines. Healthy on both
# degeneracy metrics, which is what makes it the "well-behaved" script.
_ENUMERATION = (
    "1. A cast iron skillet holds heat because the metal is thick, not conductive.\n"
    "2. Runways carry the number of their magnetic heading, so drift forces renaming.\n"
    "3. Nutmeg and mace grow on one tree: the seed and the lacy aril around it.\n"
    "4. Venetian canals sit on pilings of alder, which hardens rather than rots underwater.\n"
    "5. Bicycle wheels stay true through spoke tension, not through rim stiffness.\n"
    "6. Vanilla orchids outside Mexico need hand pollination; their bee never travelled.\n"
    "7. Library book spines were once shelved inward, with titles written on the fore edge.\n"
    "8. Fireflies time their flashes into species-specific codes to avoid wooing strangers.\n"
    "9. The Dead Sea keeps swimmers afloat on dissolved magnesium and potassium salts.\n"
    "10. Cathedral flying buttresses move thrust outward so the walls can carry glass.\n"
)

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
        if prompt.startswith("Write a numbered list of distinct"):
            return _ENUMERATION
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


class CodecFailingBackend(ScriptedBackend):
    """Answers every family EXCEPT codecs, where nothing ever lands.

    A cell of 0/5 is DECIDED unusable at the first look (Wilson-95 on
    0/5 is [0.0, 0.434] — both ends unusable), so under a look schedule
    every cell honestly stops at n=5. That is the case a fixed-n
    orchestrator misreads as budget death.
    """

    def _reply_text(self, prompt: str, seed: int) -> str:
        if prompt.startswith("You are repairing one bug"):
            return super()._reply_text(prompt, seed)  # the loop family
        if prompt.startswith(JSON_DIRECTIVE):
            return "sorry, I would rather describe it in prose."
        for _, _, _, _, original, _expected in fixtures.EXPECTED:
            if original in prompt:
                return "sorry, I would rather describe it in prose."
        return super()._reply_text(prompt, seed)


class LongOutputDegradingBackend(ScriptedBackend):
    """Healthy enumeration at short targets, a looping phrase at long ones.

    The failure the rung ladder exists to locate: a model that holds
    together for 512 tokens and collapses at 2048. Every other family is
    answered exactly as ScriptedBackend answers it, so a profile built
    on this backend differs from the well-behaved one in the long_output
    family alone.
    """

    def __init__(self, degrade_at: int = 2048, model: str = "fake-model") -> None:
        super().__init__(model)
        self.degrade_at = degrade_at
        self._target: int | None = None

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        max_tokens: int,
        num_ctx: int | None = None,
    ) -> Reply:
        # The rung IS max_tokens: _reply_text only sees the prompt, which
        # is identical at every rung, so the target is stashed here.
        self._target = max_tokens
        return super().generate(prompt, seed=seed, max_tokens=max_tokens,
                                num_ctx=num_ctx)

    def _reply_text(self, prompt: str, seed: int) -> str:
        if (
            prompt.startswith("Write a numbered list of distinct")
            and self._target is not None
            and self._target >= self.degrade_at
        ):
            return "the same line over and over again. " * 60
        return super()._reply_text(prompt, seed)


class LongOutputTerseBackend(ScriptedBackend):
    """Answers the long-output prompt with three words at every rung.

    Nothing scorable comes back (the n-gram window needs four), so every
    rung lands degenerate=None: calls were spent and NOTHING was
    measured — the case a naive "no rung flagged, so it is healthy"
    reading turns into a false clean bill.
    """

    def _reply_text(self, prompt: str, seed: int) -> str:
        if prompt.startswith("Write a numbered list of distinct"):
            return "no thanks"
        return super()._reply_text(prompt, seed)


def _tools_turn(messages: list[dict]) -> tuple[int, str | None]:
    """(which scripted task, the tool result) for a scripted-tools transcript.

    The tool result is None on turn 1 and the canary-carrying string on
    turn 2, so it identifies the turn as well as supplying its content.
    An unscripted task raises, the same way an unscripted prompt does:
    a fake that quietly answers a question it was never taught would let
    a probe change its instrument without a single test noticing.
    """
    user = next((m["content"] for m in messages if m["role"] == "user"), None)
    index = next(
        (i for i, task in enumerate(_TOOL_TASKS) if task[0] == user), None
    )
    if index is None:
        raise AssertionError(
            f"scripted tools backend got an unexpected task: {user!r}"
        )
    return index, next(
        (m["content"] for m in messages if m["role"] == "tool"), None
    )


class ScriptedToolsBackend(ScriptedBackend):
    """A well-behaved endpoint that also speaks the native tool protocol.

    Answers the scripted-tools-v1 transcript the way a model that has
    fully got it would: turn 1 emits exactly the golden call, turn 2
    answers in prose quoting the tool result (canary and all) and calls
    nothing further. Keyed on the SCRIPT — which task, which turn — never
    on model output, because in a scripted probe there is none to branch
    on.
    """

    def chat_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        seed: int,
        max_tokens: int,
    ) -> ToolReply:
        self.calls += 1
        index, result = _tools_turn(messages)
        _, name, arguments = _TOOL_TASKS[index]
        if result is None:
            text, calls = "", (ToolCall(name=name, arguments=dict(arguments)),)
        else:
            text, calls = f"The {name} call returned: {result}", ()
        return ToolReply(
            text=text,
            tool_calls=calls,
            tokens_in=max(
                1,
                sum(len(m["content"]) for m in messages) // CHARS_PER_TOKEN,
            ),
            tokens_out=max(1, len(text) // CHARS_PER_TOKEN),
            stop_reason="stop",
            raw={"scripted": True},
        )


class ToolsUnsupportedBackend(ScriptedToolsBackend):
    """Answers every text probe; refuses the tools parameter.

    The capability fact in fake form — an endpoint whose model has no
    tool template. The refusal carries the endpoint's own error body,
    because that body is the primary source assay classified from.
    """

    def chat_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        seed: int,
        max_tokens: int,
    ) -> ToolReply:
        self.calls += 1
        refusal = ToolsUnsupported(
            "HTTP 400 from /api/chat: endpoint refused the tools parameter"
        )
        refusal.raw = {"error": f"{self.model} does not support tools"}
        raise refusal


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
