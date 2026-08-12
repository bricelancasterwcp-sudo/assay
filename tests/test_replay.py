"""Record/replay transcripts (plan Task 3, spec §10).

CallRecorder wraps a live backend and writes one JSONL row per call
keyed on (model, prompt, seed). CallReplayer is STRICT: a key the
transcript lacks — or asked for more times than recorded — raises
TranscriptMiss, never falls through to live or to a canned value.
"""

import pytest

from assay.backends.base import Backend, BackendCaps, ModelInfo, Reply
from assay.errors import ContractViolation
from assay.replay import CallRecorder, CallReplayer, TranscriptMiss

CAPS = BackendCaps(
    reports_counts=True,
    per_request_ctx=True,
    truncate_control=True,
    metadata_access=True,
)

MODEL = "qwen2.5-coder:7b-instruct-q8_0"


class ScriptedBackend:
    """Fake inner backend that yields a scripted sequence of outcomes.

    Each script entry is a Reply to return or an Exception to raise,
    consumed in order regardless of the prompt.
    """

    def __init__(self, script):
        self.caps = CAPS
        self.model = MODEL
        self._script = list(script)

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None):
        outcome = self._script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def model_info(self):
        return ModelInfo(
            name=self.model,
            quant=None,
            weights_bytes=None,
            training_ctx=None,
            block_count=None,
            kv_head_count=None,
            head_dim=None,
            loaded=None,
            source="api_show",
        )


def make_reply(text, **overrides) -> Reply:
    fields = {
        "text": text,
        "tokens_in": 12,
        "tokens_out": 3,
        "stop_reason": "stop",
        "raw": {"done": True},
    }
    fields.update(overrides)
    return Reply(**fields)


def test_recorded_replies_replay_verbatim_in_order(tmp_path):
    # Two calls with the SAME (model, prompt, seed) returning different
    # texts must replay in recording order — same key, N rows, N replays.
    path = tmp_path / "transcript.jsonl"
    inner = ScriptedBackend(
        [
            make_reply("first answer", tokens_in=10, tokens_out=2),
            make_reply("second answer", tokens_in=11, tokens_out=4, stop_reason="length"),
        ]
    )
    recorder = CallRecorder(inner, path)
    live_one = recorder.generate("same prompt", seed=7, max_tokens=64)
    live_two = recorder.generate("same prompt", seed=7, max_tokens=64)
    assert live_one.text == "first answer"
    assert live_two.text == "second answer"

    replayer = CallReplayer(path, model=MODEL, caps=CAPS)
    replay_one = replayer.generate("same prompt", seed=7, max_tokens=64)
    replay_two = replayer.generate("same prompt", seed=7, max_tokens=64)

    assert replay_one.text == "first answer"
    assert (replay_one.tokens_in, replay_one.tokens_out) == (10, 2)
    assert replay_one.stop_reason == "stop"
    assert replay_two.text == "second answer"
    assert (replay_two.tokens_in, replay_two.tokens_out) == (11, 4)
    assert replay_two.stop_reason == "length"


def test_miss_raises_transcript_miss(tmp_path):
    path = tmp_path / "transcript.jsonl"
    inner = ScriptedBackend([make_reply("only answer"), make_reply("only answer")])
    recorder = CallRecorder(inner, path)
    recorder.generate("recorded prompt", seed=1, max_tokens=32)
    recorder.generate("recorded prompt", seed=1, max_tokens=32)

    replayer = CallReplayer(path, model=MODEL, caps=CAPS)

    # A prompt the transcript has never seen: strict miss.
    with pytest.raises(TranscriptMiss):
        replayer.generate("unseen prompt", seed=1, max_tokens=32)

    # Recorded twice, asked three times: the third ask is ALSO a miss.
    replayer.generate("recorded prompt", seed=1, max_tokens=32)
    replayer.generate("recorded prompt", seed=1, max_tokens=32)
    with pytest.raises(TranscriptMiss):
        replayer.generate("recorded prompt", seed=1, max_tokens=32)


def test_recorded_contract_violation_replays_as_contract_violation(tmp_path):
    path = tmp_path / "transcript.jsonl"
    inner = ScriptedBackend([ContractViolation("stats-free 200")])
    recorder = CallRecorder(inner, path)

    # Live: the recorder writes an error row AND re-raises.
    with pytest.raises(ContractViolation):
        recorder.generate("bug prompt", seed=3, max_tokens=32)

    replayer = CallReplayer(path, model=MODEL, caps=CAPS)
    with pytest.raises(ContractViolation):
        replayer.generate("bug prompt", seed=3, max_tokens=32)
