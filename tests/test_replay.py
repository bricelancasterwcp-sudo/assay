"""Record/replay transcripts (plan Task 3, spec §10).

CallRecorder wraps a live backend and writes one JSONL row per call
keyed on (model, prompt, seed) — on (model, canonical messages+tools,
seed) for tool calls. CallReplayer is STRICT: a key the transcript
lacks — or asked for more times than recorded — raises TranscriptMiss,
never falls through to live or to a canned value.
"""

import json
import pathlib
import threading

import pytest

from assay.backends.base import (
    BackendCaps,
    ModelInfo,
    Reply,
    ToolCall,
    ToolReply,
    ToolsUnsupported,
)
from assay.errors import ContractViolation, InfrastructureError
from assay.replay import (
    CallRecorder,
    CallReplayer,
    TranscriptMiss,
    key_for,
    tools_key_material,
)

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

    def chat_tools(self, messages, tools, *, seed, max_tokens):
        # Same script, same rule: entries are consumed in call order, so a
        # test that mixes generate and chat_tools still reads top to bottom.
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


# --- v1.6: the tool surface -------------------------------------------------
#
# A tool call is not a prompt: its identity is the whole (messages, tools)
# payload. Rows carry `kind` so a transcript can hold both call shapes and
# neither can answer for the other.

MESSAGES = [
    {"role": "system", "content": "You are a tool-using assistant."},
    {"role": "user", "content": "Open the file `config.yaml`."},
]
TOOLSET = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def make_tool_reply(calls=(), text="", **overrides) -> ToolReply:
    fields = {
        "text": text,
        "tool_calls": tuple(calls),
        "tokens_in": 40,
        "tokens_out": 9,
        "stop_reason": "stop",
        "raw": {"done": True},
    }
    fields.update(overrides)
    return ToolReply(**fields)


def read_rows(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_recorded_tool_replies_replay_verbatim_in_order(tmp_path):
    # Same (messages, tools, seed) twice, two different outcomes: the
    # transcript replays them in recording order, calls and all.
    path = tmp_path / "tools.jsonl"
    first = make_tool_reply([ToolCall("read_file", {"path": "config.yaml"})])
    second = make_tool_reply(
        [ToolCall("read_file", None), ToolCall("run_tests", {})],
        text="on it",
        tokens_in=41,
        tokens_out=12,
        stop_reason="length",
    )
    recorder = CallRecorder(ScriptedBackend([first, second]), path)
    assert recorder.chat_tools(MESSAGES, TOOLSET, seed=5, max_tokens=256) == first
    assert recorder.chat_tools(MESSAGES, TOOLSET, seed=5, max_tokens=256) == second

    replayer = CallReplayer(path, model=MODEL, caps=CAPS)
    one = replayer.chat_tools(MESSAGES, TOOLSET, seed=5, max_tokens=256)
    two = replayer.chat_tools(MESSAGES, TOOLSET, seed=5, max_tokens=256)

    assert one.tool_calls == (ToolCall("read_file", {"path": "config.yaml"}),)
    assert (one.text, one.tokens_in, one.tokens_out, one.stop_reason) == (
        "", 40, 9, "stop")
    # None arguments (unreadable on the wire) survive the round trip as
    # None — never as {}, which would read as "called with no arguments".
    assert two.tool_calls == (ToolCall("read_file", None), ToolCall("run_tests", {}))
    assert two.tool_calls[0].arguments is None
    assert two.tool_calls[1].arguments == {}
    assert (two.text, two.tokens_in, two.tokens_out, two.stop_reason) == (
        "on it", 41, 12, "length")
    assert one.raw["replayed"] is True


def test_a_tool_call_miss_raises_transcript_miss(tmp_path):
    path = tmp_path / "tools.jsonl"
    recorder = CallRecorder(ScriptedBackend([make_tool_reply()]), path)
    recorder.chat_tools(MESSAGES, TOOLSET, seed=5, max_tokens=256)

    replayer = CallReplayer(path, model=MODEL, caps=CAPS)
    other = [{"role": "user", "content": "something else entirely"}]
    with pytest.raises(TranscriptMiss):
        replayer.chat_tools(other, TOOLSET, seed=5, max_tokens=256)
    # A different toolset is a different call, even with the same messages.
    with pytest.raises(TranscriptMiss):
        replayer.chat_tools(MESSAGES, [], seed=5, max_tokens=256)
    # Recorded once, asked twice: the second ask is a miss.
    replayer.chat_tools(MESSAGES, TOOLSET, seed=5, max_tokens=256)
    with pytest.raises(TranscriptMiss):
        replayer.chat_tools(MESSAGES, TOOLSET, seed=5, max_tokens=256)


def test_key_material_ignores_dict_key_order(tmp_path):
    # The identity of a call is its CONTENT, not the order a caller happened
    # to build its dicts in. Recording and replaying the same payload with
    # the keys written in a different order must hit the same row.
    path = tmp_path / "tools.jsonl"
    recorder = CallRecorder(ScriptedBackend([make_tool_reply(text="hit")]), path)
    recorder.chat_tools(
        [{"role": "user", "content": "ping"}],
        [{"type": "function", "function": {"name": "run_tests"}}],
        seed=2,
        max_tokens=64,
    )

    replayer = CallReplayer(path, model=MODEL, caps=CAPS)
    reply = replayer.chat_tools(
        [{"content": "ping", "role": "user"}],
        [{"function": {"name": "run_tests"}, "type": "function"}],
        seed=2,
        max_tokens=64,
    )
    assert reply.text == "hit"


def test_recorded_tools_unsupported_replays_as_tools_unsupported(tmp_path):
    # An endpoint refusing the tools parameter is a capability FACT, so it
    # has to survive a transcript: the replay raises the same type, which
    # is deliberately not an AssayError.
    path = tmp_path / "tools.jsonl"
    recorder = CallRecorder(ScriptedBackend([ToolsUnsupported("no tools here")]), path)
    with pytest.raises(ToolsUnsupported):
        recorder.chat_tools(MESSAGES, TOOLSET, seed=5, max_tokens=256)

    row = read_rows(path)[0]
    assert (row["kind"], row["outcome"], row["error_type"]) == (
        "chat_tools", "error", "ToolsUnsupported")

    replayer = CallReplayer(path, model=MODEL, caps=CAPS)
    with pytest.raises(ToolsUnsupported) as caught:
        replayer.chat_tools(MESSAGES, TOOLSET, seed=5, max_tokens=256)
    assert type(caught.value) is ToolsUnsupported


def test_a_refusal_body_survives_the_transcript(tmp_path):
    # The refusal body is the PRIMARY SOURCE behind tools_supported=False.
    # A transcript that dropped it would leave assay asserting its own
    # classification with the evidence thrown away.
    path = tmp_path / "tools.jsonl"
    body = {"error": "qwen2.5-coder:7b does not support tools"}
    refusal = ToolsUnsupported("HTTP 400: endpoint refused the tools parameter")
    refusal.raw = body
    recorder = CallRecorder(ScriptedBackend([refusal]), path)
    with pytest.raises(ToolsUnsupported):
        recorder.chat_tools(MESSAGES, TOOLSET, seed=5, max_tokens=256)

    assert read_rows(path)[0]["error_raw"] == body

    replayer = CallReplayer(path, model=MODEL, caps=CAPS)
    with pytest.raises(ToolsUnsupported) as caught:
        replayer.chat_tools(MESSAGES, TOOLSET, seed=5, max_tokens=256)
    assert caught.value.raw == body


def test_a_generate_error_body_survives_the_transcript_too(tmp_path):
    # Symmetric, not a tools-only column: the 11.5k-bug body that makes a
    # ContractViolation legible has to replay as well.
    path = tmp_path / "generate.jsonl"
    body = {"model": MODEL, "response": "an answer", "done": True}
    violation = ContractViolation("promised counts, delivered none")
    violation.raw = body
    recorder = CallRecorder(ScriptedBackend([violation]), path)
    with pytest.raises(ContractViolation):
        recorder.generate("bug prompt", seed=3, max_tokens=32)

    assert read_rows(path)[0]["error_raw"] == body

    replayer = CallReplayer(path, model=MODEL, caps=CAPS)
    with pytest.raises(ContractViolation) as caught:
        replayer.generate("bug prompt", seed=3, max_tokens=32)
    assert caught.value.raw == body


def test_an_error_that_carries_no_body_records_none(tmp_path):
    # A transport failure never had a body. None means "no body", and is
    # never softened into {} — which would read as "the server said
    # nothing", a different fact.
    path = tmp_path / "generate.jsonl"
    recorder = CallRecorder(ScriptedBackend([InfrastructureError("refused")]), path)
    with pytest.raises(InfrastructureError):
        recorder.generate("a prompt", seed=1, max_tokens=32)

    assert read_rows(path)[0]["error_raw"] is None

    replayer = CallReplayer(path, model=MODEL, caps=CAPS)
    with pytest.raises(InfrastructureError) as caught:
        replayer.generate("a prompt", seed=1, max_tokens=32)
    assert caught.value.raw is None


def test_an_error_row_without_the_column_replays_with_no_body(tmp_path):
    # v1.5 rows have neither `kind` nor `error_raw`. Absent means "not
    # recorded", which replays as None — never as a fabricated body.
    path = tmp_path / "v15.jsonl"
    path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "key": key_for(MODEL, "old prompt", 9),
                "seed": 9,
                "outcome": "error",
                "text": None,
                "tokens_in": None,
                "tokens_out": None,
                "stop_reason": None,
                "error_type": "ContractViolation",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    replayer = CallReplayer(path, model=MODEL, caps=CAPS)
    with pytest.raises(ContractViolation) as caught:
        replayer.generate("old prompt", seed=9, max_tokens=32)
    assert caught.value.raw is None


def test_an_unrecordable_exception_escapes_with_no_row_written(tmp_path):
    # The recorded-error tuple is narrow ON PURPOSE: a bug in assay is not
    # a measurement, and a row claiming the endpoint erred would be a lie
    # about what happened.
    path = tmp_path / "tools.jsonl"
    recorder = CallRecorder(ScriptedBackend([ValueError("a bug in assay")]), path)
    with pytest.raises(ValueError):
        recorder.chat_tools(MESSAGES, TOOLSET, seed=5, max_tokens=256)
    assert read_rows(path) == []

    other = tmp_path / "generate.jsonl"
    recorder = CallRecorder(ScriptedBackend([ValueError("a bug in assay")]), other)
    with pytest.raises(ValueError):
        recorder.generate("a prompt", seed=5, max_tokens=32)
    assert read_rows(other) == []


def test_rows_under_one_key_replay_in_file_order_with_no_lookahead(tmp_path):
    # Two kinds under one key (constructible, since a generate prompt can
    # BE the canonical tool payload): the replayer reads the head of the
    # queue and never scans past it for a row of the kind being asked for.
    path = tmp_path / "interleaved.jsonl"
    canonical = tools_key_material(MESSAGES, TOOLSET)
    recorder = CallRecorder(
        ScriptedBackend([make_tool_reply(text="tool row"), make_reply("plain row")]),
        path,
    )
    recorder.chat_tools(MESSAGES, TOOLSET, seed=8, max_tokens=32)
    recorder.generate(canonical, seed=8, max_tokens=32)
    assert len({row["key"] for row in read_rows(path)}) == 1  # one key, two kinds

    replayer = CallReplayer(path, model=MODEL, caps=CAPS)
    # The generate row is IN the queue, second. Asking generate first is
    # still a miss: file order rules, and the tool row stays put.
    with pytest.raises(TranscriptMiss):
        replayer.generate(canonical, seed=8, max_tokens=32)
    assert replayer.chat_tools(MESSAGES, TOOLSET, seed=8, max_tokens=32).text == (
        "tool row")
    assert replayer.generate(canonical, seed=8, max_tokens=32).text == "plain row"


def test_recorded_contract_violation_on_a_tool_call_replays_as_itself(tmp_path):
    path = tmp_path / "tools.jsonl"
    recorder = CallRecorder(ScriptedBackend([ContractViolation("stats-free 200")]), path)
    with pytest.raises(ContractViolation):
        recorder.chat_tools(MESSAGES, TOOLSET, seed=5, max_tokens=256)

    replayer = CallReplayer(path, model=MODEL, caps=CAPS)
    with pytest.raises(ContractViolation):
        replayer.chat_tools(MESSAGES, TOOLSET, seed=5, max_tokens=256)


def test_rows_are_tagged_with_the_call_shape_that_made_them(tmp_path):
    path = tmp_path / "mixed.jsonl"
    recorder = CallRecorder(
        ScriptedBackend(
            [
                make_reply("prose"),
                make_tool_reply([ToolCall("read_file", {"path": "a.txt"})]),
            ]
        ),
        path,
    )
    recorder.generate("a prompt", seed=1, max_tokens=32)
    recorder.chat_tools(MESSAGES, TOOLSET, seed=1, max_tokens=32)

    generate_row, tools_row = read_rows(path)
    assert generate_row["kind"] == "generate"
    assert "tool_calls" not in generate_row  # a generate row has no calls to carry
    assert tools_row["kind"] == "chat_tools"
    assert tools_row["tool_calls"] == [{"name": "read_file",
                                        "arguments": {"path": "a.txt"}}]
    assert tools_row["key"] == key_for(
        MODEL, tools_key_material(MESSAGES, TOOLSET), 1)


def test_neither_call_shape_can_be_answered_by_the_other(tmp_path):
    # The adversarial case: a generate call whose prompt IS the canonical
    # tool payload collides on the key by construction. `kind` is what
    # keeps a tool row from answering it — and the row stays unconsumed.
    path = tmp_path / "mixed.jsonl"
    canonical = tools_key_material(MESSAGES, TOOLSET)
    recorder = CallRecorder(
        ScriptedBackend([make_tool_reply(text="tool row"), make_reply("plain row")]),
        path,
    )
    recorder.chat_tools(MESSAGES, TOOLSET, seed=4, max_tokens=32)
    recorder.generate(canonical, seed=6, max_tokens=32)

    replayer = CallReplayer(path, model=MODEL, caps=CAPS)
    # A generate call landing on the tool row's key.
    with pytest.raises(TranscriptMiss):
        replayer.generate(canonical, seed=4, max_tokens=32)
    # And a tool call landing on the generate row's key.
    with pytest.raises(TranscriptMiss):
        replayer.chat_tools(MESSAGES, TOOLSET, seed=6, max_tokens=32)
    # Neither miss consumed the row it refused to answer with.
    assert replayer.chat_tools(MESSAGES, TOOLSET, seed=4, max_tokens=32).text == (
        "tool row")
    assert replayer.generate(canonical, seed=6, max_tokens=32).text == "plain row"


def test_a_row_without_a_kind_replays_as_generate(tmp_path):
    # v1.5 wrote rows with no `kind` at all. Missing means "generate", so
    # transcripts recorded before the tool surface existed still replay.
    path = tmp_path / "v15.jsonl"
    path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "key": key_for(MODEL, "old prompt", 9),
                "seed": 9,
                "outcome": "reply",
                "text": "an answer from v1.5",
                "tokens_in": 5,
                "tokens_out": 4,
                "stop_reason": "stop",
                "error_type": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    replayer = CallReplayer(path, model=MODEL, caps=CAPS)
    assert replayer.generate("old prompt", seed=9, max_tokens=32).text == (
        "an answer from v1.5")


def test_the_committed_v15_anchor_transcript_still_replays():
    # Pinned against real committed evidence, not a fixture: the degeneracy
    # anchor was captured under v1.5 and its rows carry no `kind`. If the
    # tool surface ever made `kind` mandatory, this fails.
    from assay.long_output import _PROMPT  # the prompt that recorded them

    anchor = pathlib.Path(__file__).resolve().parents[1] / (
        "docs/superpowers/evidence/degenerate-anchor/gemma2-9b-longoutput.jsonl")
    rows = read_rows(anchor)
    assert rows and all("kind" not in row for row in rows)

    replayer = CallReplayer(anchor, model=rows[0]["model"], caps=CAPS)
    for row in rows:
        replayed = replayer.generate(_PROMPT, seed=row["seed"], max_tokens=4096)
        assert replayed.text == row["text"]


def test_call_recorder_keeps_every_row_whole_under_concurrent_writers(tmp_path):
    """CARRIED-DEBT item 101: the concurrency case the suite lacked.

    `CallRecorder` takes a write lock, and the parallel family creates
    exactly the configuration that would need one — k threads recording
    into a single recorder. This test pins the property that matters to
    a consumer: under concurrent writers every row is present and every
    row is whole, because a half-written row is not a smaller
    transcript, it is an unparseable one, and the transcript IS the
    evidence.

    What this test does NOT establish, measured rather than assumed:
    the lock is not load-bearing here. `_write_row` does
    open-append/write-one-row/close, and on Linux `write()` to a
    regular file is atomic per-inode while `O_APPEND` makes the offset
    update atomic, so rows cannot interleave with or without the lock —
    confirmed at 100 B, 8 KB and 64 KB payloads, 8 threads, all six
    configurations clean. The guard therefore earns its place against a
    FUTURE write path (a long-lived handle, or a row assembled across
    several writes), not against today's. The lock stays for the same
    reason: it costs nothing and the property it protects is one a
    refactor could silently take away.

    The inner fake is deliberately STATELESS, so a missing or corrupt
    row could only be the recorder's doing.
    """
    class _AlwaysReplies:
        caps = CAPS
        model = MODEL

        def generate(self, prompt, *, seed, max_tokens, num_ctx=None):
            return make_reply(f"reply to {prompt}")

    path = tmp_path / "transcript.jsonl"
    recorder = CallRecorder(_AlwaysReplies(), path)
    n_threads, per_thread = 8, 25

    def _write(worker):
        for i in range(per_thread):
            recorder.generate(f"w{worker} c{i}", seed=worker * 100 + i,
                              max_tokens=16)

    threads = [threading.Thread(target=_write, args=(w,))
               for w in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == n_threads * per_thread
    for row in rows:
        json.loads(row)  # every row whole, not merely present
