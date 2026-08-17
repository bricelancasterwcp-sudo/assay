"""Native tool-calling probe (v1.7): the ``scripted-tools-v2`` instrument.

Every other family in assay measures what a model WRITES. This one
measures what it CALLS — the protocol an agent harness actually runs on,
where a model that writes a beautiful patch is still useless if it
cannot emit one well-formed function call. The two failures are not
correlated by anything we can assume, so they are measured separately.

The instrument is a SCRIPT, not a benchmark. Twenty heterogeneous
tasks, each with one obviously-correct tool and (where the task names a
file or a query verbatim) one exact expected argument value, so argument
checking is mechanical rather than semantic. Two turns per task, and
NOTHING branches on what the model said:

- **T1** — a short system line plus the task's user message, with
  ``TOOLSET`` offered. Scored: exactly one call was emitted; its name is
  the right tool; its arguments are schema-valid and match the pinned
  value. The per-task AND of those three is the ``composite``, which is
  what the verdict ladders on — a model that calls the right tool with
  junk arguments has not done the job.
- **T2** — the same messages plus a CANNED assistant message carrying
  the GOLDEN call (never the model's, so every model is asked the same
  second question) and a ``role: "tool"`` result embedding a seeded
  canary. Scored: the canary comes back in the text AND no further tool
  call is emitted. This is the half that catches a model which can call
  a tool but cannot read the answer.

The pool is twenty rather than v1's five so that a bigger n can come
from more TASKS: at the pinned temperature 0.2, re-running the same five
tasks with fresh seeds measures sampler variance, not capability. Tasks
0–4 are the v1 pool VERBATIM and in order, which is what keeps a
five-task reading comparable across the version boundary and lets the
committed tools-anchor replay byte-identically. The TOOLSET did NOT grow
with the pool — its three schemas are frozen (``toolset-v1``), and it is
the pool that the instrument name carries. The sequential look schedule
that spends the extra fifteen tasks is not wired yet: ``probe_tools``
scores the first five, exactly as v1 did, so today those tasks are
authored and pinned rather than sampled.

Honesty rules, the same ones every probe here follows:

- THE INSTRUMENT ANNOUNCES ITS OWN RUBRIC. ``_SYSTEM`` tells the model
  to call exactly one tool, to use the arguments the request names, and
  to quote the result token verbatim — which is, criterion for
  criterion, what ``call_rate``, ``args_valid_rate`` and
  ``result_use_rate`` score. So these are rates of INSTRUCTED behavior,
  not of spontaneous behavior: ``call_rate`` says "told to call one
  tool, it called one tool", never "it reached for a tool unprompted".
  The rubric is announced on purpose (every model is asked in the same
  words, so nothing is measured except the model), but a reader
  comparing these numbers to an agent harness that does NOT spell the
  rules out should expect this instrument to read high.
- ``supported`` is a three-state fact. ``None`` = never attempted
  (budget died first); ``False`` = the endpoint REFUSED the tools
  parameter, which is a measured capability, not a failure; ``True`` =
  it spoke the protocol at least once.
- A refusal on the FIRST call ends the probe with ``supported=False``
  and every rate None — there is nothing to average, and burning nine
  more calls to collect nine more refusals measures nothing new.
- A refusal AFTER a turn has been scored keeps ``supported=True`` and
  the honest partial: the endpoint demonstrably does speak the protocol,
  and one late refusal (a canned continuation a particular server would
  not accept — see the wire note below) must not erase that.
- ``right_tool_rate`` and ``args_valid_rate`` are over the T1s that
  called ANYTHING. A task where no call was emitted has no tool name and
  no arguments to judge; scoring it zero would double-count the miss
  that ``call_rate`` already carries and disguise "never called" as
  "called badly". Nothing called at all → both are None, never 0.0.
- ``args_valid_rate`` scores the call against the TASK's pinned
  arguments, not against whatever tool the model happened to name: a
  call to the wrong tool cannot carry the right arguments. The
  diagnostic split that matters (picked the wrong tool vs. fumbled the
  arguments) is readable from ``right_tool_rate`` beside it.
- Unreadable arguments (``ToolCall.arguments is None`` — a malformed
  JSON string on the OpenAI wire) score INVALID, never absent. ``{}``
  and None are never conflated: ``{}`` is a call with no arguments,
  which is exactly right for ``run_tests`` and wrong for the rest.
- Every task in ``TASKS`` pins an exact argument value today, and
  equality with a pinned value is strictly stronger than the schema
  check beside it (it implies the keys, the types and the values). The
  schema check is therefore a REDUNDANT BACKSTOP at present, kept
  because it is the whole of the args verdict for any task that pins
  ``None``; ``tests/test_tools.py`` exercises its rules directly rather
  than pretending the probe covers them.
- Budget death stops the probe and yields partial n. Infrastructure
  errors propagate: a call that failed at the transport is not a call
  the model got wrong.

Seeds: ``seed_base + i`` for task i's T1, ``seed_base + 100 + i`` for
its T2. Distinct and deterministic, so a transcript replays exactly and
no two turns of the instrument share a key. The offset clears the whole
pool rather than only the old five: at twenty tasks the T1 seeds run
1400–1419 and the T2 seeds 1500–1519, still without touching.

Meter charge per call: ``max(1, len(tools_key_material(messages,
TOOLSET)) // 4)`` — the SAME canonical string ``assay.replay`` hashes
into the transcript key. Charging and keying cannot drift apart because
they measure one string, not two independently-written serializations.

Two wire notes, recorded because they bound what this probe can claim:

- ``/api/chat`` carries no truncate flag, where ``/api/generate`` does
  (``"truncate": False``, sent by the Ollama backend). So
  ``caps.truncate_control=True`` is a promise about the GENERATE path
  only: a tool payload larger than the context window would be SILENTLY
  truncated rather than refused, and the probe would score a model on a
  question it never fully saw. At this instrument's sizes (~1 KB of
  messages plus schemas per call, against context windows measured in
  tens of KB and up) that is unreachable, so nothing is engineered for
  it; it is written down because it becomes real the moment someone
  grows the toolset.
- The canned T2 assistant call carries ``arguments`` as an OBJECT, the
  Ollama-native shape and the shape ``ToolCall`` normalizes to. A strict
  OpenAI-compat server that requires the JSON-STRING form may reject
  that message with a 4xx naming tools, which the classifier reads as a
  refusal. That is precisely why a late refusal keeps ``supported=True``
  and its partial rather than overwriting a measured capability with
  ``unsupported``.
"""

from __future__ import annotations

from dataclasses import dataclass

from assay.backends.base import Backend, ToolReply, ToolsUnsupported
from assay.budget import BudgetMeter
from assay.errors import BudgetExhausted
from assay.replay import tools_key_material

TOOLS_INSTRUMENT = "scripted-tools-v2"
TOOLSET_NAME = "toolset-v1"

TOOLSET = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read one file and return its contents.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_tests",
        "description": "Run the project's test suite.",
        "parameters": {"type": "object", "properties": {},
                       "required": []}}},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Search the documentation for a term.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
]
"""The registered instrument: three schemas, verbatim and frozen.

Changing a name, a description or a parameter changes what the numbers
mean, so the toolset carries its own version (``TOOLSET_NAME``) into
every profile's lens beside the instrument name.
"""

TASKS = (
    ("Open the file `config.yaml` and show me what is in it.",
     "read_file", {"path": "config.yaml"}),
    ("Check whether the test suite passes right now.",
     "run_tests", {}),
    ("Find what the documentation says about `retry_policy`.",
     "search_docs", {"query": "retry_policy"}),
    ("Read `src/main.py` for me.",
     "read_file", {"path": "src/main.py"}),
    ("Look up `rate limiting` in the docs.",
     "search_docs", {"query": "rate limiting"}),
    # --- v1.7 (scripted-tools-v2): fifteen more, same authoring law ---
    ("Show me the contents of `Makefile`.",
     "read_file", {"path": "Makefile"}),
    ("What does the documentation say about `connection pooling`?",
     "search_docs", {"query": "connection pooling"}),
    ("Are the tests green right now?",
     "run_tests", {}),
    ("I need `docs/CHANGELOG.md` read out.",
     "read_file", {"path": "docs/CHANGELOG.md"}),
    ("Search the docs for `timeout defaults`.",
     "search_docs", {"query": "timeout defaults"}),
    ("What's in `pyproject.toml`?",
     "read_file", {"path": "pyproject.toml"}),
    ("Run the suite and tell me.",
     "run_tests", {}),
    ("I never remember what the docs say about `batch size`, so look it up.",
     "search_docs", {"query": "batch size"}),
    ("Pull up `docker-compose.yml` so I can see the service definitions.",
     "read_file", {"path": "docker-compose.yml"}),
    ("Is there anything in the documentation about `schema migration`?",
     "search_docs", {"query": "schema migration"}),
    ("Verify nothing is broken by running the tests.",
     "run_tests", {}),
    ("Print `.github/workflows/ci.yml` for me.",
     "read_file", {"path": ".github/workflows/ci.yml"}),
    ("The docs must cover `authentication tokens` somewhere; find it.",
     "search_docs", {"query": "authentication tokens"}),
    ("Before I edit it, let me see `src/db/session.py`.",
     "read_file", {"path": "src/db/session.py"}),
    ("Go ahead and run the project's tests.",
     "run_tests", {}),
)
"""(user message, expected tool, expected arguments or None).

Twenty tasks, of which the first five are the v1 pool VERBATIM and in
order — the frozen prefix a five-task reading is scored on.

Heterogeneous on purpose, including in every prefix a look can stop at:
all three tools appear in the first five tasks, in the first ten and in
all twenty. Over the POOL, each tool appears at least four times and the
two argument-taking tools carry fifteen distinct pinned values between
them, while ``run_tests`` takes none at all — so a model that has
learned to emit one memorised call shape scores differently from one
that reads the request. Its five no-argument tasks therefore differ by
PHRASING (imperative, question, indirect) rather than by value, which is
the only axis they have.

Each message names its file or query VERBATIM, in backticks — the
expected value is quoted from the user's own words, which is what makes
argument checking mechanical rather than a judgement call. A new task
that broke that law would silently turn a pinned-argument check into a
guess, so ``tests/test_tools.py`` checks it over the whole pool, and
checks the messages stay ASCII with it: every prompt assay sends is
ASCII today, and a stray dash would put an encoding and tokenization
difference between two tasks that are otherwise asking the same thing.
"""

_SYSTEM = (
    "You are a tool-using assistant. When one of the supplied tools fits "
    "the user's request, call exactly one tool with the arguments the "
    "request names. When a tool result is supplied, answer the user from "
    "it and quote its result token verbatim."
)

_SEED_BASE = 1400
_T2_SEED_OFFSET = 100
# A tool call is a few dozen tokens and T2's answer is one sentence, so
# 256 is headroom rather than a target. It is not free of consequence:
# a model that rambles for 256 tokens before quoting the canary is cut
# off and scores a result-use miss — which is the reading that
# instruction ("quote its result token") deserves.
_MAX_TOKENS = 256
# Charge sizing ONLY, never reported as a measurement — the same rough
# proxy the codec probes use, applied to the same string the transcript
# key hashes.
_CHARS_PER_TOKEN = 4

_SCHEMAS = {
    tool["function"]["name"]: tool["function"]["parameters"]
    for tool in TOOLSET
}
_JSON_TYPES = {"string": str}
"""The JSON Schema types this validator knows. ``TOOLSET`` declares
nothing else; a type outside this map is left unchecked rather than
silently passed off as validated, and adding one to the toolset means
adding it here."""


@dataclass(frozen=True)
class Tools:
    """One tool-calling measurement. Every rate is None when unmeasured."""

    supported: bool | None        # None = never attempted; False = refused
    call_rate: float | None       # exactly-one-call over scored T1s
    right_tool_rate: float | None  # over the T1s that called at all
    args_valid_rate: float | None  # over the T1s that called at all
    result_use_rate: float | None  # canary echoed, no spurious call, over T2s
    composite: float | None       # per-task AND of the three T1 criteria
    n_tasks: int                  # T1 turns scored — the composite's n
    n_turns: int                  # every turn scored, T1 and T2
    # Ambient facts of the readout, counted over the same scored turns
    # and never fed into a rate: a "length" stop is a reply the token
    # ceiling cut off; an unreported stop is a backend that never said.
    # The rubric's reading of a truncated miss stands (see the module
    # docstring) — these exist so a reader weighing a miss can SEE the
    # ceiling beside it. Defaulted so a profile written before they
    # existed parses as None — unmeasured, never a measured 0.
    n_truncated: int | None = None      # scored turns with stop_reason "length"
    n_stop_unreported: int | None = None  # scored turns with stop_reason None


def canary(seed: int) -> str:
    """The token a T2 tool result carries and the reply must echo."""
    return f"CANARY-{seed}"


def t1_messages(task_index: int) -> list[dict]:
    """Turn 1: the system line and the task, nothing else."""
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": TASKS[task_index][0]},
    ]


def t2_messages(task_index: int, seed: int) -> list[dict]:
    """Turn 2: turn 1 plus the CANNED golden call and its canary result.

    A fresh list every time — turn 1's messages are never appended to,
    so the transcript this probe sends for T1 stays exactly what T1 was
    scored on.
    """
    _, name, arguments = TASKS[task_index]
    call_id = f"call_{task_index}"
    return t1_messages(task_index) + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": dict(arguments or {})},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": (
                f"Tool {name} completed. Result token: {canary(seed)}"
            ),
        },
    ]


def _schema_valid(name: str, arguments: dict | None) -> bool:
    """Do these arguments fit the named tool's registered schema?

    Required keys present, no unknown keys, declared types honoured. An
    unregistered tool name and unreadable arguments (None) are both
    invalid: neither can be checked against a schema, and "could not be
    checked" is not "passed".
    """
    parameters = _SCHEMAS.get(name)
    if parameters is None or not isinstance(arguments, dict):
        return False
    properties = parameters["properties"]
    if any(key not in properties for key in arguments):
        return False
    if any(key not in arguments for key in parameters["required"]):
        return False
    for key, value in arguments.items():
        expected = _JSON_TYPES.get(properties[key].get("type"))
        if expected is not None and not isinstance(value, expected):
            return False
    return True


def _score_t1(
    reply: ToolReply, expected_tool: str, expected_args: dict | None
) -> tuple[bool, bool, bool, bool]:
    """(called, exactly one call, right tool, valid arguments).

    When a model emits several calls, the FIRST is the one scored — the
    instrument cannot know which of them it "meant", and the case has
    already failed ``call_rate`` and with it the composite.
    """
    calls = reply.tool_calls
    if not calls:
        return False, False, False, False
    call = calls[0]
    valid = _schema_valid(call.name, call.arguments) and (
        expected_args is None or call.arguments == expected_args
    )
    return True, len(calls) == 1, call.name == expected_tool, valid


def _nothing_scored(supported: bool | None) -> Tools:
    """No turn was scored: every rate is None, and n is 0 — never 0.0."""
    return Tools(
        supported=supported,
        call_rate=None,
        right_tool_rate=None,
        args_valid_rate=None,
        result_use_rate=None,
        composite=None,
        n_tasks=0,
        n_turns=0,
    )


def _ask(
    backend: Backend, meter: BudgetMeter, messages: list[dict], seed: int
) -> ToolReply:
    """Charge for one tool call and make it, in that order."""
    material = tools_key_material(messages, TOOLSET)
    meter.charge(max(1, len(material) // _CHARS_PER_TOKEN))
    return backend.chat_tools(
        messages, TOOLSET, seed=seed, max_tokens=_MAX_TOKENS
    )


def probe_tools(
    backend: Backend, meter: BudgetMeter, *, seed_base: int = _SEED_BASE
) -> Tools:
    """Run the first five scripted tasks, two turns each, and score them.

    Ten calls at full health. Fewer when the endpoint refuses tools (one)
    or the budget runs out mid-run (whatever it paid for), and the
    returned ``n_tasks``/``n_turns`` say which — a short run is reported
    as a short run, not padded with zeros.
    """
    supported: bool | None = None
    scored_t1 = one_call = right_tool = valid_args = composite = called = 0
    scored_t2 = result_used = truncated = unreported = 0

    # Task 2 replaces this constant with the look schedule.
    for index, (_, expected_tool, expected_args) in enumerate(TASKS[:5]):
        t1_seed = seed_base + index
        try:
            reply = _ask(backend, meter, t1_messages(index), t1_seed)
        except BudgetExhausted:
            break
        except ToolsUnsupported:
            if supported is None:
                return _nothing_scored(supported=False)
            break
        supported = True
        scored_t1 += 1
        truncated += reply.stop_reason == "length"
        unreported += reply.stop_reason is None
        did_call, exactly_one, correct, args_ok = _score_t1(
            reply, expected_tool, expected_args
        )
        called += did_call
        one_call += exactly_one
        right_tool += correct
        valid_args += args_ok
        composite += exactly_one and correct and args_ok

        t2_seed = seed_base + _T2_SEED_OFFSET + index
        try:
            reply = _ask(backend, meter, t2_messages(index, t2_seed), t2_seed)
        except BudgetExhausted:
            break
        except ToolsUnsupported:
            break
        scored_t2 += 1
        truncated += reply.stop_reason == "length"
        unreported += reply.stop_reason is None
        result_used += canary(t2_seed) in reply.text and not reply.tool_calls

    if scored_t1 == 0:
        return _nothing_scored(supported)
    return Tools(
        supported=supported,
        call_rate=one_call / scored_t1,
        right_tool_rate=(right_tool / called) if called else None,
        args_valid_rate=(valid_args / called) if called else None,
        result_use_rate=(result_used / scored_t2) if scored_t2 else None,
        composite=composite / scored_t1,
        n_tasks=scored_t1,
        n_turns=scored_t1 + scored_t2,
        n_truncated=truncated,
        n_stop_unreported=unreported,
    )
