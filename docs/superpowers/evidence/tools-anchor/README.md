# The tools anchor

Live capture behind the v1.6 tool surface, the loop error script and the
MoE geometry metadata. Three results in three lines:

1. **The unsupported path is real.** `gemma2:9b` refused the `tools`
   parameter with `HTTP 400 {"error": "registry.ollama.ai/library/gemma2:9b
   does not support tools"}`, and the classifier's behavior-class bet — 4xx
   whose error text mentions "tool" — read it correctly on the first live
   encounter. `supported=False`, every rate `None`, one call spent.
2. **The instrument's whole thesis showed up in the data.**
   `qwen2.5-coder:7b-instruct-q8_0` wrote a perfectly correct call —
   `{"name": "read_file", "arguments": {"path": "config.yaml"}}` — as
   **plain text**, five times out of five, and emitted **zero** native tool
   calls. `supported=True`, `call_rate=0.0`. A harness gets nothing from
   that model. A probe that measured what models WRITE would have scored it
   full marks.
3. **The doom loop is measured, not hypothesised.** Shown its patch failing
   with "SEARCH text not found", `qwen2.5-coder:7b` re-emitted the identical
   failing SEARCH block **3/3**. `recovery_rate=0.0`, `doom_loop_rate=1.0`.

A fourth result, unplanned: the v1.6 `head_dim` fix (prefer the stated
`attention.key_length` over `embedding_length // head_count`) **changes the
committed v1.4 kv numbers** for both models checked — see
[The MoE check](#the-moe-check-and-what-it-turned-up).

## Provenance

| | |
|---|---|
| Captured | 2026-08-16 |
| Daemon | ollama **0.32.13** at `http://127.0.0.1:11434` |
| Hardware | NVIDIA RTX 5080, 16303 MiB |
| Probes | `assay.tools.probe_tools`, `assay.loop.probe_loop`, `assay.geometry.plan_window` — unmodified |
| Probe version | assay 0.7.0 |
| Instruments | `scripted-tools-v1` / `toolset-v1`; `scripted-loop-v2` |
| Tools seeds | T1 1400-1404, T2 1500-1504 (`_SEED_BASE`, `_SEED_BASE + 100 + i`) |
| Loop seeds | golden 800/801/802, error 850/851/852 (`_SEED_BASE`, `+ _ERROR_SEED_OFFSET`) |
| Temperature | 0.2 (`PROBE_TEMPERATURE`) |
| `num_ctx` | unset — daemon default, exactly as `run.py` calls the probes |
| Models | 4 tools + 1 loop, one resident at a time, `ollama stop` between |
| Calls | 46 live: 31 tools (1 for gemma2, which stopped at the refusal, + 10 each for the other three) and 15 loop |

The probes were driven directly rather than through `assay probe`, wrapped in
`CallRecorder`, because the anchor needs the tools and loop families and
nothing else. Prompts, toolset, seeds, temperature and turn scripts are all
the probes' own. **No tuning and no prompt-shopping**: no model was re-run,
no wording was adjusted, and the one model that turned out to disagree with
its own reputation (`qwen2.5-coder:7b` is advertised `tools`-capable by the
daemon and does speak the parameter) is reported exactly as it measured.

Machine-readable values: `results.json`. Transcripts: `tools-*.jsonl`,
`loop-*.jsonl`, one JSONL row per call in `CallRecorder` format, replayable
through `CallReplayer`. Verbatim `/api/show` bodies: `show-*.json`.

## Tools capture

`probe_tools` — 5 tasks x 2 turns = 10 calls at full health.

| model | supported | call | right tool | args valid | result use | composite | n tasks / turns |
|---|---|---|---|---|---|---|---|
| gemma2:9b | **false** | — | — | — | — | — | 0 / 0 |
| llama3.1:8b | true | 1.00 | 1.00 | 1.00 | **0.40** | 1.00 | 5 / 10 |
| mistral-nemo:latest | true | 1.00 | 1.00 | 1.00 | **0.00** | 1.00 | 5 / 10 |
| qwen2.5-coder:7b-instruct-q8_0 | true | **0.00** | — | — | 0.80 | **0.00** | 5 / 10 |

Read the dashes as `None`, which is what the probe returns and what the
transcripts replay to: gemma2 never got to a scored turn, and
qwen2.5-coder never emitted a call for a tool name or an argument to be
judged against. `right_tool_rate` and `args_valid_rate` are over the T1s
that called ANYTHING — scoring them `0.0` would double-count the miss
`call_rate` already carries.

### gemma2:9b — the unsupported capture

The refusal, verbatim from `tools-gemma2-9b.jsonl`:

```json
{"kind": "chat_tools", "seed": 1400, "outcome": "error",
 "error_type": "ToolsUnsupported",
 "error_raw": {"error": "registry.ollama.ai/library/gemma2:9b does not support tools"}}
```

The daemon agrees with the file: `/api/show` reports `capabilities:
["completion"]` for gemma2:9b — no `tools` — because gemma2's chat template
has no tool section. This is a REAL capability fact off a real endpoint, not
a fake HTTP 400, and it validates two design bets at once: the
behavior-class classifier (4xx + "tool" in the error body, rather than
matching Ollama's exact wording) and the "refusal on the FIRST call ends the
probe" rule, which spent 1 call instead of 10 to learn one thing.

### qwen2.5-coder:7b — supported, and it never calls

All five T1 replies, in full (`tools-qwen2.5-coder-7b-instruct-q8_0.jsonl`,
`tool_calls: []` on every one):

```
{"name": "read_file", "arguments": {"path": "config.yaml"}}
{"name": "run_tests", "arguments": {}}
{"name": "search_docs", "arguments": {"query": "retry_policy"}}
{"name": "read_file", "arguments": {"path": "src/main.py"}}
{"name": "search_docs", "arguments": {"query": "rate limiting"}}
```

Five for five: right tool, right arguments, **in the content channel**. The
endpoint accepted `tools` (so `supported=True` — the capability is real);
the model simply does not use the protocol it was offered. On T2, where the
answer is handed to it in a `role: "tool"` message, it quotes the canary
4/5 — it can READ a tool result fine. It only cannot EMIT a call.

This is the separation the family exists for, stated in `tools.py`'s opening
paragraph and now measured: a model that writes a beautiful call is still
useless to a harness that reads `message.tool_calls`.

### llama3.1:8b and mistral-nemo — the other half of the instrument

Both score a perfect T1 (`composite=1.00`) and then diverge on the half that
asks whether the model can USE a tool result. mistral-nemo scores **0.00**:
handed `Tool read_file completed. Result token: CANARY-1500`, it answers
with an entirely invented file —

```yaml
server:
  host: localhost
  port: 8080

database:
  username: admin
  password: secret
```

— and never mentions the canary on any of the five. llama3.1 quotes it on 2
of 5 (once while inventing a claim around it: "It is set to `CANARY-1502` by
default") and answers from thin air on the other 3. Neither model emitted a
spurious second tool call, so every miss here is the model answering past
the tool result it was given rather than a protocol slip.

Note the ordering this produces: on `call_rate` alone qwen2.5-coder is last
and the other two are tied at the top, while on `result_use_rate` the order
is qwen2.5-coder (0.80) > llama3.1 (0.40) > mistral-nemo (0.00). No single
number ranks these three, which is why the family reports five.

## Loop error-script capture

`probe_loop(runs=3)` against `qwen2.5-coder:7b-instruct-q8_0` — 15 calls
(3 golden runs x 3 turns + 3 error runs x 2 turns).

| metric | value | n |
|---|---|---|
| action_fidelity | 1.00 | 15 turns |
| patch_rate | **0.00** | 3 golden runs |
| finish_rate | 1.00 | 3 golden runs |
| repeat_rate | 0.00 | 15 turns |
| anchor_violations | 0 | — |
| **recovery_rate** | **0.00** | 3 error runs |
| **doom_loop_rate** | **1.00** | 3 error runs |

The error script's second turn hands the model its own failed patch, the
rejection, and the unchanged file. All three replies (seeds 850/851/852) are
byte-identical to each other and to the block they were just shown failing:

```
patch tiny.py
```
```
<<<<<<< SEARCH
subtotal * 1.08
=======
return subtotal * 1.08
>>>>>>> REPLACE
```

That is the canned broken patch verbatim — the real target line with its
indentation stripped, the measured qwen signature the script was built
from. The model re-emits it after being told in plain words that this exact
SEARCH text was not found. `doom_loop_rate=1.0` is the failure that ended
robigo runs, reproduced in three calls.

Two honest caveats on this run: `repeat_rate` is 0.00 because repeats are
counted WITHIN a run and each error run is two different turns — the three
identical replies live in three different runs, so the doom lens is what
catches them, not the repeat lens. And `finish_rate=1.00` beside
`patch_rate=0.00` says the model says "done" when the environment tells it
the tests pass, whether or not its own patch was ever well-formed.

## The MoE check, and what it turned up

The task expected `qwen3.8:27b` to be the box's resident MoE. **It is not**,
per its own metadata: `/api/show` for `qwen3.8:27b` (architecture `qwen35`)
reports **no `expert_*` keys at all** — the extraction correctly returns
`None` for both fields, which is the dense/unreported reading, never `0`.
The genuine MoE on this box is `deepseek-coder-v2:16b-lite-instruct-q5_K_M`
(architecture `deepseek2`), which reports five expert keys. Both were
measured; both `/api/show` bodies are committed as `show-*.json`.

| | deepseek-coder-v2:16b-lite-q5_K_M | qwen3.8:27b |
|---|---|---|
| architecture | `deepseek2` | `qwen35` |
| `expert_count` | **64** | `None` (not reported) |
| `expert_used_count` | **6** | `None` (not reported) |
| other expert keys | `expert_shared_count=2`, `expert_feed_forward_length=1408`, `expert_weights_scale=1` | — |
| `block_count` | 27 | 65 |
| `attention.head_count_kv` | 16 | 4 |
| `head_dim` (from `attention.key_length`) | **192** | **256** |
| `head_dim` if derived (`embedding // heads`) | 128 | 213 |
| kv bytes/token (v1.6) | 331 776 = **324 KiB** | 266 240 = **260 KiB** |
| kv KiB/token in the committed v1.4 profile | **216** | **216** |

`expert_count` and `expert_used_count` resolve non-`None` off the live
daemon for the real MoE, `head_dim` comes from the stated `key_length` on
both, and `plan_window` carries the routing metadata through unchanged while
the kv arithmetic ignores it (experts live in the FFN weights, which the
cache never holds).

### The v1.4 comparison does NOT match, and the fix is why

Both committed v1.4 profiles report `kv_kib_per_token: 216`. That is not a
coincidence of the models — it is the signature of the DERIVATION v1.4 used,
and the two derivations land within 336 bytes of each other by accident
(27x16x128 = 55 296 vs 65x4x213 = 55 380 K-elements per token). Feeding the
derived `head_dim` back through today's `plan_window` reproduces each v1.4
profile's `usable_window` **exactly**, which pins the head_dim source as the
only thing that changed:

| model | head_dim | KiB/token | usable_window under that profile's own VRAM reading |
|---|---|---|---|
| deepseek-coder-v2 | 128 (v1.4 derived) | 216 | **8092** — matches the committed v1.4 profile |
| deepseek-coder-v2 | 192 (v1.6, stated) | 324 | **5394** |
| qwen3.8:27b | 213 (v1.4 derived) | 216 | **4922** — matches the committed v1.4 profile |
| qwen3.8:27b | 256 (v1.6, stated) | 260 | **4096** |

So the v1.6 reading is 50% more cache per token on deepseek-coder-v2 and 20%
more on qwen3.8:27b, and the v1.4 profiles **over-promised** the usable
window by 33% and 20% respectively. The v1.4 numbers are left as committed —
evidence is not rewritten to suit a later fix — and this table is the
erratum. Anyone reading a v1.4 profile's `geometry` block for one of these
architectures should treat its `kv_kib_per_token` and `usable_window` as
optimistic.

Geometry as measured live today (the VRAM term moves with whatever else is
resident, so these are readings, not properties):

| model | loaded | vram_free_mib | usable_window | limited_by |
|---|---|---|---|---|
| deepseek-coder-v2:16b-lite-q5_K_M | true | 1220 | 2237 | vram |
| qwen3.8:27b | false | 13962 | 0 | vram |

The `0` is the residency rule doing its job, not a bug: unloaded, the model's
17.7 GB of weights must come out of a 16 GB card's free VRAM before any
cache fits, so nothing does.

## What the suite pins

Offline acceptance tests read these committed files — no daemon, no GPU, no
network:

- `tests/test_tools.py` — every `tools-*.jsonl` replays through strict
  `CallReplayer` to the `results.json` values, including `supported=False`
  with the refusal body restored on `ToolsUnsupported.raw`.
- `tests/test_loop.py` — `loop-*.jsonl` replays to the recorded
  `recovery_rate` / `doom_loop_rate` / `n_error_runs`.
- `tests/test_geometry.py` — the committed `/api/show` bodies drive
  `OllamaNative.model_info()` and `plan_window`, pinning the expert fields,
  the `key_length` head_dim and the v1.4 erratum arithmetic above.
