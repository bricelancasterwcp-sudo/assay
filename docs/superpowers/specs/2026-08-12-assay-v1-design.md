# assay v1 — design

2026-08-12. Approved direction: standalone public repo; Ollama-native +
OpenAI-compatible backends; full probe suite (geometry + ceiling +
envelope + codec).

## §0 Purpose, positioning, non-goals

**assay** is a stdlib-only Python library and CLI that measures what a
locally-served LLM endpoint can *actually* do — context geometry, the
daemon's real prompt ceiling, format discipline, and edit-codec landing —
and emits a versioned capability profile that applications consume
before trusting the model with work.

It measures **instrument fitness, not intelligence**. A model that lands
0% of search/replace edits (measured live on granite-code:8b by robigo)
is not "dumb"; it is unusable *through that codec*, and an application
that knew this before shipping work to it would have chosen another
format or another model. No serving layer or wrapper today ships
measurement of its own effect; robigo's spec §0.1 survey established
that gap, and assay is that measurement extracted into a reusable
instrument.

Ancestry: assay ports probe designs and arithmetic from
[robigo](https://github.com/bricelancasterwcp-sudo/robigo) (MIT, same
author) — the window law, the KV arithmetic, the stage 1–2 probe shapes,
the error taxonomy, and the measurement discipline. It adds what robigo
never had: an empirical ceiling probe (robigo's ~11.5k Ollama daemon
ceiling was found *by hand*), graded-size codec fixtures (robigo's
probes only measured ≤31-token files — a caveat recorded in its own
output), a `json_object` codec, an OpenAI-compatible backend, and a
profile schema designed for third-party consumption.

**Non-goals (v1):**

- Not a server, proxy, or model manager. (That is "option B" — a
  possible successor project that would wrap llama-server with honest
  contracts. assay is the instrument such a layer would embed.)
- No repair-loop or agentic probes. Multi-turn repair measurement is
  robigo stage 4's job and stays there.
- No probing of paid cloud endpoints. Probes burn tokens; against a
  metered API that is money. Deferred, behind an explicit flag, v2 at
  the earliest.
- No grammar-constrained generation anywhere in the probes. Both robigo
  and Black Oxide independently measured that grammar constraints
  *deform rather than reject* — a constrained probe measures the
  constraint, not the model. assay measures the model.
- No community manifest registry in v1 (profile schema is designed so
  one could exist later).
- Not a benchmark of reasoning/coding quality. There are plenty.

## §1 Consumers

1. **Castle VTT** (first application consumer): a "probe my model"
   action in AI settings runs assay against whatever base_url/model the
   user configured, then adapts — output format choice by codec
   landing, prompt caps by measured ceiling, and a user-facing warning
   when the configured model is `unusable` for extraction. (The VTT
   integration is its own spec under direction C; assay's obligation
   ends at a sufficient profile JSON.)
2. **robigo** (eventual): stages 0–2 could later delegate to assay to
   avoid two drifting copies of the same probes. Not attempted in v1;
   robigo is mid-gate and its build order is frozen.
3. **Humans via CLI**: "run `assay` against your setup and paste the
   profile" — a reproducible, budget-capped diagnostic for the
   local-LLM community, and the public artifact that carries the
   findings (the 14× KiB/token spread, the ceiling failure modes).

## §2 Architecture

```
assay/
├── backends/
│   ├── base.py        # Backend protocol, Reply, BackendCaps, errors
│   ├── ollama.py      # native /api/generate + /api/show + /api/tags
│   └── openai_compat.py  # /v1/chat/completions + /v1/models
├── geometry.py        # KV arithmetic, VRAM read, window law
├── ceiling.py         # calibration + doubling/bisection + canary
├── envelope.py        # format-fidelity probe
├── codecs.py          # search_replace / whole_file / json_object probes
├── fixtures/          # graded-size probe fixtures (tiny/small/medium)
├── profile.py         # versioned profile schema, verdicts, render
├── budget.py          # Budget, spend accounting
├── replay.py          # call recording/replay (ported transcript design)
└── cli.py             # assay probe / geometry / ceiling / codecs / envelope
```

Four probe families over two backends, feeding one profile. Every probe
family takes a `Budget` and a seeded RNG, and returns a result object
whose every field is either a measurement or `None` (§11).

Python 3.12+, zero runtime dependencies (stdlib `urllib`, `json`,
`hashlib`, `subprocess` for `nvidia-smi`). Synchronous HTTP, exactly
like robigo; async consumers wrap in `asyncio.to_thread`.

## §3 Backends

One protocol, two implementations.

```python
@dataclass(frozen=True)
class Reply:
    text: str
    tokens_in: int | None    # None = backend does not report; NEVER estimated
    tokens_out: int | None
    stop_reason: str | None  # "stop" | "length" | None (unreported)
    raw: dict                # verbatim response body, for evidence trails

@dataclass(frozen=True)
class BackendCaps:
    reports_counts: bool     # tokens_in/out expected on every reply
    per_request_ctx: bool    # can set context per request (Ollama options.num_ctx)
    truncate_control: bool   # can request hard-fail instead of silent truncation
    metadata_access: bool    # can read model architecture metadata (§4)

class Backend(Protocol):
    caps: BackendCaps
    def generate(self, prompt: str, *, seed: int, max_tokens: int) -> Reply: ...
    def model_info(self) -> ModelInfo: ...   # §4
```

- **OllamaNative** (`/api/generate`): sends `options.seed`,
  `options.num_predict`, top-level `truncate: false` where supported,
  and `options.num_ctx` when a probe needs a specific serving context.
  `caps = (reports_counts=True, per_request_ctx=True,
  truncate_control=True, metadata_access=True)`.
- **OpenAICompat** (`/v1/chat/completions`): bare `{model, messages,
  max_tokens, seed}` (seed passed; honored by llama-server, ignored
  elsewhere — recorded in provenance). Counts from `usage` when
  present; `reports_counts` is determined **empirically during
  calibration** (§5), not assumed.
- **Auto-detect**: if `GET /api/tags` on the same host answers, the
  endpoint is Ollama and the native backend is preferred; otherwise
  OpenAI-compat. Overridable with `--backend`.

**Response-contract validation lives here.** If `caps.reports_counts`
and a 200 reply lacks counts or `stop_reason` (the exact signature of
the Ollama ~11.5k bug: valid-looking content, `"done": false`, no
`prompt_eval_count`), the backend raises `ContractViolation` — an
`InfrastructureError`, never a model result. The ceiling probe catches
it as evidence (§5); everything else propagates it. Error taxonomy:

```
AssayError
├── InfrastructureError      # transport, HTTP 5xx, timeouts
│   └── ContractViolation    # 200 that breaks the response contract
└── (there is no "ModelError")  # a bad/refused/rambling reply is DATA
```

## §4 Geometry

The robigo law, ported: `usable_window = min(training_ctx, kv_fit,
user_cap)`, reported **with which term bound it**.

- KV bytes/token = `2 × block_count × kv_head_count × head_dim ×
  bytes_per_element` (fp16 = 2; kv-8bit halves it).
- Architecture numbers come from `metadata_access`: Ollama's
  `/api/show` `model_info` carries `*.context_length`, `*.block_count`,
  attention head counts and embedding dims for GGUF models — so
  geometry works even against a *remote* Ollama. Weights size comes
  from `/api/tags` (robigo's measured gotcha: `/api/show` has no size).
  Direct GGUF blob parsing (robigo's reader) is ported as a fallback
  for on-box use with a models-dir path.
- Free VRAM: `nvidia-smi --query-gpu=memory.free` (v1 is NVIDIA-only).
  No `nvidia-smi`, or OpenAI-compat with no metadata → `geometry:
  null` plus a `dropped`-style note naming exactly what was
  unavailable. **An unmeasurable geometry is reported as absent, never
  guessed** — the OpenAI-compat backend without metadata yields
  `geometry: null` and the profile says so.

## §5 Ceiling probe (new)

Empirically finds the largest prompt the *daemon serving path* handles
correctly, and names the failure mode past it. This automates the
manual discovery of robigo's Ollama ceiling (≤ ~11.5k fine; ≥ ~11.8k
fails 100% with stats-free 200s) — a bug invisible to any
documentation-driven approach.

**Step 1 — calibration.** Build filler from a fixed word list (seeded,
naturalistic — not one repeated token, which tokenizes
unrepresentatively). Send one small probe (~500 estimated tokens); read
`tokens_in`; derive empirical `chars_per_token`. This kills robigo's
recorded ~45% stage-0 pessimism (its 3-chars/token estimate vs ~6
measured). If counts are unavailable, calibration records
`chars_per_token: null`, a conservative 3.0 is used for sizing, and the
profile marks all ceiling evidence **canary-only** (weaker, stated).

**Step 2 — canary ladder.** For each size (1k tokens, doubling to
`min(training_ctx, user cap, budget)`), N seeds (default 2) of:

```
[instruction, FIRST line: "Begin your reply with exactly the word
ASSAY-<seed>."] + filler(size)
```

The instruction sits at the **front** because Ollama front-truncates —
truncation eats the canary, and a reply that does not begin with it is
the detection signal. Signals per call:

| signal | meaning |
|---|---|
| HTTP error / explicit overflow error | `hard_error` (an honest server) |
| `ContractViolation` | `missing_stats` (the 11.5k class) |
| `tokens_in` < 80% of sent estimate (calibrated) | `silent_truncation` |
| canary absent but `tokens_in` ≈ sent | `attention_loss` — a MODEL result, not a ceiling; recorded, does not stop the ladder |
| canary absent, counts unavailable | `canary_loss` (ambiguous, stated) |

**Step 3 — bisection.** Between last-good and first-bad to ~10%
resolution, within budget (`max_calls` default 12, `max_prompt_tokens`
default 150k — the whole probe is bounded even at 32k contexts).

Output: `{max_verified, first_failure, failure_mode, evidence: [per-call
records], counts_available}`. If nothing fails up to the cap:
`failure_mode: "none_up_to_cap"` with the cap named — *not* a claim
about larger sizes.

## §6 Envelope probe

Ported from robigo stage 1, generalized past robigo's five-verb
specifics: N seeded probes (default 10 quick / 30 full) instruct an
exact one-line reply from a tiny menu ("reply with exactly one line:
`<VERB> <ARG>` where VERB is one of …"). Fidelity = fraction of replies
that are exactly format-valid. Failures are classified (extra prose,
wrong shape, refusal) because a 60%-fidelity-from-chattiness model and
a 60%-from-refusals model need different application responses. No
grammar forcing (§0).

## §7 Codec probes

Ported from robigo stage 2, with graded sizes and a third codec. For
each codec × size grade, N seeded probes (default 5 quick / 10 full)
ask for one specific tiny change against a fixture; **landing** =
assay's applier accepts the reply AND the applied result equals the
expected output exactly (a single trailing-newline difference is
tolerated; nothing else is).

- **Grades:** `tiny` (~30 tokens — robigo-comparable), `small` (~120),
  `medium` (~400). The landing-vs-size curve is the deliverable;
  robigo's tiny-only probes are the recorded caveat this fixes.
- **`search_replace`**: aider's SEARCH/REPLACE block format, applier
  semantics ported from robigo (exact match, whole lines). Chosen for
  the pretraining prior — robigo law 12.
- **`whole_file`**: full-content replacement; landing requires exact
  expected content.
- **`json_object`** (new): given a small extraction task, reply must
  `json.loads`, contain the required keys (one nested), with correct
  types. No `format: json`, no schema forcing — unforced landing is
  the number an application can act on, because forcing *deforms* (§0).
  This is the codec the VTT importer consumes directly.

## §8 Profile schema

One versioned JSON document; every field a measurement, a `None` with a
named reason, or provenance.

```jsonc
{
  "assay_profile_version": 1,
  "probe_version": "0.1.0",
  "endpoint": {"kind": "ollama", "base_url": "http://127.0.0.1:11434", "autodetected": true},
  "model": {"name": "qwen2.5-coder:7b-instruct-q8_0", "quant": "q8_0",
             "weights_bytes": 8100000000, "training_ctx": 32768},
  "geometry": {                       // or null, with reason in dropped[]
    "kv_kib_per_token": 56, "vram_free_mib": 14558,
    "usable_window": 32768, "limited_by": "training_ctx",
    "source": "api_show"
  },
  "ceiling": {
    "max_verified": 11500, "first_failure": 11800,
    "failure_mode": "missing_stats", "counts_available": true,
    "evidence": [ /* per-call: size, seed, signal, raw excerpt */ ]
  },
  "envelope": {"fidelity": 0.97, "n": 30,
                "failures": {"prose": 1, "shape": 0, "refusal": 0}},
  "codecs": {
    "search_replace": {"tiny": {"lands": 1.0, "n": 10},
                        "small": {"lands": 0.9, "n": 10},
                        "medium": {"lands": 0.6, "n": 10}},
    "whole_file":     { /* same shape */ },
    "json_object":    { /* same shape */ }
  },
  "verdicts": {
    "structured_extraction": "ready",   // ready | risky | unusable | unmeasured
    "patch_editing": "ready",           // max(sr, wf).small = 0.9 → ready per §8 rule
    "long_context": "risky"             // max_verified 11500 w/ missing_stats → risky
  },
  "provenance": {
    "started": "2026-08-12T21:00:00Z", "finished": "...",
    "mode": "full", "seeds": [0,1,2],
    "budget": {"max_calls": 200, "max_prompt_tokens": 400000},
    "spent": {"calls": 143, "prompt_tokens": 210344},
    "calibration": {"chars_per_token": 5.9},
    "dropped": ["geometry: no nvidia-smi on PATH"]
  }
}
```

**Verdict rules (v1 defaults — applications may ignore them and read
raw numbers):**

- `structured_extraction`: `ready` if `json_object.small.lands ≥ 0.9`
  and ceiling failure mode is not `silent_truncation` below 4k;
  `risky` if ≥ 0.6; else `unusable`. `unmeasured` when the inputs are
  `None`.
- `patch_editing`: same thresholds on
  `max(search_replace, whole_file).small`.
- `long_context`: `ready` if `max_verified ≥ 16k` with an honest
  failure mode (`hard_error`/`none_up_to_cap`); `risky` if the mode is
  `silent_truncation`/`missing_stats` (the daemon lies past the edge);
  `unmeasured` otherwise.

None-vs-zero rule, ported verbatim from robigo: **unmeasured is `None`
and named in `provenance.dropped`; measured-and-zero is `0.0`**. A
consumer must be able to distinguish "assay could not measure envelope
fidelity here" from "the model failed every probe".

`render_table()` produces the human view (same spirit as robigo's).

## §9 CLI and library

```
assay probe http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0
      [--quick | --full] [--backend ollama|openai] [--json out.json]
      [--max-calls N] [--max-prompt-tokens N] [--window-cap N]
assay geometry ... / assay ceiling ... / assay envelope ... / assay codecs ...
```

`--quick` ≈ under ~90s of inference (calibration, geometry, short
ceiling ladder, reduced-N probes); `--full` ≈ minutes (full seeds, full
ladder). Exit codes: `0` profile produced (whatever it says), `2`
budget exhausted before any family completed, `4` infrastructure — the
robigo taxonomy, minus the model-outcome codes assay does not have.

Library: `assay.probe(base_url, model, *, budget: Budget, mode:
"quick"|"full") -> Profile`. **`budget` is a required argument** — a
library consumer burning a user's GPU time must say how much, no
silent default (CLI supplies its own documented defaults).

## §10 Replay and testing

- `replay.py` ports robigo's transcript design: `CallRecorder` wraps a
  backend, one JSONL row per call keyed on `(model, prompt, seed)`
  (NUL-separated SHA-256, same as robigo); `CallReplayer` is **strict**
  — a miss raises, never falls through to live (robigo's spec 5.3
  reasoning, kept verbatim).
- The entire probe suite must run green from fakes and recorded
  transcripts with **no GPU and no daemon**. Live probing is a smoke
  path behind `ASSAY_LIVE=1`.
- Every test is mutation-tested before it counts: a test that passes
  with the code deleted or inverted is a defect. The recurring robigo
  bug class — **a value that looks like a measurement but is not** (a
  window reported past what was verified, a landing rate that measured
  the prompt, a fidelity computed over zero probes) — is the named
  review item for every probe family.
- Determinism: fixed seeds throughout; `random.Random(seed)` only, no
  ambient RNG; wall-clock only in provenance.

## §11 Boundaries, risks, honesty notes

- **NVIDIA-only VRAM** in v1; elsewhere geometry degrades honestly
  (§4). Apple/ROCm are v2 candidates.
- **OpenAI-compat variance**: servers that omit `usage` reduce ceiling
  evidence to canary-only; `reports_counts` is measured, not assumed,
  and the profile states which evidence class it got.
- **Seeds over `/v1`** are honored by llama-server but ignored by some
  servers; provenance records that determinism could not be verified
  when replies differ across identical seeded calls (which is itself a
  one-call check during calibration).
- **Probe cost consent**: probes consume the user's GPU time; budget is
  mandatory in library mode (§9) and every profile records spent vs
  granted.
- **Single-box generality**: like robigo's three biases, assay's
  numbers describe *this daemon on this box with this build*; the
  profile carries endpoint identity and probe version so profiles are
  comparable but never silently generalized.

## §12 Success criteria (falsifiable, before any VTT integration)

1. `assay probe --quick` against qwen2.5-coder:7b-instruct-q8_0 on this
   box completes within budget and reproduces, blind: geometry with a
   named binding term; the daemon ceiling detected near ~11.5k with
   `failure_mode: missing_stats`; codec landings measured at all three
   grades.
2. Family split reproduction: granite-code:8b `search_replace.tiny`
   lands ≈ 0%, qwen ≥ 90% — matching robigo's independent stage-2
   measurements without consulting them.
3. Full test suite green with no GPU, under 60 seconds.
4. A written check that the profile JSON alone is sufficient for the
   VTT's three planned adaptations (format choice, prompt caps, user
   warning) — the VTT-side spec is separate, but sufficiency is
   assay's exit gate.

Criterion 1–2 live runs are scheduled **after** the robigo gate run
releases the GPU; implementation starts on fakes/replay regardless.

## §13 Deferred, with rulings

- **Cloud probing behind `--allow-paid` + token budget** — deferred to
  v2; ruled out of v1 because unmetered-vs-metered is a consent
  boundary, not a technical one.
- **Community manifest registry** (profiles-by-family database) —
  deferred; schema versioning exists so it can happen without breaking
  changes.
- **Serving-layer proxy ("option B")** — separate project if assay
  earns it; assay stays embeddable either way.
- **robigo delegating stages 0–2 to assay** — only after robigo's gate
  outcome settles what robigo is; two implementations may coexist
  briefly, and that is acceptable drift for now (noted against the
  one-definition rule, deliberately).
- **Deformation telemetry** (how hard grammar forcing steered) —
  belongs to the serving layer, where the sampler is visible; assay
  measures unforced behavior only.
