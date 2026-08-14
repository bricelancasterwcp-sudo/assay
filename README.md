# assay

A stdlib-only Python library and CLI that measures what a locally-served
LLM endpoint can *actually* do — context geometry, the daemon's real
prompt ceiling, format discipline, and edit-codec landing — and emits a
versioned capability profile that applications consume before trusting
the model with work.

assay measures **instrument fitness, not intelligence**. A model that
lands 0% of search/replace edits is not "dumb"; it is unusable *through
that codec*, and an application that knows this before shipping work to
it can choose another format or another model.




## v0.4 (v1.3): the fixtures are part of the lens — external review applied

An independent review (another Claude session reading the source cold)
found the v1 codec matrix measured **one fixture per cell** — a landing
rate was sampler variance on a single prompt, not codec capability —
exactly this project's own recorded bug class ("hand-written fixtures
have accidental properties," robigo CARRIED-DEBT lesson 4) applied to
itself. v0.4 answers all findings:

- **`codec-fixtures-v2`**: five heterogeneous tasks per grade across
  five defect classes (dropped return, off-by-one, wrong operator,
  inverted guard, wrong variable) on three clean base modules, plus
  five JSON task variants; the fixture-set name travels in the codec
  lens and provenance. An authoring-integrity test compiles every
  fixture and asserts single-line diffs.
- **Wilson intervals and provisional verdicts**: codec-backed verdicts
  carry `interval95` and `provisional: true` whenever the interval
  endpoints ladder to different rungs (5/5 spans ~[0.57, 1.0] — ready
  and risky are indistinguishable at quick-mode n, and the profile now
  says so instead of point-estimating).
- `applies_and_parses(python)`: the landing lens names its language
  assumption. Refusal classification requires a refusal marker AND no
  verb attempt (shape failures containing "can't" no longer misfile).
  README example updated to schema v3.

## v0.3 (v1.2): speed is a capability, tiers are declared

Two new measurements per profile — **decode tok/s** (chat usability)
and **prefill tok/s** (agent usability: agent loops are
prefill-dominated) — taken from the server's own timings where the
backend reports them, with wall-clock fallbacks whose weaker evidence
class is named. Two new verdicts, `chat_speed` and `agent_speed`, judge
them against stated floors that travel in the lens. Profiles also carry
an operator-declared hardware **tier** with a mandatory
emulated/real-hardware marking (`--tier` requires `--emulated` or
`--real-hardware`) — an emulated number can never masquerade as a
real-hardware one. This is the instrument layer for a per-tier consumer
hardware capability matrix.

## v0.2 (v1.1): the lens is part of the verdict

The first live validation measured the same model at 0% and 100% edit
landing under two different instruments — so as of v0.2, every verdict
names its lens (landing definition, presentation, sampler), codec cells
carry **both** landing lenses (`lands` = byte-equality,
`lands_applies` = applies-and-parses), `patch_editing` is judged under
applies-and-parses, consumers can supply their own codec directives
(`--directives`, `CodecDirectives`) so the landing rate predicts their
application's actual prompt shape, and geometry reads the post-load
serving state. Profile schema version is now **2**.

## Why it exists

Three findings, all measured live against real local endpoints, motivated
this instrument (probe designs ported from
[robigo](https://github.com/bricelancasterwcp-sudo/robigo), MIT, same
author):

1. **Silent front-truncation.** Serving layers can silently truncate an
   oversized prompt from the front and return a confident reply about
   whatever survived. Nothing in the response says it happened. assay's
   ceiling probe rides a canary instruction at the front of every probe
   prompt precisely so truncation eats it — the missing canary is the
   detection signal.
2. **The stats-free-200 ceiling class.** An Ollama daemon was measured
   accepting prompts up to ~11.5k tokens and then, past that, returning
   HTTP 200s with plausible text but **no token counts** — a response
   that breaks its own contract while looking healthy. assay treats that
   as `ContractViolation` infrastructure evidence (`missing_stats`), and
   the ceiling probe bisects to where it starts.
3. **The codec landing split.** Different model families land the same
   edit format wildly differently: granite-code:8b landed ≈ 0% of
   SEARCH/REPLACE edits where qwen2.5-coder:7b landed ≥ 90% — on
   identical prompts. Format choice is a per-model measurement, not a
   preference.

## Install

Python 3.12+. **Zero runtime dependencies** (stdlib `urllib`, `json`,
`hashlib`, `subprocess`).

```sh
pip install -e .          # dev install from a checkout
pip install -e .[dev]     # + pytest
```

## Quick start

```sh
# Full suite, quick mode (~90s of inference), against a local Ollama:
assay probe http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0 --quick --json profile.json

# One family at a time:
assay geometry http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0
assay ceiling  http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0
assay envelope http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0
assay codecs   http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0
```

Flags: `--quick | --full`, `--backend ollama|openai` (else auto-detect),
`--json PATH`, `--record PATH` (JSONL call transcript), `--max-calls N`,
`--max-prompt-tokens N`, `--window-cap N`.

Exit codes: `0` profile produced (whatever it says), `2` budget exhausted
before any probe family completed, `4` infrastructure failure before any
measurement.

As a library:

```python
from assay import Budget, probe

profile = probe(
    "http://127.0.0.1:11434",
    "qwen2.5-coder:7b-instruct-q8_0",
    budget=Budget(max_calls=80, max_prompt_tokens=120_000),
    mode="quick",
)
print(profile.to_json())
```

`budget` is a **required** argument: a library consumer burning a user's
GPU time must say how much. There is no silent default. The CLI supplies
documented defaults — quick: 80 calls / 120k prompt tokens; full: 250
calls / 500k — overridable with `--max-calls` / `--max-prompt-tokens`.

## The profile

One versioned JSON document (`assay_profile_version: 1`). Every field is
a measurement, a `None` with a named reason, or provenance.

| Field | What it says |
|---|---|
| `endpoint` | `kind` (`ollama`/`openai`), `base_url`, whether the kind was autodetected |
| `model` | `name`, `quant`, `weights_bytes`, `training_ctx` — as reported, never guessed |
| `geometry` | `kv_kib_per_token`, `vram_free_mib`, `usable_window`, and `limited_by` — **which** term (`training_ctx` / `vram` / `user_cap`) actually bound the window |
| `ceiling` | `max_verified`, `first_failure`, `failure_mode` (`hard_error` / `missing_stats` / `silent_truncation` / `canary_loss` / `none_up_to_cap` / `budget`), plus per-call evidence |
| `envelope` | exact-format fidelity over N one-line probes, with failures classified (`prose` / `shape` / `refusal`) |
| `codecs` | landing rate per codec (`search_replace`, `whole_file`, `json_object`) × size grade (`tiny`, `small`, `medium`) |
| `verdicts` | `structured_extraction`, `patch_editing`, `long_context` — each `ready` / `risky` / `unusable` / `unmeasured` |
| `provenance` | started/finished, mode, seeds, budget granted vs spent, calibration, and `dropped` |

No probe uses grammar/JSON forcing: constrained generation deforms
rather than rejects, so a forced probe measures the constraint, not the
model. assay measures unforced behavior — the number an application can
act on.

## The None-vs-zero rule

**Unmeasured is `None` and named in `provenance.dropped`; measured-and-
zero is `0.0`.** A consumer must always be able to distinguish "assay
could not measure envelope fidelity here" from "the model failed every
probe". No field in the profile defaults to a value that looks like a
measurement, and verdicts computed from unmeasured inputs say
`unmeasured`, never `unusable`.

## Budget and consent

Probes consume the endpoint's GPU time. Every model call is charged
against the explicit `Budget` before it is made; the profile records
spent-vs-granted. If the budget dies mid-run, every unfinished family is
`None` and named in `dropped` — partial results report exactly what was
verified, never more. assay does not probe paid cloud endpoints (v1):
against a metered API those tokens are money.

## Scope honesty

A profile describes **this daemon on this box with this build** — the
serving path, not the model in the abstract. Endpoint identity and probe
version ride in every profile so profiles are comparable, but they are
never silently generalized. VRAM reading is NVIDIA-only in v1
(`nvidia-smi`); elsewhere geometry degrades honestly to `None`.

The test suite runs entirely from scripted fakes and recorded
transcripts — no GPU, no daemon, no sockets:

```sh
PYTHONPATH=src python -m pytest tests/ -q
```

## License

MIT.
