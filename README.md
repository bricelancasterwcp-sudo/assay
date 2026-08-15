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


## v0.6 (v1.5): sequential verdicts, profile diff, long-output integrity

Three changes that sharpen the instrument rather than widen it (package
0.6.0, schema v5). No probe measures anything new: what changed is how
much evidence a verdict demands, what happens between two profiles, and
one failure mode nothing was watching.

- **Sequential testing** — codec cells no longer run a fixed n and then
  apologise in `provisional`. They sample against a pre-registered look
  schedule (n ∈ {5, 10, 20, 35}) and stop at the first look where the
  Wilson-95 interval decides a rung. An unusable cell settles in five
  calls; only a genuinely undecided one pays for 35 — which is why the
  honest mode is now the default one. See [Sequential
  testing](#sequential-testing).
- **`assay diff OLD.json NEW.json`** — capability is a point-in-time
  measurement of a serving state, so the object worth looking at is the
  difference between two timestamps. diff gates on subject identity
  first, then reports only what moved *beyond noise*: codec cells when
  their Wilson intervals are disjoint, speeds against a 2-SE band
  computed from the per-call samples the speed probe now records (and,
  where a side has too few of those, an assumed threshold that says so).
  `--gate` turns it into a CI check that fails on regressions only. See
  [assay diff](#assay-diff).
- **Long-output integrity** — the v1 live validation watched a model
  hold protocol, keep its stats, and emit degenerate repetition:
  quality dies before protocol does, and no probe was watching. An
  escalating rung ladder (512/1024/2048/4096 target tokens) now scores
  each generation for degeneracy, and the `long_output` verdict names
  the rung where it starts. See [Long-output
  integrity](#long-output-integrity).

This wave amends v1.3's verdict semantics. The amendment is recorded
rather than quietly applied, because a number whose definition changed
without saying so is worse than no number:

> This amends v1.3's verdict semantics (spec of 2026-08-13: fixed n=5
> with provisional marking; `--thorough` n=35 added in 0.4.1, sequential
> testing recorded as deferred). The deferral is now executed. Old
> profiles (schema ≤ v4) remain readable; their verdicts carry no
> `stopping_rule` and are treated as fixed-n by `assay diff`.

## v0.5 (v1.4): shapes, loops, and the matrix page

Three additions that convert the 14B subject-row lessons into
instrument (schema v4):

- **Fixed-request-shape ceiling matrix** — the right-sized ladder asks
  "what can this daemon serve?"; applications pin `num_ctx` once, so
  `ceiling_shapes` probes each pinned shape (2k/4k/8k). The daemon that
  served 14B-Q4 to 16k right-sized but errored above ~1.8k at a fixed
  8k — turning a benchmark row to 0/940 — is now a three-second
  pre-flight finding.
- **Mini-loop discipline probe** — the same model landed 97% of
  single-call codecs and 0/940 in a real loop; single-call probes
  cannot see loop failure. A scripted three-turn repair (canned
  environment, scored replies, `scripted-loop-v1` in the lens) measures
  action fidelity, patch landing, finishing, repeats, and anchor
  violations; the `loop_discipline` verdict downgrades
  follows-but-never-advances to risky.
- **`assay report *.json → report.html`** — one self-contained page
  (stdlib, inline CSS, no JS, no server) rendering N profiles as the
  capability matrix: verdict badges wear provisional dashes and
  intervals, emulated tiers are labelled, lenses are one hover away,
  dropped lists print in full.

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
- **`--thorough`**: at quick/full n, essentially every codec verdict is
  honestly `provisional` — 5/5 spans [0.57, 1.0] and even 9/10 spans
  [0.60, 0.98]. That is the instrument stating its resolution, not a
  bug; `--thorough` (35 samples per cell, 7 reps × 5 tasks) is the
  smallest n where a perfect cell clears `ready` undisputed (Wilson
  lower 0.9011). A sequential rule that samples only while an interval
  straddles a boundary is the recorded next step if thorough earns use.

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
# Full suite, default mode (sequential codec sampling), local Ollama:
assay probe http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0 --json profile.json

# Time-boxed instead: fixed n=5 per codec cell, and the lens says so.
assay probe http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0 --quick --json profile.json

# One family at a time:
assay geometry http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0
assay ceiling  http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0
assay envelope http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0
assay codecs   http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0

# Offline, on profiles already written — neither touches an endpoint:
assay diff   before.json after.json          # what moved beyond noise
assay report profile-*.json --out report.html  # N profiles as one matrix page
```

Flags: `--full | --thorough | --quick` (`--full` is the default;
`--thorough` is an alias of it, kept so old invocations still parse),
`--backend ollama|openai` (else auto-detect), `--json PATH`,
`--record PATH` (JSONL call transcript), `--max-calls N`,
`--max-prompt-tokens N`, `--window-cap N`, `--tier NAME` (requires
`--emulated` or `--real-hardware`), `--directives JSON` (your own codec
presentation).

Exit codes for the probing commands (`probe` and the family slices):
`0` profile/slice produced (whatever it says), `2` budget exhausted
before any probe family completed, `4` infrastructure failure before any
measurement. `report` spends no budget and so can only exit `0` or `4`
(an unreadable or version-less profile). `diff` measures nothing either,
and uses its own codes — see [assay diff](#assay-diff).

As a library:

```python
from assay import Budget, probe

profile = probe(
    "http://127.0.0.1:11434",
    "qwen2.5-coder:7b-instruct-q8_0",
    budget=Budget(max_calls=500, max_prompt_tokens=1_000_000),
    mode="full",
)
print(profile.to_json())
```

`budget` is a **required** argument: a library consumer burning a user's
GPU time must say how much. There is no silent default. The CLI supplies
documented defaults — quick: 110 calls / 200k prompt tokens; full and
thorough: 500 calls / 1M — overridable with `--max-calls` /
`--max-prompt-tokens`. Those cover the WORST case (a full run in which
no codec cell decides early and every one runs to the 35-sample cap); a
typical run stops well short. `mode` is explicit above because the
library's default is still `"quick"` — the cheap one — while the CLI
defaults to `--full`.

## The profile

One versioned JSON document (`assay_profile_version: 5`). Every field is
a measurement, a `None` with a named reason, or provenance.

| Field | What it says |
|---|---|
| `endpoint` | `kind` (`ollama`/`openai`), `base_url`, whether the kind was autodetected |
| `model` | `name`, `quant`, `weights_bytes`, `training_ctx` — as reported, never guessed |
| `geometry` | `kv_kib_per_token`, `vram_free_mib`, `usable_window`, and `limited_by` — **which** term (`training_ctx` / `vram` / `user_cap`) actually bound the window |
| `ceiling` | `max_verified`, `first_failure`, `failure_mode` (`hard_error` / `missing_stats` / `silent_truncation` / `canary_loss` / `none_up_to_cap` / `budget`), plus per-call evidence |
| `ceiling_shapes` | the same question asked at each **pinned** `num_ctx` an application might set (2k/4k/8k), because a daemon can serve 16k right-sized and error above ~1.8k at a fixed 8k |
| `envelope` | exact-format fidelity over N one-line probes, with failures classified (`prose` / `shape` / `refusal`) |
| `codecs` | landing rate per codec (`search_replace`, `whole_file`, `json_object`) × size grade (`tiny`, `small`, `medium`), under both landing lenses, with the `n` each cell actually spent |
| `speed` | `decode_tps` (chat usability) and `prefill_tps` (agent usability), their `evidence` class, and — new in v5 — the per-call `decode_samples` / `prefill_samples` a diff needs to tell noise from drift |
| `loop` | scripted three-turn repair: `action_fidelity`, `patch_rate`, `finish_rate`, `repeat_rate`, `anchor_violations` — single-call probes cannot see loop failure |
| `long_output` | per-rung `target_tokens`, `generated_tokens`, `distinct_ratio`, `zlib_ratio`, `degenerate`, plus a `skipped` list naming why each unattempted rung did not run |
| `verdicts` | `structured_extraction`, `patch_editing`, `long_context`, `loop_discipline`, `chat_speed`, `agent_speed`, `long_output` — each `ready` / `risky` / `unusable` / `unmeasured` (`long_output` may also read `degrades-at-N`), each carrying its own lens |
| `provenance` | started/finished, mode, seeds, budget granted vs spent, calibration, and `dropped` |

No probe uses grammar/JSON forcing: constrained generation deforms
rather than rejects, so a forced probe measures the constraint, not the
model. assay measures unforced behavior — the number an application can
act on.

## Sequential testing

A landing rate of 5/5 spans a Wilson-95 interval of roughly [0.57, 1.0]:
`ready` and `risky` are indistinguishable at that n, and v1.3 handled it
by marking the verdict `provisional` and moving on. v1.5 spends the
calls instead — but only the calls the question needs.

- **Look schedule: n ∈ {5, 10, 20, 35}**, and a cell is examined *only*
  at those points. Peeking between them inflates the false-decision
  rate, so the schedule is pre-registered rather than continuous.
- **Stopping rule: `decided`** — at a look, compute
  `wilson95(passes, n)` and stop when both interval endpoints ladder to
  the **same** rung (`ladder(lo) == ladder(hi)`). That is the exact
  negation of v1.3's provisional condition, so nothing stops early that
  v1.3 would have called undecided. It also stops a decided-*risky*
  cell, which the originally registered two-condition form (stop if
  `lo >= ready`, stop if `hi < risky`) would have missed; the spec was
  amended before any code existed and the amendment is footnoted there.
- **35 is the cap, not a promise.** A cell that never decides runs to
  35 — the smallest n at which a perfect cell clears `ready`
  undisputed (Wilson lower on 35/35 is 0.9011 against the 0.9 floor).
  A never-landing cell settles at 5. Attempts round-robin across the
  cell's heterogeneous fixtures, so a cell that stops early has still
  sampled every defect class it could reach.
- **The stop test reads the codec's *verdict* lens**: byte-equality for
  `json_object` (where validation is the landing) and
  applies-and-parses for the patch codecs, so no cell is ever ended by
  a lens no verdict uses.
- **`--quick` keeps fixed n=5** for time-boxed probes, and says so: the
  **two codec lenses** — `structured_extraction` and `patch_editing`,
  the only verdicts sequential sampling governs — carry `stopping_rule`
  (`"fixed-n"` or `"wilson95-looks-5-10-20-35"`) plus the `n_used` they
  were computed from. The other five verdicts have their own lens
  shapes and neither key. A codec verdict that stopped at n=5 is
  distinguishable from a v1.3 fixed-n=5 verdict by its lens, not by
  guessing from context; an unmeasured cell gets no `n_used` entry at
  all, because `n_used: 0` would read as a verdict graded on zero
  samples.
- **The budget is still the outer bound.** A cell stopped by the meter
  mid-schedule reports its honest partial n, and the profile can tell
  that apart from a cell the rule decided.

## Long-output integrity

From the v1 live validation, at a 15.8k prompt: protocol held, stats
were intact, `done_reason: length` — and the output was degenerate
repetition. Quality dies before protocol does, and every other probe in
this instrument watches protocol.

The `long_output` family climbs an escalating ladder of generation
targets — **512, 1024, 2048, 4096 tokens** — on one frozen enumeration
task, one call per rung, and scores each reply two ways:

- **distinct 4-gram ratio** — distinct 4-grams / total 4-grams, which
  catches a model looping phrases;
- **zlib ratio** — `len(compress(text)) / len(text)`, which catches any
  compressible collapse too tight for the n-gram window to see (one
  character repeated forever has perfect 4-gram diversity).

Either metric below its floor calls the rung degenerate. **The floors
are assumed, not derived** (`distinct < 0.30`, `zlib < 0.20`): they were
picked to sit far below anything healthy output has been observed to
produce, not fitted to a measured distribution, and the lens says so in
`thresholds: "assumed-not-derived-…"`. While that string starts with
`assumed`, every measured `long_output` verdict is forced
`provisional` — the instrument states its own resolution rather than
letting a smoke alarm read as a measurement. Deriving real floors needs
a committed degenerate anchor, which is the recorded next step.

The committed transcripts under `docs/evidence-transcripts/` are code
and JSON — the wrong genre to calibrate prose degeneracy, since code is
legitimately repetitive — so they serve as **false-positive guards**
instead: 248 healthy replies across 23 transcripts, none of which may
flag. The tightest scores 0.275 on zlib, only 1.38× the floor, and that
number is pinned in the tests.

The verdict names the extent, not just the outcome: `ready` (no scorable
rung degenerate), `degrades-at-2048` (the first degenerate rung, named),
`unusable` (already degenerate at the smallest rung this ladder could
measure), `unmeasured` (nothing scorable came back). A reply too short
to score is unmeasurable, never healthy — it spent a call and learned
nothing, and it never stands in for a clean rung. Rungs that did not run
say why (`"4096: above measured ceiling"`, `"2048: budget exhausted"`),
and the lens carries `rungs_scored` and `deepest_scored_tokens` so
`ready` on a ladder the ceiling cut off at 1024 is not confused with
`ready` verified clean to 4096.

## assay diff

`assay diff OLD.json NEW.json` compares two profiles of the same subject
and reports what moved. It touches no endpoint and spends no budget.

**The identity gate runs first.** Same model name, same quant, same
weight size, same declared tier, same emulated/real-hardware marking —
any mismatch and the pair is *not comparable*, with nothing scored at
all. Half a comparison between two different models is worse than none,
because a rung difference between two subjects is not drift.

One-sided fields split two ways, and the split is deliberate.
`model.quant` and `model.weights_bytes` known on only one side are a
**note, not a mismatch** — an older profile simply did not record them,
and the pair still compares. `provenance.tier` and
`provenance.emulated` are **still fatal** when either side lacks them:
an undeclared tier is unknown hardware, which is precisely what this
gate exists to catch. What softens there is only the wording — "not
recorded on one side" rather than "differs", because a profile that
predates the marking did not *disagree* about the machine, and a note
saying it did would send a reader hunting for hardware that changed.
The practical consequence, worth knowing before you wire this into CI:
**a pre-tier baseline cannot be diffed against a tier-marked profile.**
It exits 2. Re-baseline with a marked run.

**Then noise is separated from drift**, per family:

- **codec cells** flag only when the two Wilson intervals are
  **disjoint**; overlapping intervals are reported as within-noise by
  name, not silently dropped;
- **speeds** flag beyond a 2-SE Welch band — but only when **both**
  sides carry at least two per-call samples. Whenever either side has
  fewer, diff falls back to a fixed 20% relative threshold and stamps
  the line `threshold-20pct-assumed`. That covers more than old
  profiles: a pre-v5 profile with no samples at all, a cell that
  sampled and accepted nothing, **and any `--quick` profile**, which
  spends a single decode call. Two current v5 quick profiles diff their
  speeds under the assumed rule, not the derived one;
- **ceiling rungs, shape flips, and verdict ladders** are exact
  comparisons. Each change carries a `direction` — `regression`,
  `improvement`, or `neutral` (honest-to-honest ceiling modes and
  provisional-flag flips are facts, not grades) — and a `basis` naming
  the rule that flagged it: `rung-change`, `flip`,
  `disjoint-intervals`, `beyond-2se`, `threshold-20pct-assumed`;
- **a cell present on one side only** goes in `dropped` — never scored
  as regression or improvement. Absence of evidence is absence.

```
$ cd docs/superpowers/evidence
$ assay diff live/qwen2.5-coder-7b-instruct-q8_0-quick.json \
             live-run2/qwen2.5-coder-7b-instruct-q8_0-quick.json
no drift beyond noise
within noise: ceiling.max_verified, ceiling.failure_mode, verdict.long_context, …

$ assay diff live/qwen2.5-coder-7b-instruct-q8_0-quick.json \
             live/granite-code-8b-instruct-q8_0-quick.json
not comparable
  model.name differs: 'qwen2.5-coder:7b-instruct-q8_0' -> 'granite-code:8b-instruct-q8_0'
  model.weights_bytes differs: 8098539207 -> 8565533673
```

Those two committed live-validation runs — same models, same daemon,
thirteen minutes apart — are the project's acceptance anchor: the
sampler-level variation a diff worth trusting must **not** flag. It is a
test, not an anecdote: all three model pairs must read within noise.

Exit codes are diff's own, because diff measures nothing and its `1`
carries an answer:

| Code | Meaning |
|---|---|
| `0` | comparable, nothing moved beyond noise (with `--gate`: nothing moved in the regression direction) |
| `1` | drift found (with `--gate`: a **regression** was found; an improvement alone still exits 0) |
| `2` | not comparable — a different model, quant, weight size, or hardware tier, **or** a tier/emulated marking recorded on only one side |
| `4` | a profile file could not be read or parsed. Never `1`: exit 1 claims a measured change, and an unreadable file measured nothing |

`--gate` is the CI shape: a model that got *faster* should not fail a
build, so only worsening drift exits 1. `--json PATH` writes the full
result for machine consumption.

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

The long-output ladder is the one family whose charge is dominated by
**generation** rather than prompt: a 4096-token rung shares the context
window with its prompt and is not the same load as a 512-token one, so
each rung is charged its target. Sequential sampling makes the other
direction true — a full run's budget covers the case where no codec cell
decides early, and a typical run spends a fraction of it.

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
