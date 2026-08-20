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

Version history — what each release changed in what the numbers
*mean* — lives in [CHANGELOG.md](CHANGELOG.md); the gaps a release
deliberately carried rather than closed are ledgered in
[docs/CARRIED-DEBT.md](docs/CARRIED-DEBT.md).

**The measurements.** Fifteen models profiled on one enthusiast-tier box
(RTX 5080 16 GiB, ollama 0.32.13) on 2026-08-17 are published as the
capability matrix at
**<https://bricelancasterwcp-sudo.github.io/assay/matrix/>** — the same
page is committed at [`docs/matrix/index.html`](docs/matrix/index.html)
and opens offline. The profiles it is built from, their call
transcripts, and per-model diffs against the previous campaign are in
[`docs/superpowers/evidence/tier-enthusiast-2026-08/`](docs/superpowers/evidence/tier-enthusiast-2026-08/);
read
[`diffs/README.md`](docs/superpowers/evidence/tier-enthusiast-2026-08/diffs/README.md)
before treating any movement between the two campaigns as a model
change — the instrument changed too, and that file names every confound.

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

# Budget mode: the most profile 60 calls (and 90 seconds) can buy.
assay probe http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0 --budget-calls 60 --budget-seconds 90

# One family at a time:
assay geometry http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0
assay ceiling  http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0
assay envelope http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0
assay codecs   http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0

# Offline, on profiles already written — neither touches an endpoint:
assay diff   before.json after.json          # what moved beyond noise
assay cover  floor.json candidate.json       # does candidate cover floor?
assay report profile-*.json --out report.html  # N profiles as one matrix page
```

Flags: `--full | --thorough | --quick` (`--full` is the default;
`--thorough` is an alias of it, kept so old invocations still parse),
`--budget-calls N` / `--budget-seconds S` (either one selects **budget
mode** — see below — and they do not combine with the mode flags),
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

`assay cover <floor.json> <candidate.json>` measures nothing either,
and asks the crossed-pair question `diff` refuses: does the candidate
provide, at least as well, every cell the floor measured? Model name,
quant and weights may differ — crossed models are the point — but
hardware class (tier/emulated) and probe instrument must match
exactly. Its own exit codes, same shape as `diff --gate`: `0` covered,
`1` not covered, `2` refused (identity or instrument mismatch), `3`
incomplete (the floor measured a cell the candidate did not),
precedence `2 > 3 > 1 > 0`. A profile that cannot be read at all still
exits `4`, the same infrastructure path every command shares — that
failure is not a coverage answer, so it sits outside cover's 0/1/2/3
taxonomy.

As a library:

```python
from assay import Budget, probe

profile = probe(
    "http://127.0.0.1:11434",
    "qwen2.5-coder:7b-instruct-q8_0",
    budget=Budget(max_calls=610, max_prompt_tokens=1_000_000),
    mode="full",
)
print(profile.to_json())
```

`budget` is a **required** argument: a library consumer burning a user's
GPU time must say how much. There is no silent default. The CLI supplies
documented defaults — quick: 130 calls / 220k prompt tokens; full and
thorough: 610 calls / 1M — overridable with `--max-calls` /
`--max-prompt-tokens`. Those cover the WORST case (a full run in which
no codec cell decides early and every one runs to the 35-sample cap); a
typical run stops well short. `mode` is explicit above because the
library's default is still `"quick"` — the cheap one — while the CLI
defaults to `--full`. `mode="budget"` runs the priority-ordered consumer
probe, taking its ceilings from the `Budget` itself (`max_calls`, and
`max_seconds` for the wall clock).

## The profile

One versioned JSON document (`assay_profile_version: 10`). Every field is
a measurement, a `None` with a named reason, or provenance.

| Field | What it says |
|---|---|
| `endpoint` | `kind` (`ollama`/`openai`), `base_url`, whether the kind was autodetected |
| `model` | `name`, `quant`, `weights_bytes`, `training_ctx` — as reported, never guessed |
| `geometry` | `kv_kib_per_token`, `vram_free_mib`, `usable_window`, and `limited_by` — **which** term (`training_ctx` / `vram` / `user_cap`) actually bound the window; plus `expert_count` / `expert_used_count` where the metadata reports MoE routing (`None` on a dense model — see [MoE geometry](#moe-geometry)) |
| `ceiling` | `max_verified`, `first_failure`, `failure_mode` (`hard_error` / `missing_stats` / `silent_truncation` / `canary_loss` / `none_up_to_cap` / `budget`), plus per-call evidence |
| `ceiling_shapes` | the same question asked at each **pinned** `num_ctx` an application might set (2k/4k/8k), because a daemon can serve 16k right-sized and error above ~1.8k at a fixed 8k |
| `envelope` | exact-format fidelity over N one-line probes, with failures classified (`prose` / `shape` / `refusal`) |
| `codecs` | landing rate per codec × grade, under both landing lenses, with the `n` each cell actually spent. `search_replace` and `whole_file` are graded by **size** (`tiny`, `small`, `medium`); `json_object` is graded by size **and by shape** — `nested`, `tabular`, `constrained`, new in v1.7, so six cells rather than three (see the CHANGELOG) |
| `speed` | `decode_tps` (chat usability) and `prefill_tps` (agent usability), their `evidence` class, and — new in v5 — the per-call `decode_samples` / `prefill_samples` a diff needs to tell noise from drift |
| `loop` | scripted repair over **two** scripts. `action_fidelity` and `repeat_rate` are rates over the shared `n_turns`, and `anchor_violations` a count over the same turns — all three span golden and error turns alike; `patch_rate` and `finish_rate` are golden-only, over `n_runs`; and — new in v6 — `recovery_rate` / `doom_loop_rate` are error-only, over `n_error_runs`. Single-call probes cannot see loop failure |
| `long_output` | per-rung `target_tokens`, `generated_tokens`, `distinct_ratio`, `zlib_ratio`, `degenerate`, plus a `skipped` list naming why each unattempted rung did not run |
| `tools` | native tool calling over a **20-task** pool (`scripted-tools-v2`): `supported` (three-state), `call_rate`, `right_tool_rate`, `args_valid_rate`, `result_use_rate`, the `composite` the verdict ladders on, and `n_tasks` / `n_turns` — rates of **instructed** behavior, see [Native tool calling](#native-tool-calling); `n_truncated` / `n_stop_unreported` (new in v7), the scored turns the token ceiling cut off and the ones whose backend never said how they stopped (recorded beside the rates, never fed into them); and — new in v8 — `stopping_rule`, because full mode now samples the pool **sequentially** at looks {5, 10, 20} while `--quick` keeps the frozen five |
| `parallel` | full mode only: one `rows` entry per k ∈ {2, 4} — `per_lane_decode_tps`, `total_throughput_tps`, `degradation_ratio` against the same run's single-lane `speed.decode_tps`, the `mode` (`parallel` / `serialized`) the whole family exists to report, `n_lanes_ok`, `lane_errors`, `evidence` — beside the `baseline_decode_tps` it divided by, the `overlap_fraction` the overlap test used with its `overlap_provenance` (**chosen**, not derived — new in v1.9, schema v10; classifies overlap as a fraction of the shorter lane's span, not a fixed number of seconds, so the reading is correct at any time scale, see the CHANGELOG), and a `skipped` list naming every k the budget refused. Measurement-only through v1.7; `verdicts.parallel` ladders it starting v1.8 (schema v9), on **chosen** (not derived) floors. A document written before v10 carries `tolerance_s` / `tolerance_provenance` instead — a seconds tolerance, not a fraction — and this project's renderers read each era in its own terms rather than converting one into the other |
| `verdicts` | `structured_extraction`, `patch_editing`, `long_context`, `loop_discipline`, `chat_speed`, `agent_speed`, `long_output`, `tool_calling`, `parallel` — each `ready` / `risky` / `unusable` / `unmeasured` (`long_output` may also read `degrades-at-N`; `tool_calling` may also read `unsupported`), each carrying its own lens |
| `provenance` | started/finished, mode, seeds, budget granted vs spent, calibration, and `dropped`. The seconds pair appears only when a wall clock was actually granted — `spent.seconds: 0.0` on an unmetered run would read as a run that took no time |

No probe uses grammar/JSON forcing: constrained generation deforms
rather than rejects, so a forced probe measures the constraint, not the
model. assay measures unforced behavior — the number an application can
act on.

## MoE geometry

A mixture-of-experts model has two numbers a capacity planner needs and
a dense one does not, so `geometry` carries them when the metadata
states them: `expert_count` (experts in total) and `expert_used_count`
(experts routed per token). Both are `None` on a dense model and on any
backend that cannot read architecture metadata — a dense model is **not
a 0-expert MoE**, and writing `0` would read downstream as a measured
routing fact. `render_table` and the report print `MoE <used>-of-<count>`
only when **both** counts are measured; one measured half prints
nothing, because "MoE 8-of-None" shows an unmeasured half as though it
had been measured.

The kv-cache formula does **not** take an expert term, and that is a
design statement rather than an omission: K/V heads are dense in MoE
architectures — the experts live in the FFN weights, which the cache
never holds — so a routed model pays exactly
`2 × block_count × kv_head_count × head_dim × bytes/element` per token,
same as a dense one. The expert counts ride in `geometry` because they
explain the **weights** footprint, not the cache one.

What did change is `head_dim`. The model file's **stated**
`attention.key_length` is now preferred, and the derivation
(`embedding_length` ÷ `attention.head_count`) is kept only as the
fallback for metadata that omits it. That derivation assumes attention
width equals embedding width over heads, which is false wherever
attention is sized independently — qwen3-moe q4 states `key_length` 128
where 2048 ÷ 32 derives 64 — and a `head_dim` off by 2x silently halves
every kv number the window law rests on. Neither reported → `None`, and
the geometry is unmeasurable rather than guessed.

**That fix is an erratum against the committed v1.4 profiles**, and the
live anchor measured the size of it. Both
`docs/superpowers/evidence/tier-enthusiast/` profiles checked report
`kv_kib_per_token: 216` — which is the signature of the derivation, not
a property of the models. Read from the stated `key_length` instead,
`deepseek-coder-v2:16b-lite-q5_K_M` costs **324** KiB/token (head_dim
192, not 128) and `qwen3.8:27b` costs **260** (head_dim 256, not 213);
under each profile's own VRAM reading the planned window falls from 8092
to 5394 and from 4922 to 4096 — **33.3% and 16.8% of the promised window
is not there** (shortfall over what v1.4 promised; the same gap stated as
excess over the true window is 50.0% and 20.2%, identically the kv excess
above, since the window is inversely proportional to cost per token
wherever VRAM binds — the two bases must not be mixed). They are left as
committed — evidence is not rewritten to suit a later fix — a test
reproduces each old figure from the derived `head_dim` so the discrepancy
stays explained rather than tidied, and the erratum is filed beside the
profiles at
[`evidence/tier-enthusiast/ERRATA.md`](docs/superpowers/evidence/tier-enthusiast/ERRATA.md).
A pre-registered sweep (2026-08-17) then classified **every** committed
profile in the repository against E1: two more corrections in that
errata file (gemma2-9b's kv figure was wrong while its promised window
held; mistral-nemo's window was **under**-promised — the stated
`key_length` can sit on either side of the derivation), codegemma's v1
profiles corrected beside themselves (336 → 448 KiB/token — the
validation write-up's "6×" ratio is really 8×), and the rest settled
clean or unchanged-by-construction; protocol, verbatim captures, and
the full table live in
[`evidence/e1-sweep/`](docs/superpowers/evidence/e1-sweep/PROTOCOL.md).
The same capture also corrected an assumption about which model
was the MoE: `qwen3.8:27b` (architecture `qwen35`) reports **no**
`expert_*` keys at all and correctly reads `None`/`None`, while
`deepseek-coder-v2:16b-lite` (`deepseek2`) reports **64 experts, 6
routed** — the numbers the renderer prints as `MoE 6-of-64`.

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
- **`--quick` keeps fixed n=5** for time-boxed probes, and says so:
  **three lenses** — `structured_extraction` and `patch_editing`, the
  codec verdicts, and, since v1.7, `tool_calling` — carry
  `stopping_rule` (`"fixed-n"`, or the schedule the sample actually
  walked: `"wilson95-looks-5-10-20-35"` for a codec cell,
  `"wilson95-looks-5-10-20"` for the tools pool, whose task count bounds
  its last look) plus the `n_used` they were computed from. The other
  five verdicts have their own lens shapes and neither key. A codec
  verdict that stopped at n=5 is
  distinguishable from a v1.3 fixed-n=5 verdict by its lens, not by
  guessing from context; an unmeasured cell gets no `n_used` entry at
  all, because `n_used: 0` would read as a verdict graded on zero
  samples.
- **The budget is still the outer bound.** A cell stopped by the meter
  mid-schedule reports its honest partial n, and the profile can tell
  that apart from a cell the rule decided.

## Native tool calling

Every other family that puts a question to the model measures what it
**writes**. An agent harness runs on what the model **calls**, and
nothing lets us assume the two are correlated: a model that writes a
clean patch is still useless in a harness if it cannot emit one
well-formed function call. So the two are measured separately.

`scripted-tools-v2` is a script, not a benchmark. **Twenty**
heterogeneous tasks — every tool appears at least four times, the two
argument-taking tools carry fifteen distinct pinned values between them,
and `run_tests` takes no arguments at all, so a model that has learned
one memorised call shape scores differently from one that reads the
request — against `toolset-v1`, three schemas frozen verbatim and
offered on every call. The first five tasks are the v1 pool **verbatim
and in order**, at the v1 seeds. Two turns per task, and nothing
branches on what the model said:

- **T1** asks the task with the toolset offered, and scores three
  things: exactly one call was emitted, its name is the right tool, and
  its arguments are schema-valid **and** equal to the value the request
  named verbatim (every task that names a file or a query quotes it from
  the user's own words, which is what makes argument checking mechanical
  rather than a judgement call; `run_tests` names neither and pins the
  empty argument object). `composite` is the per-task AND of those
  three, and it is what the verdict ladders on — the right tool called
  with junk arguments has not done the job.
- **T2** replays the same messages plus a **canned** golden call (never
  the model's, so every model is asked the same second question) and a
  `role: "tool"` result carrying a seeded canary. It scores
  `result_use_rate`: the canary comes back in the text **and** no
  further call is emitted. This is the half that catches a model which
  can call a tool but cannot read the answer.

`right_tool_rate` and `args_valid_rate` are over the T1s that called
**anything**. A task where nothing was called has no tool name and no
arguments to judge; scoring it zero would double-count the miss
`call_rate` already carries and disguise "never called" as "called
badly". Nothing called at all → both are `None`, never `0.0`. `n_tasks`
is the composite's denominator and is what rides in the verdict lens as
`n_used` — the T2 turns score result-use, not the composite, so quoting
`n_turns` there would claim evidence the verdict never saw.

**The pool is sampled sequentially** (v1.7), and the twenty tasks are
what make that honest: five scripted tasks cannot yield n=20, because
re-running the same five with fresh seeds at temperature 0.2 measures
the sampler rather than the model. Full mode examines the composite at
looks {5, 10, 20} — a truncation of the codec schedule, bounded by the
pool itself; there is no 35th task, and growing the pool would restore
the look — and stops at the first look whose Wilson-95 interval ladders
both endpoints to the same rung. `--quick` keeps the frozen five-task
prefix, so a quick number still compares across the version boundary
and the committed anchor replays byte-identically. Which of the two
happened is read off `stopping_rule` in the lens, never guessed from
`n_used`: a composite over five tasks that the rule *decided* and one
over five because five was all quick asked for are different findings.

What sequential sampling buys here is **asymmetric**, and the asymmetry
is arithmetic rather than a shortfall of the pool. Enumerated over this
schedule, the composites that decide at all are exactly the `unusable`
ones: 0/5 at the first look, 0–2/10 at the second, 0–7/20 at the third.
That is the entire decided set. Nothing decides `risky` — no pass count
at n ≤ 20 has an interval that fits inside [0.6, 0.9) — and nothing
decides `ready`. What the schedule buys is still real: a pool at
composite 0.2 reads provisional at 1/5, where the interval [0.036,
0.625] straddles `unusable` and `risky`, and **decides** `unusable` at
2/10 on [0.057, 0.510] — the same rate, five tasks later. A *perfect*
pool does not decide: Wilson's lower bound on 20/20 is 0.8389 against
the 0.9 `ready` floor, so 20/20 runs to the cap and still reads `ready`
provisional. What it gains is resolution, not a decision — roughly
[0.839, 1.0] where fixed n=5 spanned [0.566, 1.0]. n=35 remains the
only n at which a perfect cell clears `ready` undisputed, and this pool
has no 35th task.

**These are rates of INSTRUCTED behavior.** The instrument's system line
announces, criterion for criterion, the rubric it scores: call exactly
one tool, use the arguments the request names, quote the result token
verbatim. That is deliberate — every model is asked in the same words,
so nothing is measured except the model — but the consequence is stated
rather than hidden. `call_rate` says *"told to call one tool, it called
one tool"*, never *"it reached for a tool unprompted"*, and a reader
comparing these numbers against an agent harness that does **not** spell
the rules out should expect this instrument to read high.

**`unsupported` is a verdict value, not a gap.** `supported` is
three-state: `None` = never attempted (the budget died first), `False` =
the endpoint **refused** the tools parameter, `True` = it spoke the
protocol at least once. The refusal is classified by behavior class, not
by wording: a 4xx whose *error fields* mention "tool" is a refusal, and
every other non-2xx is infrastructure. Only the error fields are
scanned, never the whole body — a server that echoes the failing request
echoes the `tools` array we sent, and reading our own payload back would
fabricate a capability fact. A refusal on the **first** call ends the
probe with every rate `None` (nine more refusals measure nothing new); a
refusal **after** a turn has scored keeps `supported=True` and the
honest partial, because the endpoint demonstrably does speak the
protocol.

`--record` transcripts carry tool turns as their own `kind`
(`chat_tools`, beside `generate`), keyed on a canonical serialization of
the whole payload — the messages *and* the schemas offered, since the
same conversation with a different toolset is a different question — and
a row of the other kind is a replay miss rather than a match. Both kinds
also record the endpoint's own `error_raw`, so a replayed refusal is the
refusal that happened, not a re-derived guess.

So `tool_calling` reads `ready` / `risky` / `unusable` on the composite,
`unmeasured` when the family never ran, and `unsupported` for the
refusal. It is not the only verdict that extends the four common values
— `long_output` adds `degrades-at-N` — but it is the only one whose
extra value describes the **endpoint** rather than a rung the model
reached, and `unsupported` is neither of its neighbours: `unmeasured`
would hide a fact we established, and `unusable` would blame a model
that was never asked to do the task.

## The scripted repair loop

The 2026-08-14 pair of measurements that forced this probe: one model
landed 97% of single-call codec probes and scored 0/940 in a real
multi-turn repair loop. Single-call probes structurally cannot see turn
discipline, repetition, anchor violations, or knowing when to stop.
`scripted-loop-v2` is a **scripted** repair conversation — the
environment's side is canned, the model's replies are scored per turn,
and no branching happens on what the model actually says.

The **golden** script is three turns (read → patch → done). Two of the
family's rates are its alone: **patch landing** (turn 2's payload, under
the same applies-and-parses lens the codecs use) and **finishing**
(turn 3, after being told the tests pass, is `done`), both over `n_runs`
completed golden runs.

The **error** script is the v1.6 half. The golden path only ever asks
what a model does when everything works; the failure that actually ended
robigo runs was the other one — a patch comes back "SEARCH text not
found" and the model re-emits the same block, turn after turn. So every
run also plays two turns in which the model's patch has failed: the
canned failure is the measured qwen signature, the right target line
with its indentation stripped, and it is built from the fixture's own
lines rather than hand-typed, so a canned "failure" that would in fact
apply cannot drift in (a test pins that it really does not).

Its first turn is the golden first turn **character for character** —
the model has been told nothing yet, so there is nothing to differ about
— which makes the **seeds** the only thing keeping the two draws apart.
Error runs are therefore seeded `seed_base + 50 + run` against the
golden runs' `seed_base + run`: disjoint by construction, so the error
script is a second measurement rather than a re-roll of the first. (At
the default three runs and `seed_base` 800 the golden turns draw
800/801/802 and the error turns 850/851/852.) Two rates come out:

- **`recovery_rate`** — the next reply is a `patch` at the source file
  that applies and parses;
- **`doom_loop_rate`** — the next reply re-emits the SEARCH it was just
  shown failing, compared per line whitespace-normalized so a
  cosmetically respaced re-emission still counts. It reads a reply
  carrying exactly one block, the same one-block discipline the codec
  lens applies: a ragged two-block reply is not one repeated action.

A reply can be **neither**: reading the file again is no recovery, and
it is no doom loop either. And the doom lens is gated on **application**
— a block that applied is never a doom loop, whatever else is wrong with
the reply. That gate is load-bearing rather than decorative: the same
normalization that catches respacing also erases the leading indentation
which is the *only* difference between the canned broken block and a
correct one, so without the gate a correct fix scores as a re-emission.
Both rates are `None` when the error script never ran — unmeasured is
not zero — and `n_error_runs` carries the denominator in the family and
in the verdict lens, because a budget-truncated 1/1 and a complete 5/5
both read `1.0` and nothing else tells them apart.

**The two scripts are two halves of one instrument, not two
instruments,** and three of the family's numbers say so. Both
`action_fidelity` and `repeat_rate` are rates over one shared
denominator, and `anchor_violations` is a raw count over the same
turns — all three are scored on **every** turn of **both** scripts,
golden and error alike. `n_turns` is five per
run — three golden, two error — and at the default three runs a model
that emits a clean action line on every golden turn and then answers the
rejected patch with prose scores `action_fidelity` **12/15 = 0.8**, not
the 9/9 the golden script alone would have shown. The `loop_discipline`
Wilson interval is computed over that same shared `n_turns`. An anchor
violation counts wherever it happens — patching the read-only test file
in answer to a rejection is the same sin as patching it on turn 2 — and
repeats are looked for within each run.

`loop_discipline` ladders on action fidelity and then demotes twice:
`ready` with a patch rate below 0.5 becomes `risky` (follows the loop,
never advances it — the 14B shape: fidelity 1.0, 0/940), and `ready`
with a **measured** recovery rate below 0.5 becomes `risky` too. The
recovery guard tests `is not None`, not truthiness: `0.0` is a model
that was asked and never recovered and must demote, while `None` is an
error script that never ran and must demote nothing.

## The live anchor

Both v1.6 instruments and the MoE reading are pinned to a live capture,
not to fakes: `docs/superpowers/evidence/tools-anchor/` holds the
transcripts, the verbatim `/api/show` bodies and the values they
measured (`results.json`), taken 2026-08-16 against **ollama 0.32.13**
with the probes' own prompts, toolset, seeds and scripts — one run per
model, nothing tuned. The acceptance tests replay those committed files
through the strict `CallReplayer` and re-derive every number, so the
suite stays offline while the claims stay measured.

| model | supported | call | right tool | args | result use | composite |
|---|---|---|---|---|---|---|
| gemma2:9b | **false** | — | — | — | — | — |
| llama3.1:8b | true | 1.00 | 1.00 | 1.00 | 0.40 | 1.00 |
| mistral-nemo:latest | true | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |
| qwen2.5-coder:7b-instruct-q8_0 | true | **0.00** | — | — | 0.80 | **0.00** |

Three things the capture settled:

- **The refusal is real.** gemma2:9b answered `HTTP 400 {"error":
  "registry.ollama.ai/library/gemma2:9b does not support tools"}` — the
  classifier's behavior-class rule (4xx whose text names tools, rather
  than Ollama's exact wording) read it correctly on first contact, one
  call spent instead of ten, and the body travels in `error_raw`.
- **`supported` and `call_rate` are genuinely different facts.**
  qwen2.5-coder:7b took the `tools` parameter and then wrote the correct
  call as **plain text** five times out of five, emitting no native call
  at all — while reading a supplied tool result perfectly well (0.80).
  An instrument that scored what models write would have given it full
  marks; a harness gets nothing from it. That the two rates over the
  calls that happened stay `None` rather than `0.0` is the same capture:
  there was no tool name and no argument to judge.
- **The doom loop is measured, not assumed.** Shown its patch rejected
  with "SEARCH text not found" and the file unchanged, qwen2.5-coder:7b
  re-emitted the identical failing block on **3 of 3** error runs —
  `recovery_rate` 0.00, `doom_loop_rate` 1.00, beside `action_fidelity`
  1.00 and `finish_rate` 1.00 from the same 15 calls. The demotion the
  guard exists for fires on real data.

Also captured, and worth its own line: `result_use_rate` orders these
three models the opposite way round from `call_rate` (0.80 / 0.40 /
0.00), with mistral-nemo answering four of its five tool turns from an
entirely invented file rather than the result it was handed. No single
number ranks them, which is why the family reports five.

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

Either metric below its floor calls the rung degenerate. **One floor is
derived, the other is not** — the lens says which in
`thresholds: "derived-2026-08-15 (zlib only; distinct still assumed)"`.
The degeneracy anchor (`docs/superpowers/evidence/degenerate-anchor/`)
captured 28 live enumeration replies from seven models, labelled them by
reading them, and set each floor at the midpoint between the degenerate
cluster's best value and the healthy cluster's worst:

- `zlib < 0.2557` is **derived**, from the gap between 0.2362 (a 0.5b
  model emitting one sentence on repeat) and 0.2752 (the worst healthy
  reply on record). The old assumed 0.20 missed two genuinely degenerate
  replies and cleared two more at 0.1976 and 0.1997.
- `distinct < 0.30` **stays assumed**, because on that metric the
  clusters overlap: a reply whose items 7–20 are the same sentence
  scores 0.6127, above a healthy code reply at 0.5952. Renumbering each
  looped line keeps the 4-gram window fed, which is exactly the collapse
  the zlib metric is there to catch instead.

Because the derived floor alone flags all ten degenerate samples,
measured `long_output` verdicts are no longer forced `provisional` by the
threshold cap — which still fires for any provenance that starts with
`assumed`. For this family `provisional` now reports **ladder
completeness**: it is True whenever a rung was skipped (ceiling or
budget) or came back unscorable, and False only when every configured
rung was climbed and scored. `ready` verified clean to 4096 and `ready`
on a two-rung ladder the ceiling cut off at 1024 are different findings,
and only the first one wears a settled badge.

The committed transcripts under `docs/evidence-transcripts/` are code
and JSON — the wrong genre to calibrate prose degeneracy on its own,
since code is legitimately repetitive — so they serve as
**false-positive guards**, and their worst case is what caps how high a
floor may go: 248 healthy replies across 23 transcripts, none of which
may flag. The tightest scores 0.2752 on zlib, now only 1.08× the derived
floor (it was 1.38× the assumed one), and both edges are pinned in the
tests. Deriving bought sensitivity and spent headroom, and the anchor
README says so in those words.

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
- **a cell both sides measured, but under two different rules** goes in
  `incomparable` — also never scored. `SEMANTIC_BREAKS` (`diff.py`)
  lists cells a release has redefined without changing their name,
  type, or presence. It holds **one entry today**
  (`verdict.parallel`), and it is **not a complete inventory**: at
  least five earlier releases also redefined existing cells and are not
  registered, so a pair straddling one of those is still scored
  silently. See the registry's own comment for the list.
  `incomparable` covers a **registered** cell only — either because the
  pair straddles that cell's break, or because a `probe_version` is
  missing or unparseable on either side, which makes every *registered*
  break unestablished. Cells with no entry — `ceiling`,
  `ceiling_shapes`, `codecs`, `speed`, and every unregistered verdict —
  are compared normally regardless of version, so a version-less
  profile is **not** protected across the board.

**The verdict ladder:**

```
ready > risky > degrades-at-N > unusable > unsupported
```

Between two `degrades-at` rungs the **larger** N is the better one —
degrading later is an improvement, not a regression, and a gate told the
opposite would fail a build for progress. `degrades-at-N` (the
`long_output` family's rung) sits above `unusable` because a model that
holds together for a while is better than one that never does, and
below `risky`.
`unsupported` (the `tools` family's rung) is the **bottom**, not a tie
with `unusable`: being asked and failing every task is more than never
being asked. That is what makes `ready → unsupported` a regression
`--gate` fails on and `unsupported → ready` an improvement it does not.
`unmeasured` is the ladder's one remaining value and is deliberately
absent from it — it is dropped, never ranked. `unsupported` is the word
that looks like absence and is not: the endpoint was asked and said no,
so it ranks.

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
| `3` | **incomplete or incomparable** — either at least one cell was measured on exactly one side (`dropped`), or a cell both sides measured was measured under two different rules (`incomparable`, a registered `SEMANTIC_BREAKS` entry). Outranks `1`: neither a vanished family nor a cell the two instruments defined differently is a measured move. A pair spanning a schema bump reads `3` whenever the newer schema actually measured a cell the older one lacks — not merely because the schemas differ — and a budget-mode profile compared against a full one reads `3` under the same rule, whenever the full run measured a cell the budget run skipped |
| `4` | a profile file could not be read or parsed. Never `1`: exit 1 claims a measured change, and an unreadable file measured nothing |

`--gate` is the CI shape: a model that got *faster* should not fail a
build, so only worsening drift exits 1. `--json PATH` writes the full
result for machine consumption.

Exit `3` is why a green `assay diff --gate` can be trusted. Before
v1.8 a pair whose newer profile simply stopped measuring five families
exited `0` — "no drift beyond noise" — because a vanished family is
unmeasured rather than worse. It was measured in the field: a v8
profile compared against a v4 one passed the gate while `long_output`,
`tool_calling` and three deep JSON cells went unmeasured on one side.
The rule now is that an incomplete comparison says so.

**v1.10 added a second cause for exit 3.** `SEMANTIC_BREAKS` (`diff.py`)
names cells whose measurement rule changed at a given release —
`verdict.parallel`'s v1.9 fraction rewrite is the one entry today. A
pair straddling a registered break is never scored, even though both
sides measured the cell: it lands in `incomparable` rather than being
compared as if the two runs agreed on what they measured, and exit 3
fires exactly as it does for `dropped`. `identity_gate` still does not
check `probe_version` — a version difference is not a different
endpoint — so this is deliberately narrow: it stops one instrument
change from reading as an endpoint fact, not general version-awareness.

**How narrow, stated plainly.** The registry is not a complete record
of this project's rule changes. At least five earlier releases —
v1.1, v1.3, v1.5, v1.6 and v1.7 — also redefined cells that already
existed, and none of them is registered, so a pair straddling one of
those is scored today with no warning. The v1.3 fixture-set change
alone redefines every `codec.*` cell; the campaign evidence directory
says so in its own words, that `diff` "deliberately does not gate on
fixture-set name, so it will subtract across the change without saying
so". They are not backfilled because doing so needs decisions the
current design cannot express — several breaks apply only in some
modes, and the `codec.*` family has no call site that reads the
registry at all. Recorded as open debt in `docs/CARRIED-DEBT.md`
(v1.10, "Diff", items 1 and 2). Read a clean `incomparable: ()` as
"nothing REGISTERED was straddled", not as "the two runs measured the
same things".

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

**The defaults must cover the worst case, not the clean one** — a
default below the suite's own call count exhausts mid-family on every
run. Quick is **130 calls / 220k prompt tokens**, raised in v1.6 from
110 / 200k, and the reason is exactly the failure that rule exists to
prevent: the loop's error script (+2 calls per run, so the loop family
went 9 → 15) and the new tools family (+10) pushed quick's worst case to
**109 of 110** — one call short of a mid-family death, on the mode an
operator reaches for when they are in a hurry. 130 restores the headroom
the 110 was chosen for, and the token ceiling rises with it so the two
stay proportionate; a call budget that outruns its token budget just
moves the death to the other meter. Measured on the scripted suite, a
clean quick run spends 117 calls and 79,420 of the 220,000 tokens — up
from 102 in v1.6 because `json_object` gained three deeper grades, which
takes the codec family from 45 calls to 60. Add the most a failing
ceiling ladder can spend bisecting (4 calls at quick's one seed) and
quick's worst case is **121 of 130**. Those per-family numbers are
declared, not counted: `assay.run.worst_case_calls` prices every family
from the constants its own probe consumes, and the tests sum the table
against the metered run.

Full and thorough are **610 calls / 1M tokens**, raised in v1.7 from
500 / 1M for the same reason quick moved in v1.6. Full is sequential, so
its worst case is the old thorough worst case — no codec cell decides
early and every one runs to the 35-sample cap — which the deep json
grades take from 315 codec calls to 420. A clean full run on the
scripted suite measures **552 calls** and 230,293 tokens, and at the old
500 it exhausted after the codec matrix and reported the long-output
ladder and the whole tools family as `dropped`. The ceiling is derived
from that measurement rather than guessed: about 10% over the clean run —
600 when the clean run was 546, and 610 once the parallel family's six
concurrent lanes took it to 552. The first claim on that headroom is a
failing ceiling ladder's bisection, at most 8 calls at full's two seeds,
which puts full's worst case at **560 of 610**. The default follows the
measurement;
the acceptance does not move to fit it. The token ceiling does not
move — 230k of 1M is not close. Quick measures no concurrency, so its
numbers are unchanged.

### Budget mode

`--budget-calls N` (and optionally `--budget-seconds S`) answers a
different question from the other modes: not "measure everything" but
**"measure the most load-bearing profile you can for N calls"**. It is
the settings-time probe an application runs against whatever endpoint a
user pointed it at.

Families run in a pre-registered priority — geometry, envelope, speed,
the `json_object` cells, the patch cells, tools, ceiling, loop,
long-output (`assay.run.PRIORITY`, pinned by a test so a reordering
cannot happen quietly). Before each one starts, its **declared worst
case** (`worst_case_calls`, plus the bisection a failing ceiling ladder
would need) is checked against what is left. A family that does not fit
is dropped by name — `"<family>: budget — would exceed remaining"` —
and never started, because a family cut off halfway spends calls on a
number no verdict can be read off. **What starts, finishes — on the
call meter**, which is the only ceiling a preflight can reserve
against: the cost table declares calls, and there is no per-family
token declaration to hold back. A family can still die on the **token**
meter mid-run; when it does it keeps what it measured and the families
after it drop, which is the same honest-partial path every other mode
uses.

A refusal is not the end of the run: the order is a priority, not a
cliff, so a cheaper family further down still runs. The wall clock cuts
nothing at all — checked at family boundaries and between calls, never
mid-call, so a run that runs out of time stops before the next call
rather than leaving a half-measurement behind, and the families after it
read `"<family>: budget — seconds"`. Two families are not in the
priority and say so as mode facts: `ceiling_shapes` (a capacity-planning
question, not a settings-time one) and `parallel` (full mode measures
concurrency).

Budget mode measures quick-shaped families — fixed n, no sequential
schedules — because its scarce resource is coverage, not depth. Whatever
it did not buy reads `unmeasured` through the same None-vs-zero path as
any other unmeasured family.

One difference from `--quick` shows up in a **diff** rather than in an
error: budget mode narrows the ceiling ladder by the model's
`training_ctx` (as `--full` does and `--quick` does not), which is what
lets its preflight reserve half as much on the small-context models
where a consumer's budget is tightest. On such a model a budget profile
can therefore report a lower `ceiling.max_verified` than a quick profile
of the same endpoint — a mode artifact, not a regression.

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
