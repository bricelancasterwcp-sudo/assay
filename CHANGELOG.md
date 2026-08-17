# Changelog

Newest first. Each release names the wave it shipped (v1.x) beside the
package version, and states what changed in what the numbers MEAN,
not just what code moved — a version bump here is a claim about the
instrument.

## v0.9 (v1.7): consumers and the matrix

The first three waves built an instrument an operator runs. This one
makes it cheap enough for an **application** to run at settings time,
and widens two families whose readings were resolution-limited rather
than wrong (package 0.9.0, schema v8). No verdict changed its
definition. Two of them got more evidence behind them, one new family
reports a fact no verdict ladders on yet, and one new mode answers a
question the other modes cannot.

- **Tools sampled sequentially** — `scripted-tools-v1` becomes
  `scripted-tools-v2`: the task pool grows 5 → 20 (`toolset-v1` stays
  frozen — the pool is what gained identity), and full mode examines the
  composite at looks {5, 10, 20}, stopping at the first look whose
  Wilson-95 interval ladders both endpoints to one rung. What that
  changes in the numbers is **asymmetric**, and the asymmetry is
  arithmetic rather than a shortfall of the pool. `unusable` and clearly
  `risky` pools can now DECIDE, where v1.6 could only report them
  provisional at n=5. A *perfect* pool still cannot: Wilson's lower
  bound on 20/20 is 0.8389 against the 0.9 `ready` floor, so 20/20 runs
  to the cap and reads `ready` provisional — at roughly [0.839, 1.0]
  where fixed n=5 spanned [0.566, 1.0]. That is resolution bought, not a
  decision; n=35 remains the only n at which a perfect cell clears
  `ready` undisputed, and this pool has no 35th task. `--quick` keeps
  the v1 five verbatim at the v1 seeds, so quick numbers stay comparable
  across the boundary and the committed tools-anchor replays
  byte-identically. Which sample happened is read off `stopping_rule` in
  the `tool_calling` lens (`fixed-n` / `wilson95-looks-5-10-20`), never
  guessed from `n_used`.
- **Deeper JSON — `codec-fixtures-v3`** — `json_object` gains `nested`,
  `tabular` and `constrained` beside the three size grades, so the codec
  matrix now reports six json cells rather than three. They are new
  **columns**. `structured_extraction` ladders on exactly the cell it
  always did and **did not move**: a json verdict that changed across
  this boundary changed because the model did, not because the grade
  under it was silently redefined. The v2 fixtures are byte-untouched
  (pinned by a test), so every pre-existing cell still compares like
  with like in a `diff`.
- **`parallel` — a measurement-only family** — what k ∈ {2, 4}
  concurrent requests do to one endpoint. **No verdict this wave**:
  there is no measured floor to ladder a degradation ratio against, and
  a rung invented for one would be the overclaim the rest of this schema
  exists to refuse. The headline is not a rate but a scheduling fact —
  `mode` reads `parallel` or `serialized`, and a serialized endpoint
  multiplies latency by k instead of sharing throughput, which is the
  difference between an agent fleet that works and one that does not.
  Three honesty rules travel with it. Rates come from server timings,
  never the client's clock (the clock decides the scheduling fact and
  nothing else). The overlap tolerance is a **chosen** constant and says
  so in `tolerance_provenance` — thresholds are derived, not chosen, and
  one that is not must be flagged until the campaign's live rows can
  sanity-check it. And nothing partial is reported as whole:
  `total_throughput_tps` is `None` unless **every** returned lane
  reported timings, an errored lane is named in `lane_errors` rather
  than averaged in as a zero, and a k the budget refused is named in
  `skipped` rather than left an absent row a reader has to explain.
- **Budget mode — `--budget-calls N [--budget-seconds S]`** — the
  settings-time question: not "measure everything" but *"measure the
  most load-bearing profile you can for N calls"*. Families run in a
  pre-registered priority (`run.PRIORITY`, pinned as data), and each is
  preflighted against its own **declared** worst case before it starts.
  A family that does not fit is dropped **by name** — `"<family>:
  budget — would exceed remaining"`, or `"— seconds"` when the clock is
  what ran out — and never started, because a family cut off halfway
  spends calls on a number no verdict can be read off. What starts,
  finishes. A refusal does not end the run: the priority is an order,
  not a cliff, so a cheaper family below it still measures. The wall
  clock is checked between calls, never mid-call — a cut call is an
  uncontrolled instrument variable. The flags live on `probe` alone, the
  only command that implements the priority. One consequence shows up in
  a **diff** rather than an error, and is stated rather than left to be
  discovered: budget mode narrows the ceiling ladder by the model's
  `training_ctx` (as `--full` does and `--quick` does not), so on a
  small-context model a budget profile can report a lower
  `ceiling.max_verified` than a quick profile of the same endpoint — a
  mode artifact, not a regression.
- **The default budgets follow the measurement** — full and thorough go
  500 → **610 calls**; the token ceilings do not move (230k of 1M is not
  close). The deep json grades take full's codec term 315 → 420 and
  sequential tools takes that family 10 → 40, and at 500 a clean full
  run died after the codec matrix with the long-output ladder and the
  whole tools family in `dropped`. The new ceiling is derived from a
  re-measurement rather than guessed: a clean full run on the scripted
  suite spends **552** calls (546 before the parallel family's six
  lanes), ~10% over that is 607, rounded to 610 — and the first claim on
  that headroom is the one term the per-family numbers do not carry, a
  failing ceiling ladder's bisection at most 8 calls over full's two
  seeds, which puts full's worst case at **560 of 610**. Quick's
  ceilings are unchanged at 130 / 220k: its clean run is 117 and its
  worst case 121, up from v1.6 because the deep grades cost it 15 codec
  calls, and it measures no concurrency. Those per-family numbers are
  now **declared, not hand-counted**: `run.WORST_CASE` prices every
  family from the constants its own probe consumes, the tests sum the
  table against the metered runs, and budget mode preflights on the same
  table — so a family that grows re-prices itself everywhere at once
  instead of waiting for a mid-family death to correct a hand count.
- **The matrix** — this release is the instrument the 2026-08
  tier-enthusiast re-profile campaign runs, and the capability matrix
  published on GitHub Pages is what those v8 profiles feed. The campaign
  writes to a new dated evidence directory; the existing one and its
  errata stand untouched, because evidence is not rewritten.

## v0.8 (v1.6 fast-follow): the tools probe records its own ceiling

The tools probe caps every reply at 256 generated tokens, and through
v0.7 a reply that hit that ceiling was indistinguishable in the profile
from one that stopped on its own — a result-use miss on a cut-off turn
read exactly like a miss with headroom. The `tools` family now carries
`n_truncated` (scored turns whose reply reported `stop_reason:
"length"`) and `n_stop_unreported` (scored turns whose backend reported
no stop reason at all), and the `tool_calling` lens repeats both beside
the rates. The rates themselves DO NOT move: the rubric's reading of a
truncated miss stands as pre-registered — a model told to quote the
result token that rambles past the ceiling has still not quoted it.
These are ambient facts of the readout, recorded so a reader weighing a
miss can see the ceiling beside it. `0` is a measured zero (every
scored turn was inspected; the unreported case has its own counter);
`None` is a profile written before the counters existed, or a probe
that never scored a turn. Profile schema is now **7** (package 0.8.0);
the committed tools-anchor results gained both counters by replaying
the committed transcripts through the updated probe. The version ledger
also moved out of the README into this file in the same release.

## v0.7 (v1.6): tool calling, and what a model does with a rejected patch

Through v1.5, every family that put a question to the model scored what
it **wrote**. Two of the failures that actually end agent runs are not
writing failures at all: a model that cannot emit a well-formed function
call, and a model that answers "SEARCH text not found" by re-sending the
same block. v1.6 measures both (package 0.7.0, schema v6).

- **Native tool calling** — `scripted-tools-v1`: five heterogeneous
  tasks, two turns each, against a frozen three-tool `toolset-v1`. Four
  rates (`call_rate`, `right_tool_rate`, `args_valid_rate`,
  `result_use_rate`) plus the `composite` a new `tool_calling` verdict
  ladders on. An endpoint that refuses the tools parameter is
  `unsupported` — a measured capability, not a gap in the run. See
  [Native tool calling](#native-tool-calling).
- **The loop's error script** — `scripted-loop-v1` becomes
  `scripted-loop-v2`. Every run now also plays a two-turn script in
  which the model's patch has **failed to apply**, and scores
  `recovery_rate` and `doom_loop_rate` over an explicit `n_error_runs`.
  Its turns join the shared action-fidelity, repeat and anchor
  denominator, so `n_turns` goes 3 → 5 per run. A measured recovery
  below 0.5 demotes `loop_discipline` off `ready`. See [The scripted
  repair loop](#the-scripted-repair-loop).
- **MoE-aware geometry** — `expert_count` / `expert_used_count` where
  the metadata states them, and `head_dim` now prefers the model file's
  stated `attention.key_length` over the embedding÷heads derivation,
  which is wrong by 2x on architectures that size attention
  independently. The kv formula is unchanged and now says why: it is
  expert-invariant by design, not by omission. See [MoE
  geometry](#moe-geometry).
- **`unsupported` ranks in the gate** — the verdict ladder gains a
  bottom rung below `unusable`, so `ready → unsupported` is a regression
  `assay diff --gate` fails on and `unsupported → ready` is an
  improvement it does not. The refusal is a measurement, so it ranks
  rather than dropping.
- **A live anchor for all three** — a real refusal (gemma2:9b), a real
  doom loop (3 of 3), a model that speaks tools and never calls one, and
  the MoE metadata read off the daemon, captured 2026-08-16 and committed
  under `docs/superpowers/evidence/tools-anchor/` with the acceptance
  tests that replay it offline. The `head_dim` fix turned out to be an
  erratum against two committed v1.4 profiles; the anchor quantifies it.
  See [The live anchor](#the-live-anchor).

One schema irregularity is recorded rather than tidied away, on the same
principle as v1.5's verdict amendment:

> Geometry's two expert keys landed one commit **before** the version
> bump, so for that window `assay_profile_version: 5` covered two
> geometry shapes — with and without `expert_count` /
> `expert_used_count`. No published profile is inside it, and
> `Profile.from_json` defaults both keys to `None` so either shape
> parses; the note exists so a reader who finds a v5 geometry carrying
> expert keys does not have to rediscover why.

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

