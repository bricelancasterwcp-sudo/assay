# Carried debt — v1.7 (recorded 2026-08-17 at the wave's close)

Known gaps deliberately carried out of v1.7, with the rulings that
carried them. None blocks the release.

**This file is self-contained.** Every deferred item and every ruling
recorded during the wave is written out below; nothing is delegated to a
pointer. The raw process record it was derived from is committed
verbatim beside the plan at
[`superpowers/plans/2026-08-17-assay-v1.7-consumers-ledger.md`](superpowers/plans/2026-08-17-assay-v1.7-consumers-ledger.md)
— that is provenance for the derivation, not a place detail hides.

**Reading the entries.** An item ~~struck through~~ was **closed** during
the wave, and the text that closed it follows in bold. An item left
plain is **open**. Nothing is deleted: an item that was raised and closed
is a different fact from one that was never raised, and a reader must be
able to tell them apart. The `(T<n>)` marks the task that raised it.

---

## What this slice settled

v1.7 shipped six instrument components and the evidence that exercises
them. **Sequential tools sampling** grew the pool 5 → 20
(`scripted-tools-v2`, `toolset-v1` frozen) and walks looks {5, 10, 20},
stopping at the first look whose Wilson-95 interval ladders both
endpoints to one rung. **Deep JSON grades** (`codec-fixtures-v3`) added
`nested` / `tabular` / `constrained` as new *columns* beside the three
size grades, leaving `structured_extraction`'s cell exactly where it
was. The **`parallel` family** reports what k ∈ {2, 4} concurrent
requests do to one endpoint — measurement-only, no verdict, because
there was no measured floor to ladder one against. **Budget mode**
(`--budget-calls N [--budget-seconds S]`) answers the settings-time
question by running families in a pre-registered priority, preflighting
each against its own *declared* worst case and dropping what does not
fit **by name**. The **derived cost table** (`run.WORST_CASE`) prices
every family from the constants its own probe consumes, which is what
moved the full/thorough default 500 → 610 twice by measurement rather
than by guess. And the **matrix build** (`scripts/build_matrix.py`)
renders a directory of profiles into one deterministic committed page.
All of it was then run for real: the **2026-08-17 campaign** profiled
fifteen models at `--full` on one enthusiast-16gb box in 3h07m, wrote
fifteen v8 profiles and fifteen transcripts, and published them as the
capability matrix; the **`tools-anchor-v2`** live anchor pins
`scripted-tools-v2` against a verbatim slice of that campaign's own
`llama3.1:8b` transcript, so the anchor and the profile are one
measurement reached by two roads and cannot drift apart without a test
failing.

Fourteen tasks, 729 → 985 tests.

---

## Standing rulings (do not re-litigate without a recorded amendment)

Every ruling made during the wave, in the order made. "Cost if wrong"
was recorded for each at the time and is preserved where it bites.

1. **A perfect pool does not decide early.** `decided(n, n)` is False at
   5, 10 and 20 (Wilson lowers 0.566 / 0.722 / 0.839, all under the 0.9
   `ready` floor); only 35/35 decides `ready`. The spec asks only that
   sampling stop at the first look whose interval decides a rung, and
   does not promise a perfect pool decides at all. Early stopping is
   exercised by the *unusable* pool (`decided(0, 5)` True). **(T0)**
2. **CORRECTION to ruling 1, entered at T10 and left standing as the
   record.** Ruling 1 originally read that the early-decide benefit
   "holds for unusable *(and narrowly risky)*". **The "narrowly risky"
   hedge is wrong**: no composite decides `risky` at any look ≤ 20,
   because `lo ≥ 0.6` and `hi < 0.9` never overlap. The reviewer's
   enumeration of the decided set — n=5 {0}, n=10 {0,1,2}, n=20 {0..7}
   — is the record. **(T10)**
3. **T1's planned distinctness test was arithmetically impossible.**
   `{(tool, args)} == 20` cannot hold beside `run_tests >= 4` (no-arg
   entries collapse to one key; max 17). Substitution accepted as the
   spec's intent: 15 distinct `(tool, args)` pairs among the
   arg-pinning tasks, 20 distinct messages, plus an isascii pin. **(T1)**
4. ~~**Default full/thorough `max_calls` 500 → 600**, because a clean
   full run measured 546 and ~10% headroom covers ceiling-bisection
   extras; tokens untouched, quick untouched.~~ **SUPERSEDED by ruling
   10.** **(T3)**
5. **Deep JSON grades always use the built-in directive.**
   `CodecDirectives.json_object` substitutes for the FLAT grades only;
   consumer substitution for deep grades is future scope. **(T3)**
6. **The parallel family has NO warmup call**, against a spec §3 that
   said "~1 warmup + 6 lane calls". `probe_parallel` requires a same-run
   speed baseline, so the model is already warm; a warmup would be a
   charge that measures nothing. Spec amended non-silently in T10's docs
   sweep. **(T5)**
7. **`total_throughput_tps` is `None` unless EVERY returned lane
   reported server timings**, and is the sum over returned lanes when
   they all did. `per_lane` stays a mean over reporting lanes with its
   evidence class (the speed family's honest-partial idiom). **No
   `n_lanes_reporting` field** — the `None` rule removes the ambiguity
   that a count would have papered over. **(T5)**
8. **Partial-k refusal is named in the document, not inferred from an
   absent row**: `Parallel.skipped`, mirroring `LongOutput.skipped`.
   Fixed in one round together with the `render_table` inversion,
   because the campaign's evidence would otherwise have carried both
   gaps. **(T6)**
9. **`BudgetMeter.would_exceed_n(calls, prompt_tokens)` covering ALL
   limits including seconds** was mandated as a T9 pre-step, because
   `parallel._affordable` hand-rolled its limit check and knew nothing
   of the clock — once `--budget-seconds` was wired, a mid-k trip would
   have recorded lanes that never launched *and* discarded measured rows
   behind a false drop line. **(T7)**
10. **Default full/thorough `max_calls` 600 → 610** (552 × 1.1 rounded
    up): 552 measured left 48 against a pre-registered ≥50-call headroom
    invariant, so the same knob is raised by the same derivation rather
    than the invariant being weakened. **(T6, ratified)**
11. **`calibrate()` charges 2 calls, not 1** — the plan's cost table was
    wrong and the corrected sum (552, exactly) is what proves the
    table against the metered run. **(T8)**
12. **The budget-mode ceiling preflight reserves
    `worst_case_calls("ceiling") + ceiling.bisection_worst_case_steps`**
    — "what starts, finishes" includes the bisection tail, and a family
    admitted on ladder-only cost could die mid-bisection. **(T8)**
13. **Budget mode narrows the ceiling ladder by `training_ctx`** (as
    `--full` does and `--quick` does not) — a deliberate difference,
    recorded in provenance and stated in the CHANGELOG, so a lower
    `ceiling.max_verified` in a budget profile reads as a mode artifact
    rather than a regression. **(T9)**
14. **T9's deferred README "what starts, finishes" minor GRADUATED into
    T10's fix round**, because the same false absolute had reached the
    release entry and fixing one site but not the other would be
    incoherent. Both sites now qualify it: the preflight guarantees
    CALLS; the token meter can still cut mid-family; the wall clock is
    checked between calls. **(T10)**
15. **T11's brief authored a contradiction** — a "byte-for-byte
    identical on `None` args" clause beside a mandated per-row `probe=`
    addition. Resolved in the only coherent direction (the content is
    mandated in, so the docstring must say so). Recorded as a **plan
    defect**, not an implementer error. **(T11)**
16. **The E1 sweep's completeness claim was rescoped to E1's blast
    radius by `probe_version`** rather than left to rot, with the
    original text preserved and a dated amendment beside it; PROTOCOL.md
    untouched as pre-registered. Enforced by test, not asserted. **(T12)**
17. **`qwen3:14b` was re-run clean from zero, not resumed** — an exit-4
    infrastructure kill with no profile written is the house rule's
    rerunnable case. Pre-registered: a second HTTP 500 would have been a
    **FINDING** (ship 14 rows plus a named absence), not another
    attempt. **(T13)**
18. **CARRIED-DEBT.md is self-contained and the raw wave ledger commits
    verbatim to the plans directory.** A debt file that delegates to a
    gitignored worktree-local path dies with the worktree. Cost if
    wrong: some absorbed lines are noise — cheaper than a dead pointer.
    **(T14)**

---

## Deferred, by area

### Tools

1. **Pool-to-35 is a decided-`ready` option, priced and not taken.** By
   ruling 1, `ready` is unreachable non-provisionally under looks
   {5, 10, 20}: a perfect 20/20 reads `ready` provisional at Wilson
   lower 0.8389 against the 0.9 floor. n = 35 is the smallest n at which
   a perfect cell clears `ready` undisputed and this pool has no 35th
   task. **Ruling: not smuggled into v1.7.** Cost of the option: +30
   calls on full (15 tasks × 2 turns) over today's 40. **(T0/T2)**
2. `probe_tools` has no empty-schedule guard, where `codecs` refuses
   `()` with a `ValueError` — asymmetric across a shared renderer. **(T2)**
3. Budget-death mid-schedule, at a non-look-point n, is pinned by no
   test. **(T2)**
4. The tools rates appear a **third** time in the verdicts lens,
   unpinned against the other two copies. **(T13)**
5. The `look_schedule`-live asymmetry is commented but not otherwise
   reconciled. **(T13)**
6. `tests/test_tools.py` is 1112 lines — a split candidate. **(T13)**
7. ~~Look literals (5, 10, 20) hardcoded at `tests/test_tools.py:243`;
   T2 must touch them.~~ **CLOSED at T2** — the prefix test now iterates
   `TOOLS_LOOK_SCHEDULE`, and the surviving literal at line 280 is a
   deliberate pin on the constant, not a duplicated magic number. **(T1)**

### Codecs and fixtures

8. **Three grade-ordering surfaces agree only by coincidence** —
   `profile.py` follows the live `GRADES`; `report.py` and `diff.py`
   freeze the triple. The fix is a cross-surface equality test over a
   deep fixture; nothing fails today if they diverge. **(T4)**
9. `PATCH_CODECS` is defined **negatively**, so a future codec joins the
   patch set silently. **(T8)**
10. The subset-order and subset-seed invariants are documented but
    unpinned. **(T8)**
11. The nested `lon` fixture lacks a wrong-type row. **(T3)**
12. No test asserts that an unmeasured deep cell reaches `run.py`'s
    `dropped`. **(T3)**
13. The test-side directive-routing helpers are duplicated four times.
    **(T3)**
14. ~~`profile._render_codecs` and `report._codec_grid` hardcode
    tiny/small/medium, so deep cells are omitted from every render.~~
    **CLOSED at T4** — a shared derived grade-column idiom, with the old
    three-grade rendering sha256-stable across all 20 committed
    profiles. **(T3)**

### Parallel

15. **No verdict floors — and they are now derivable.** v1.7 shipped
    `parallel` as measurement-only on the stated ground that a rung
    invented without a measured floor is the overclaim the rest of the
    schema exists to refuse. That ground is gone: fifteen live rows now
    exist, all reading `mode: parallel` at both k = 2 and k = 4, with
    `degradation_ratio` between 0.995 and 1.007 across a 10× span of
    single-lane speed (28 → 288 tok/s). **The natural v1.8 opener.**
16. **The overlap tolerance sanity check ran, and the flag stands.**
    `OVERLAP_TOLERANCE_S` is 0.25 s; every campaign profile records
    `tolerance_provenance: "chosen-2026-08-17"`. **No real endpoint in
    the campaign read `serialized`** — fifteen models × two k values,
    `n_lanes_ok` full, `lane_errors` empty, `skipped: []` throughout. So
    the constant is consistent with every live row and **exercised at
    its edge by none of them**, which is a weaker claim than a derived
    threshold. Retiring the flag needs an endpoint that actually
    serializes; this tier has not produced one.
17. Evidence-class strings are re-declared here against `speed.py`'s
    inline literals; the fix is a shared tuple in `speed.py`
    (scope-blocked at the time). **(T5)**
18. The `baseline is None` guard is missing — a `TypeError` after spend,
    rather than a clean error. **(T5)**
19. The 0.0-baseline branch is untested. **(T5)**
20. The runner's `None`-filter would renumber lanes if it ever fired.
    **(T5)**
21. `thread.start()` failure is unhandled. **(T5)**
22. The runner docstring gives a rationale for a use that does not
    exist. **(T5)**
23. The synthetic runner catches `Exception` where `BaseException` is
    the honest net. **(T5)**
24. The tolerance is not injectable from the public entry point. **(T5)**
25. ~~`_affordable` re-derives the budget rule outside the meter.~~
    **CLOSED at T9** — it consults `BudgetMeter.would_exceed_n`, which
    covers all limits including seconds (ruling 9). **(T5/T7)**
26. ~~A k=2-paid / k=4-refused run leaves k=4 absent from `rows` with
    nothing in `dropped`.~~ **CLOSED at T6** — `Parallel.skipped` names
    every refused k (ruling 8); the fix also corrected the
    all-ks-refused render case. **(T6)**
27. ~~`render_table` shows the parallel family ONLY when it is
    unmeasured — an inversion, unassigned in the plan.~~ **CLOSED at
    T6** — `render_table` gained the parallel line in the sibling
    idiom. **(T6)**

### Budget mode and the cost table

28. **The ceiling preflight prices the MODE cap, not the effective
    (`training_ctx`-narrowed) one**, so budget mode over-reserves by
    roughly 50% on small-context models. Deliberate: a preflight must
    reserve the worst case it *declares*, and reserving against the
    narrowing would admit a family it could not always pay for. **(T8/T9)**
29. **"What starts, finishes" is a CALL-meter guarantee only**, and
    **no test exercises the honest partial**: an admitted family can
    still die on the token meter mid-run and keep what it measured. The
    docstring says so (ruling 14 fixed the README and CHANGELOG); the
    test does not exist. **(T9)**
30. An asymmetric codec-half drop would leave unnamed placeholders —
    latent only because both halves cost 30 by accident. **(T9)**
31. `--max-calls` is silently overridden by `--budget-calls` — an idiom
    mismatch against every other flag conflict. **(T9)**
32. `would_exceed`'s docstring still reads as a reliable preflight
    although wall-clock passes between check and charge. **(T7)**
33. Both-limits-trip precedence is unpinned. **(T7)**
34. `budget.py:106` carries a garbled docstring sentence. **(T7)**
35. `test_budget.py:72` snapshots field-by-field where `replace()` is
    the idiom. **(T7)**
36. The seconds-unmeasured discriminator comment is incomplete (the true
    predicate is `max_seconds` set AND `calls > 0`). **(T7)**
37. `cli.py:518`'s comment reads loosely against the narrowed
    probe-only condition. **(T9)**
38. Ceiling truthiness is used where `is None` is meant. **(T9)**
39. The `budget == quick` params assertion is a tautology. **(T9)**
40. Dead backend setup lingers in the flag-conflict test. **(T9)**
41. The empty-only check sits after the unknown-name check and reads
    backwards. **(T8)**
42. `_MEASURED_CLEAN_FULL_RUN` is a third copy of the literal 552. **(T8)**
43. **Quick's margin is 6 calls (124 worst case of 130) — a live
    constraint, not a note.** The deep grades cost quick 15 codec calls
    and the next quick-mode call added anywhere spends the remainder.
    Full has headroom by derivation; quick does not. **(T3/T6)**
44. The headroom prose double-counts the bisection: 546 already
    contained ~12 ladder calls, so the true worst case was ~553 and the
    cost-narrative clause built on it was false. **(T3)**
45. `ceiling_cap_for` **branches on the mode STRING rather than on the
    params** (`if mode != "quick"`) — this is the CAUSE behind the diff
    findings at item 48: quick alone skips the `training_ctx` narrowing
    because it is singled out by name, which is why quick's ladder
    marched past `gemma2:9b`'s 8192 trained window and why quick and
    full report different caps for the same endpoint. **(T8)**
46. ~~`_WORST_CASE` is private but is an interface; T9 wants it public.~~
    **CLOSED at T9** — `run.WORST_CASE` and `worst_case_calls()` are
    public, and budget mode's preflight consumes them. **(T8)**
47. ~~Full-mode spend 441 of 500, and parallel will add ~7 — watch the
    500 ceiling.~~ **CLOSED at T3 and T6** — measured 546, then 552;
    the default moved to 610 by rulings 4 and 10, with the derived table
    (item 46) as the mechanism. **(T2)**

### Ceiling, geometry, and what the campaign's own diffs found

These three came out of reading the fifteen per-model diffs at T14.
Detail and evidence:
[`superpowers/evidence/tier-enthusiast-2026-08/diffs/README.md`](superpowers/evidence/tier-enthusiast-2026-08/diffs/README.md).

48. **`diff` scores a ceiling-CAP change as a capability improvement.**
    Twelve of fifteen diffs read
    `ceiling.max_verified: 16384 -> 32768 (improvement, rung-change)`
    where *both* sides read `failure_mode: none_up_to_cap` — neither run
    found a ceiling, and what moved was quick's cap versus full's. A
    `rung-change` across two runs with different caps and identical
    `none_up_to_cap` modes is not a comparison, and `diff` has no term
    for it. **(T14)**
49. **The ceiling ladder cannot tell a daemon telemetry dropout from a
    context failure.** `mistral-nemo:latest` returned a reply with no
    `tokens_in` / `tokens_out` / `stop_reason` at est_tokens 4096, seed
    0 (`missing_stats`); the ladder read it as a failure, bisected, hit
    it again at 3584, and settled `max_verified` 3328 — publishing
    `long_context: ready → risky`. Seed 1 passed 4096 cleanly and the
    shape ladder is byte-identical to the previous campaign at all three
    shapes. **Ruling: published exactly as measured**, with the
    explanation filed beside the evidence. Same daemon-flakiness class
    as the campaign's transient HTTP 500. **(T14)**
50. **`gemma-4-12b-it-qat-q4_0:latest` still reports `geometry: null`**
    at probe 0.9.0, exactly as at 0.5.0 — and it is the model whose
    stated `attention.key_length` (512) diverges most from the
    derivation (240, a 2.1× gap). Not an E1 case: there is no committed
    kv number to correct. The one model on the published matrix with no
    geometry at all. **(T14)**

### Profile, renderers, report, matrix

51. The byte-identity test is single-process (structural sorting closes
    the real risk). **(T11)**
52. The `.intro` CSS ships on the `None` path — one unmandated byte; a
    conditional include restores identity. **(T11)**
53. `_ERRATA_HREF` is hardcoded to the default `--out` (deliberate, and
    commented). **(T11)**
54. `UnicodeDecodeError` escapes the `BuildError` taxonomy. **(T11)**
55. `_dates`' `str[:10]` coercion is undocumented. **(T11)**
56. The `"0.5.0 in head"` assertion is weak (covered in substance by the
    per-row pins). **(T11)**
57. `report._grade_columns`' half-guard skips non-dicts while the caller
    still raises. **(T4)**
58. The mixed-page grid is identified by position rather than by model.
    **(T4)**
59. `_GRADES_V4` is defined after its use. **(T4)**
60. The changes-non-empty vacuity guard is missing. **(T4)**
61. The published matrix page was **not visually opened** during T14 —
    its content was verified by parsing, not by eye. The final review
    takes the look. **(T14)**
62. ~~The report page carries SCHEMA versions but not `probe_version`,
    so a row measured by an older instrument is indistinguishable.~~
    **CLOSED at T11** — every row's detail block names the probe
    version, and the intro says so. **(T4)**
63. ~~The intro's provenance sentence drops field-less profiles
    silently — 6 committed profiles already trigger it.~~ **CLOSED at
    T11** — each fact carries its own missing count. **(T11)**
64. ~~`render_report`'s docstring claims byte-for-byte identity that the
    commit had just made false.~~ **CLOSED at T11** (ruling 15) — the
    docstring names the two deliberate additions. **(T4/T11)**

### Campaign and evidence operations

65. The derivation script that cut the `tools-anchor-v2` transcript
    slice is **not committed** — the route is not reproducible, and the
    drift test carries the guarantee instead. **(T13)**
66. `campaign.pid` is never removed. **(T12)**
67. The run log is appended and never rotated. **(T12)**
68. A skip-92 (probe exited 0 but the profile names a different model)
    is logged with the verb `done`, and the mismatched profile is left
    on disk. **(T12)**
69. `tests/test_e1_sweep.py` reaches `KeyError` / `ValueError` where a
    named assertion belongs. **(T12)**
70. `e1-sweep/results.json`'s metadata fields are unpinned. **(T13)**
71. The T12 report mis-claimed the coder-7b/14b tps rows were the only
    ones selected — the selection was conservative but the basis went
    unnamed. **(T12)**
72. One README link uses the old slug convention (both resolve). **(T12)**
73. `ASSAY_COMMIT` merges stderr into its value on failure. **(T12)**
74. The campaign corpus glob in `tests/test_profile.py` lacks the
    version-key filter its sibling corpus has, so a non-profile JSON
    landing in that directory would `KeyError` rather than fail a named
    assertion. **(T14)**
75. ~~The "idle daemon" precondition is actually implemented as **make
    idle**, undocumented in the wrapper's deliberately-absent block.~~
    **CLOSED at T14** — `scripts/campaign-2026-08.sh`'s header now
    states that `unload_all` evicts rather than waits, that the script
    therefore needs an exclusive window, and that exits 90/91 refuse
    rather than block. **(T12)**
76. ~~The campaign wrapper is pinned to a `/tmp` session venv while a
    durable worktree `.venv` exists, and the durable log does not pin
    the instrument's version or commit.~~ **CLOSED at T12** — the
    wrapper uses `$REPO/.venv` and the run log opens with
    `assay_bin` / `assay_version` / `assay_commit`. **(T12)**
77. ~~The disturbed dry-run row is unmarked in the evidence, and the
    justification offered for it was unsound.~~ **CLOSED at T12**, then
    **superseded at T13** — the sound defence (the speed family's
    no-stall signature) is filed, and the campaign's own undisturbed run
    of that tag replaced the row. **(T12)**

### Documentation prose

78. README ragged wrap at lines 246–247. **(T10)**
79. `README:110`'s worst-case account omits the tools-cap and parallel
    terms. **(T10)**
80. The full-vs-quick sampling framing omits budget mode (also fixed-n).
    **(T10)**
81. "at least four times" is weaker than the measured five. **(T10)**
82. `run.py:896`'s bare "What starts, finishes." is paragraph-scoped
    correctly but unqualified on its own line. **(T10)**
83. CHANGELOG uses "TOKEN" caps where the README uses bold. **(T10)**
84. **"Thirty lane readings" is loose wording** in the CHANGELOG and the
    diffs read-out: thirty is the count of k-readings (15 models × 2 k
    values); the lanes themselves number 90. This file says it
    correctly; those two do not. **(T14)**
85. **CARRIED-DEBT.md is linked from nothing.** A durable ledger nobody
    can find is halfway to a dead pointer. Deliberately not fixed here
    to keep the fix round's scope tight; ledgered for the final review.
    **(T14)**
86. The T14 commit subjects were the implementer's own grouping rather
    than the plan's stated subject line, and the substitution went
    undisclosed at the time. **(T14)**
87. ~~README's "Native tool calling" section and the CHANGELOG still say
    `scripted-tools-v1` / five tasks.~~ **CLOSED at T10's docs sweep**,
    which also found and fixed two further stale claims beyond the list
    it was given. **(T1)**
88. ~~The spec's line 72 says `sequential-{5,10,20}` against the shipped
    `wilson95-looks-5-10-20`.~~ **CLOSED at T10** via a non-silent
    footnote amendment. **(T2)**
89. ~~README:649's measured cost is stale (411/218,037).~~ **CLOSED at
    T2** — re-measured to 441/226,009, and superseded again by the 552
    figure at T6. **(T2)**
90. ~~README:124's codec lens row says three size grades, and "size" is
    wrong for json's six.~~ **CLOSED at T3.** **(T3)**
91. ~~`cli.py:44` hand-counts the bisection tail at "~7" against a
    derived 4 quick / 8 full, and `run.py:148` cites that comment as
    authority.~~ **CLOSED at T8** — all four sites re-derived, with a
    repo-wide grep clean. **(T8)**
92. ~~The `TASKS` docstring overstates the per-prefix counts.~~
    **CLOSED at T1.** **(T1)**
93. ~~CHANGELOG and README claim "clearly risky pools can decide".~~
    **CLOSED at T10** (ruling 2) — the claim was false and the
    enumeration replaced it. **(T10)**
94. ~~Budget flags are registered on all five subcommands while only
    `probe` implements the mode — a false help promise that makes the
    family subcommands started-and-truncated.~~ **CLOSED at T9.** **(T9)**
95. ~~Ruling 13's quick-vs-budget ceiling-cap difference is recorded
    only in a task report.~~ **CLOSED at T9** — it is in the CHANGELOG.
    **(T9)**

### Test hygiene

96. No mutation check on `test_the_pool_mixes_tools_and_argument_values`.
    **(T1)**
97. The "15 pairs" figure is derived, not asserted. **(T1)**
98. The isascii law lives inside the verbatim test's *name*. **(T1)**
99. A duplicate `>= 4` assertion. **(T1)**
100. The task-7 phrasing echoes another entry, and the task-11 phrasing
     is the least natural in the pool (both brief-supplied). **(T1)**
101. `CallRecorder`'s write lock is untested — `test_replay` has no
     concurrency case. **(T6)**
102. The report escape-totality test covers the `mode` cell but not the
     `evidence` cell. **(T6)**
103. ~~`tests/test_run.py:29` carries a stale 546 comment.~~ **CLOSED at
     T6** as a ride-along in that task's own mandated consistency sweep.
     **(T6)**
104. ~~`tests/test_cli.py`'s evidence-glob tuple lacks
     `tier-enthusiast-2026-08`.~~ **CLOSED at T14** — added, and a
     separate campaign corpus now runs all fifteen v8 profiles through
     `Profile.from_json` + both renderers, since the back-compat corpus
     is deliberately frozen to schema {1,2,3,4}. **(T13)**
105. ~~The `deepseek-r1` / `Hermes-4` envelope 0.0 with shape 30/30 is
     the first thing a reader hits and has no read-out.~~ **CLOSED at
     T14** — written up as the reasoning-preamble signature, with the
     tool-call counter-evidence, in the diffs read-out. **(T13)**

### Not attempted

106. **Multi-turn chains need their own wave.** Every family in v1.7
     scores a single request/reply or a two-turn tool exchange. Whether
     a model holds a plan across N turns — and where it stops holding it
     — is a different measurement with a different cost curve and a
     different failure taxonomy. **Not a bolt-on to any existing
     family.** It needs its own spec.

---

## Process lessons

1. **One fix round, fourteen times.** Every task that needed fixes
   needed *exactly one* round against an allowance of five; tasks 4, 7
   and 13 needed none, and nothing ever reached round 2. The mechanism
   was not luck: each dispatch brief carried the previous tasks'
   findings forward as explicit notes, so an implementer arrived already
   knowing the trap laid for it. The ledger's "Note for Task N dispatch:"
   lines are the artifact of that, and they are worth keeping in the
   next wave's format.
2. **A claim about what a rule DECIDES must be answered by enumerating
   the rule.** T10's review caught the CHANGELOG and README claiming
   "clearly risky pools can decide"; the reviewer enumerated the decided
   set over {5, 10, 20} and found only `unusable` ever decides. The
   controller's own pre-flight ruling had hedged "and narrowly risky" —
   **the hedge was wrong and the enumeration is the record** (ruling 2).
   T8 was the same shape (`cli.py` hand-counting bisection at "~7"
   against a derived 4 quick / 8 full), and so was T1's per-prefix
   docstring. Prose review passes all three; enumeration catches all
   three.
3. **The budget tripwire, and why it fired twice.** T3 measured 546
   calls against a 500-call default and raised it as a BLOCKED item
   rather than shipping a default that drops families mid-suite; T6 did
   it again at 552 against 600. Both raises were *derived* — the
   measured clean run plus a pre-registered ≥50-call headroom invariant
   — not guessed. It fired because the dispatch mandate required every
   task that grew a family to re-measure the clean-run spend, so growth
   surfaced at the task that caused it instead of at the first user.
4. **A scope claim that time can falsify needs a test, not better
   wording.** The E1 sweep asserted it covered "every committed profile
   in the repository". True the day it was filed, false the moment a
   0.9.0 profile landed. T12 bounded it with a *dated amendment* —
   original text left as filed, `PROTOCOL.md` untouched as
   pre-registered — and then made the bound **enforced**:
   `tests/test_e1_sweep.py` recomputes the swept set from the tree by
   `probe_version`, and a companion test requires every unswept profile
   to justify itself with `probe_version >= 0.7.0`, so a stale profile
   breaks one test or the other rather than slipping between them.
5. **Rerun, don't resume — and pre-register what a second failure
   means.** The campaign's one failure (`qwen3:14b`, transient HTTP 500,
   exit 4, **no profile written**) was an infrastructure kill with no
   numbers read, so the house rule's rerunnable case applied and it
   restarted from zero. What made it safe rather than a retry loop was
   deciding *in advance* that a second 500 would be a FINDING to ship —
   fourteen rows and a named absence — not another attempt (ruling 17).
6. **Evidence is not tidied, at any of the four chances taken to tidy
   it.** The dry-run row's disturbance stayed documented after the
   campaign superseded it, because a limit deleted the moment it stops
   applying teaches nobody how the next one was caught. The
   mistral-nemo telemetry dropout ships as a published `risky` with the
   explanation beside it. The E1 profiles stand as measured with an
   errata file rather than corrected in place. And the fifteen diffs are
   committed with their confounds *named* rather than filtered.
7. **A plan can author its own contradiction, and saying so is cheaper
   than absorbing it.** Two of this wave's rulings (3 and 15) exist
   because a task brief asked for something arithmetically or logically
   impossible. Both were recorded as **plan defects** rather than
   implementer errors — which is what let the second one be spotted as a
   *pattern* rather than a one-off, and is why the pre-flight conflict
   scan is worth its cost in the next wave.
8. **A debt file that delegates is a debt file that dies.** This
   document's first draft pointed at the SDD ledger for detail. That
   path is gitignored and worktree-local: the pointer would have been
   dead the moment the worktree was removed. Ruling 18 is the fix, and
   the general form is that a durable artifact may cite an ephemeral one
   for *provenance* but never for *content*.
