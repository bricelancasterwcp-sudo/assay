# Carried debt — v1.7 (recorded 2026-08-17 at the wave's close)

Known gaps deliberately carried out of v1.7, with the rulings that
carried them. None blocks the release. This is the durable list; the
per-task detail — every deferred minor, every ruling, every fix round —
lives in the wave ledger at
`.superpowers/sdd/2026-08-17-assay-v1.7-consumers/progress.md`.

Nothing here is deleted on a later wave. An item is struck through with
what closed it, so the record of what was carried, and for how long,
survives.

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

---

## Deferred, with rulings

### Parallel

1. **No verdict floors — and they are now derivable.** v1.7 shipped
   `parallel` as measurement-only on the stated ground that a rung
   invented without a measured floor is the overclaim the rest of the
   schema exists to refuse. That ground is now gone: fifteen live rows
   exist, all reading `mode: parallel` at both k = 2 and k = 4 with
   `degradation_ratio` between 0.995 and 1.007 across a 10× span of
   single-lane speed (28 → 288 tok/s). **This is the natural v1.8
   opener** — the floors a verdict would ladder against can be derived
   from these thirty readings instead of chosen.
2. **The overlap tolerance sanity check: the flag stands.**
   `OVERLAP_TOLERANCE_S` is 0.25 s and every campaign profile records
   `tolerance_provenance: "chosen-2026-08-17"`. The live rows were
   supposed to sanity-check it and did: **no real endpoint in the
   campaign read `serialized`.** All fifteen models, both k values,
   `n_lanes_ok` full, `lane_errors` empty, `skipped: []`. The constant
   is therefore consistent with every live row and *exercised at its
   edge by none of them* — a weaker claim than a derived threshold.
   **Ruling: it stays flagged as chosen.** Retiring the flag needs an
   endpoint that actually serializes, which this tier does not have.
3. Smaller, all recorded at Tasks 5–6: evidence-class strings are
   re-declared here against `speed.py`'s inline literals (the fix is a
   shared tuple, scope-blocked at the time); `_affordable` re-derives
   the budget rule outside the meter (Task 9's `would_exceed_n` closed
   the seconds hazard, the calls path still hand-rolls); the
   `baseline is None` guard is missing (TypeError after spend, rather
   than a clean error); the 0.0-baseline branch is untested; the
   runner's `None`-filter would renumber lanes if it ever fired;
   `thread.start()` failure is unhandled; the tolerance is not
   injectable from the public entry point.

### Tools

4. **Pool-to-35 is a decided-`ready` option, priced and not taken.**
   Enumerated over looks {5, 10, 20}, the composites that decide are
   *exactly* the `unusable` ones — 0/5, 0–2/10, 0–7/20 — and nothing
   decides `risky` at any look, because no pass count at n ≤ 20 has an
   interval that fits inside [0.6, 0.9). A *perfect* pool does not
   decide either: Wilson's lower bound on 20/20 is 0.8389 against the
   0.9 `ready` floor, so it runs to the cap and reads `ready`
   provisional. n = 35 is the smallest n at which a perfect cell clears
   `ready` undisputed, and this pool has no 35th task. **Ruling: not
   smuggled into v1.7.** The option is a 35-task pool costing full mode
   +30 calls (15 tasks × 2 turns) over today's 40; it buys a
   non-provisional `ready` and nothing else.
5. Smaller: `probe_tools` has no empty-schedule guard where `codecs`
   refuses `()` with a `ValueError` (asymmetric across a shared
   renderer); budget-death mid-schedule at a non-look-point n is pinned
   by no test; the tools rates appear a third time in the verdicts lens,
   unpinned; the look literals (5, 10, 20) are hardcoded in the tests;
   `tests/test_tools.py` is 1112 lines and is a split candidate.

### Codecs and fixtures

6. **Consumer directive substitution covers FLAT grades only.** The deep
   json grades always use the built-in directive.
   `CodecDirectives.json_object` substitutes for tiny/small/medium and
   nothing else. Ruling recorded at Task 3: consumer substitution for
   deep grades is future scope, not a gap in this one.
7. **Three grade-ordering surfaces agree only by coincidence.**
   `profile.py` follows the live `GRADES`; `report.py` and `diff.py`
   freeze the triple. The fix is a cross-surface equality test over a
   deep fixture; today nothing would fail if they diverged.
8. Smaller: `PATCH_CODECS` is defined *negatively*, so a future codec
   joins the patch set silently; the subset-order and subset-seed
   invariants are documented but unpinned; the nested `lon` fixture
   lacks a wrong-type row; no test asserts that an unmeasured deep cell
   reaches `run.py`'s `dropped`; the test-side directive-routing helpers
   are duplicated four times.

### Budget mode

9. **The ceiling preflight prices the MODE cap, not the effective one.**
   Budget mode narrows the ladder by `training_ctx` at run time but
   reserves against the mode cap, so on a small-context model it
   over-reserves by roughly 50%. Deliberate at Task 9: a preflight must
   reserve the worst case it *declares*, and a reservation that assumes
   the narrowing would admit a family it could not always pay for.
10. **"What starts, finishes" is a guarantee on the CALL meter only.**
    It is the only ceiling a preflight can reserve against — the cost
    table declares calls and there is no per-family token declaration to
    hold back. An admitted family can still die on the TOKEN meter
    mid-run and be kept as an honest partial; the docstring says so and
    **no test exercises that partial**. Related and latent: an
    asymmetric codec-half drop would leave unnamed placeholders (today
    both halves cost 30 by accident, so it cannot fire).
11. Smaller: `--max-calls` is silently overridden by `--budget-calls`
    (an idiom mismatch); `would_exceed`'s docstring still reads as a
    reliable preflight although wall-clock passes between check and
    charge; both-limits-trip precedence is unpinned; the
    seconds-unmeasured discriminator comment is incomplete; `cli.py:518`
    reads loosely against the narrowed probe-only condition.

### What the campaign's own diffs found about the instrument

These three are new — they came out of reading the fifteen per-model
diffs at Task 14, not out of a task review. Detail and evidence:
[`superpowers/evidence/tier-enthusiast-2026-08/diffs/README.md`](superpowers/evidence/tier-enthusiast-2026-08/diffs/README.md).

12. **`diff` scores a ceiling-CAP change as a capability improvement.**
    Twelve of the fifteen diffs carry
    `ceiling.max_verified: 16384 -> 32768 (improvement, rung-change)`
    where *both* sides read `failure_mode: none_up_to_cap` — neither run
    found a ceiling, and the number that moved was quick's cap versus
    full's. `diff` has no term for "the cap moved" and reports a
    capability gain nobody measured. A `rung-change` across two runs
    with different caps and identical `none_up_to_cap` modes is not a
    comparison.
13. **The ceiling ladder cannot tell a daemon telemetry dropout from a
    context failure.** `mistral-nemo:latest` returned a reply with no
    `tokens_in`, `tokens_out` or `stop_reason` at est_tokens 4096,
    seed 0 (`missing_stats`); the ladder read it as a failure, bisected,
    hit it again at 3584, and settled `max_verified` 3328 — publishing
    `long_context: ready → risky`. Seed 1 passed 4096 cleanly and the
    shape ladder is byte-identical to the previous campaign at all three
    shapes. **Ruling: published exactly as measured**, with the
    explanation filed beside the evidence rather than the row being
    adjusted. A ladder that separates "the model failed" from "the
    daemon stopped reporting" is v1.8+ work. Same daemon-flakiness class
    as the campaign's one transient HTTP 500.
14. **`gemma-4-12b-it-qat-q4_0:latest` still reports `geometry: null`**
    at probe 0.9.0, exactly as it did at 0.5.0 — and it is the model
    whose stated `attention.key_length` (512) diverges most from the
    derivation (240, a 2.1× gap). Not an E1 case: there is no committed
    kv number to correct. A standing gap in metadata extraction, and the
    one model on the published matrix with no geometry at all.

### Not attempted

15. **Multi-turn chains need their own wave.** Every family in v1.7
    scores a single request/reply or a two-turn tool exchange. Whether a
    model holds a plan across N turns — and where it stops holding it —
    is a different measurement with a different cost curve and a
    different failure taxonomy. **Ruling: not a bolt-on to any existing
    family.** It needs its own spec.

### Docs, renderers, and campaign housekeeping

16. Renderers: the byte-identity test is single-process (structural
    sorting closes the real risk); the `.intro` CSS ships on the `None`
    path (one unmandated byte — a conditional include restores
    identity); `_ERRATA_HREF` is hardcoded to the default `--out`
    (deliberate, commented); `UnicodeDecodeError` escapes the
    `BuildError` taxonomy; `_dates`' `str[:10]` coercion is
    undocumented; `report._grade_columns`' half-guard skips non-dicts
    while the caller still raises; the mixed-page grid is identified by
    position rather than by model.
17. Prose: README ragged wrap at 246–247; the README worst-case account
    omits the tools-cap and parallel terms; the full-vs-quick sampling
    framing omits budget mode (also fixed-n); "at least four times" is
    weaker than the measured five; `run.py:896`'s bare "What starts,
    finishes." is paragraph-scoped correctly but unqualified on its own
    line; CHANGELOG "TOKEN" caps against README bold.
18. Campaign housekeeping: the derivation script that cut the
    `tools-anchor-v2` transcript slice is not committed (the drift test
    carries the guarantee, not the route); `campaign.pid` is never
    removed; the run log is appended and never rotated;
    `e1-sweep/results.json`'s metadata fields are unpinned; one
    README link uses the old slug convention (both resolve);
    `ASSAY_COMMIT` merges stderr into its value on failure.

---

## Process lessons

1. **One fix round, fourteen times.** Every task that needed fixes
   needed *exactly one* round against an allowance of five; tasks 4, 7
   and 13 needed none, and nothing ever reached round 2. The mechanism
   was not luck: each dispatch brief carried the previous tasks'
   findings forward as explicit notes, so an implementer arrived already
   knowing the trap that had been laid for it. The ledger's
   "Note for Task N dispatch:" lines are the artifact of that, and they
   are worth keeping in the next wave's format.
2. **A claim about what a rule DECIDES must be answered by enumerating
   the rule.** Task 10's review caught the CHANGELOG and README claiming
   "clearly risky pools can decide" — the reviewer enumerated the
   decided set over {5, 10, 20} and found that only `unusable` ever
   decides. The controller's own pre-flight ruling had hedged "and
   narrowly risky"; **the hedge was wrong and the enumeration is the
   record.** Task 8 was the same shape: `cli.py` hand-counted the
   bisection tail at "~7" against a derived 4 quick / 8 full, and a
   docstring overstated the per-prefix task counts at Task 1. Prose
   review passes all three; enumeration catches all three.
3. **The budget tripwire, and why it fired twice.** Task 3 measured 546
   calls against a 500-call default and raised it as a BLOCKED item
   rather than shipping a default that drops families mid-suite; Task 6
   did it again at 552 against 600. Both raises were *derived* — the
   measured clean run plus a pre-registered ≥50-call headroom invariant
   — not guessed. It fired because the dispatch mandate required every
   task that grew a family to re-measure the clean-run spend, so growth
   surfaced at the task that caused it instead of at the first user.
4. **A scope claim that time can falsify needs a test, not better
   wording.** The E1 sweep asserted it covered "every committed profile
   in the repository". That was true the day it was filed and false the
   moment a 0.9.0 profile landed. Task 12 bounded it with a *dated
   amendment* — original text left as filed, `PROTOCOL.md` untouched as
   pre-registered — and then made the bound **enforced**:
   `tests/test_e1_sweep.py` recomputes the swept set from the tree by
   `probe_version`, and a companion test requires every unswept profile
   to justify itself with `probe_version >= 0.7.0`, so a stale profile
   breaks one test or the other rather than slipping between them.
5. **Rerun, don't resume — and pre-register what a second failure
   means.** The campaign's one failure (`qwen3:14b`, transient HTTP 500,
   exit 4, **no profile written**) was an infrastructure kill with no
   numbers read, so the house rule's rerunnable case applied and it
   restarted from zero. What made this safe rather than a retry loop was
   deciding *in advance* that a second 500 would be a FINDING to ship —
   fourteen rows and a named absence — not another attempt.
6. **Evidence is not tidied, at any of the four chances taken to tidy
   it.** The dry-run row's disturbance stayed documented after the
   campaign superseded it, because a limit deleted the moment it stops
   applying teaches nobody how the next one was caught. The
   mistral-nemo telemetry dropout ships as a published `risky` with the
   explanation beside it. The E1 profiles stand as measured with an
   errata file rather than corrected in place. And the fifteen diffs are
   committed with their confounds *named* rather than filtered — the
   mode change, the fixture-set change and the cap change are stated at
   the top of `diffs/README.md`, so a reader can discount them instead
   of being protected from them.
