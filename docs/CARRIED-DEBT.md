# Carried debt — v1.10 (recorded 2026-08-18 at the wave's close)

Package 0.12.0. Profile schema unchanged at **v10** — this wave changes
how two profiles are COMPARED, not what a profile says. None blocks the
release.

`diff` gained `SEMANTIC_BREAKS` (`diff.py`): a registry naming cells
whose measurement rule changed at a given release, and a new
`DiffResult.incomparable` for a cell both sides measured under two
different rules. It closes v1.9's item 2 below, "the diff blind spot"
(struck through in place, per this file's own convention) — but only
for the one break the registry lists today. **Item 113's broader
observation, that `diff` has no general version-aware machinery, stays
open**: this wave gives `diff` one named exception, not a general
capability.

## Deferred, by area

### Diff

1. **`SEMANTIC_BREAKS` is typed and documented as if it named any cell,
   but only `_diff_verdicts` consults it.** `_straddles` is called from
   exactly one site (the `if _straddles(cell, old.get("probe_version"),
   new.get("probe_version")):` line in `_diff_verdicts` — cited by
   source, not by number, per this section's process lesson 2; it read
   `diff.py:453` until the fix wave's prose edits moved it to `:535`,
   making the same mistake the lesson records, one wave later), inside
   the generic `verdicts`-dict
   walk); the four raw-measurement family comparisons —
   `_diff_ceiling`, `_diff_shapes`, `_diff_codecs`, `_diff_speed` — have
   no equivalent check. The registry's type
   (`dict[str, tuple[int, ...]]`, keyed by `f"{family}.{cell}"`) and its
   own docstring name no family restriction, so an entry keyed
   `"ceiling.max_verified"` or `"codec.search_replace.small"` would
   type-check, sit in the table, and be silently ignored — never
   consulted, never causing a straddle, never producing `incomparable`
   — because the family it names has no call site that reads the
   registry. Not a defect against v1.10's own scope: the only
   redefinition on record is `verdict.parallel`, a verdicts-family cell,
   so the registry does everything it was built to do today. It is a
   trap for whoever adds the SECOND entry, if that entry ever names a
   cell outside `verdicts` — recorded now so that implementer checks the
   wiring rather than trusting the type signature as the contract.
   Closing it fully is item 113's territory, sharpened: general
   version-awareness would need a straddle check at every family's
   comparison site, not one.

   **This item compounds with item 2 below, and the two were filed
   separately by mistake.** Item 2 enumerates five unregistered
   semantic breaks; the largest of them (v1.3's fixture-set change)
   redefines the entire `codec.*` family — which is precisely the set
   of rows this item says the registry can hold in its type and will
   silently ignore. So the rows most in need of backfilling are the
   rows that would not work if backfilled. Neither gap can be closed
   without the other: registering `codec.*` requires the wiring this
   item names, and doing the wiring without the rows changes nothing.

2. **The registry's founding premise is false: at least five earlier
   releases redefined an existing cell and are NOT registered.** (C1,
   final whole-branch fix wave, 2026-08-19.) The design rests on the
   claim that every schema bump before v1.9 was **additive** — a new
   cell the old side lacked, which `dropped` and exit 3 already
   handle — "which is why one row covers eleven releases". This
   repository documents its own counter-evidence. Verified against the
   sources named:

   - **v1.1 (0.2.0)**, commit `7367802`: `_small_landing(codecs, codec)`
     gained `lens="applies_and_parses"`, repointing
     `verdict.patch_editing` at a different lens. The v0.2 CHANGELOG
     entry records the two lenses reading the same model at 0% and
     100% — the finding that motivated the release.
   - **v1.3 (0.4.0)**, commit `484bc4c`: one fixture per codec cell
     became 5 heterogeneous tasks across 5 defect classes, plus changed
     refusal classification. Redefines **every `codec.*` cell** and the
     codec-backed verdicts. Committed profiles sit on both sides of it.
   - **v1.5 (0.6.0)**: fixed n=5 with provisional marking became
     sequential testing. The CHANGELOG says it outright, and says why
     saying it matters: *"This wave **amends v1.3's verdict semantics**.
     The amendment is recorded rather than quietly applied, because a
     number whose definition changed without saying so is worse than no
     number."* An existing cell, redefined, by a release that knew it.
   - **v1.6 (0.7.0)**: `scripted-loop-v1` became `v2`, `n_turns` 3 → 5
     into the shared denominator, plus a new recovery demotion —
     `verdict.loop_discipline`.
   - **v1.7 (0.9.0)**: the tools pool grew 5 → 20 with a look schedule,
     in full mode. The CHANGELOG enumerates a verdict changing for an
     unchanged endpoint.

   The repository already knew.
   `docs/superpowers/evidence/tier-enthusiast-2026-08/diffs/README.md`
   states that `diff` *"deliberately does not gate on fixture-set name,
   so it will subtract across the change without saying so"*, and its
   finding 5 concludes *"None of these five is evidence that a model got
   worse."*

   **Reproduced at HEAD on real committed evidence**, `tier-enthusiast/
   qwen2.5-coder-7b-q8-quick.json` against `tier-enthusiast-2026-08/
   qwen2.5-coder-7b-instruct-q8_0.json`:

   ```
   old probe_version 0.3.0   new probe_version 0.9.0
   ceiling.max_verified: 16384 -> 32768 (improvement, rung-change)
   verdict.patch_editing: ready -> risky (regression, flip)
   codec.search_replace.tiny.lands_applies:   0.0 -> 0.9428571428571428 (improvement, disjoint-intervals)
   codec.search_replace.medium.lands_applies: 0.0 -> 0.6571428571428571 (improvement, disjoint-intervals)
   codec.whole_file.tiny.lands:               0.0 -> 0.6571428571428571 (improvement, disjoint-intervals)
   incomparable: ()
   dropped: 10 cells
   ```

   A 0 → 0.94 "improvement" on the instrument's strongest evidence
   class, published as an endpoint fact, with `incomparable` empty. The
   `ceiling.max_verified` line is the same class of artefact and is
   already recorded as finding 1 of that diffs README (the CAP moved
   from 16384 to 32768; neither run found a ceiling).

   **Why the rows were NOT backfilled by this wave.** Two of the
   decisions sit above the registry's type and belong to the project
   owner, not to a fix wave:

   - Several breaks are **mode-conditional**. v1.7 grew the tools pool
     5 → 20 *in full mode* and kept `--quick`'s pool verbatim, so
     `verdict.tool_calling` straddles for one pair of documents and not
     for another pair carrying the same two versions. A registry keyed
     `cell -> version` structurally cannot express that; expressing it
     needs a predicate over the profile, which is a design wave.
   - The `codec.*` rows **cannot be consulted at all**. Only
     `_diff_verdicts` reads the registry (item 1 above), so an entry
     keyed `codec.search_replace.tiny.lands` would type-check, sit in
     the table, and never be read. Registering it would create the
     appearance of a guard that does not exist — worse than the honest
     gap, by this project's own standard.

   **Warning for whoever backfills**, because this will look like a
   logic problem and is a fixture problem: `tests/test_cli.py`'s
   `_diff_payload()` emits **no** `probe_version`, so both sides parse
   to `None` and straddle every *registered* break. Registering
   `verdict.structured_extraction` or `verdict.tool_calling` will break
   `test_cli_diff_exit_code_table`, which uses those cells with that
   helper.

   **Spec erratum, same item.** The v1.10 design spec
   (`docs/superpowers/specs/2026-08-18-assay-v1.10-semantic-breaks-design.md`)
   asserts the false premise in three places, and per this project's
   convention a committed design document is not rewritten after the
   fact — this entry is the correction sitting beside it:

   - **§0 Scope and non-goals**: *"**Backfilling breaks for earlier
     versions** — every bump before v1.9 was additive; §1 says why that
     means the table starts with one row."*
   - **§1 The defect**: *"Until v1.9 that was harmless, because every
     schema bump had been **additive**: a new release measured cells the
     old one did not, so the old side simply lacked them…"* and *"v1.9
     is the first release to change **what an existing cell means**
     without changing its name, its type, or its presence."*
   - **§2 The registry**: *"Every bump before v1.9 was additive — a new
     cell the old side lacked, which `dropped` and exit 3 already
     handle — which is why one row covers eleven releases."*

   Corrected statement: v1.9 is the first release for which this project
   *registered* a semantic break, not the first to create one; the
   registry covers the breaks it lists and no others; and at least the
   five releases enumerated above redefined existing cells and remain
   unregistered, so a pair straddling one of them is still scored today
   with no warning. Non-spec surfaces carrying the same claim were
   corrected directly rather than by erratum, following the same split
   this file used for item 113: `src/assay/diff.py`'s registry comment,
   `CHANGELOG.md`'s v0.12 entry, `README.md` (both passages), and the
   v1.9 item 2 paragraph in this file. The claim also appears inside
   v1.9 item 2's **struck-through** original text above and in the
   committed plan
   (`docs/superpowers/plans/2026-08-18-assay-v1.10-semantic-breaks.md`,
   at `:105`, `:141`-`:145`, `:526`, `:531`, `:545`, `:547`); both are
   left unedited as process record, exactly as item 113 left the v1.8
   plan's repeated `by construction` phrasing.

   **How it was found**: by a whole-branch review reading the CHANGELOG
   for what earlier waves said about their own changes, rather than
   grepping for the word "additive" — the v1.5 entry states the
   counter-evidence in the project's own voice, one file away from the
   claim it falsifies.

3. **The v1.8 `dropped` invariant is not exactly true, and never
   was.** (Recorded, not fixed, by the final fix wave, 2026-08-19.)
   The invariant, stated as an equivalence in the v1.8 design spec
   (`docs/superpowers/specs/2026-08-17-assay-v1.8-gate-and-floors-design.md`:
   *"**dropped ⇔ the cell was measured on exactly one side.**"*) and
   re-quoted by the v1.10 spec §4 as *"`dropped` is non-empty **iff**
   some cell was measured on exactly one side"*, has one breach:
   `verdict.<name>.provisional`.

   ```
   old: verdict.parallel = {verdict: ready, provisional: True}   probe 0.10.0
   new: verdict.parallel = {verdict: ready}                      probe 0.11.0
     -> dropped=()  incomparable=('verdict.parallel',)  within_noise=()  changes=[]
   ```

   `verdict.parallel.provisional` was measured on exactly one side and
   appears in **nothing** — not `dropped`, not `incomparable`, not
   `changes`, not `within_noise`.

   **Verified pre-existing, and this wave did not create it.** At the
   branch point `9dd9f5d`, `_diff_verdicts`' `if scored.changes:
   continue` already swallowed a one-sided provisional whenever the
   verdict itself also moved. Measured there directly, with no v1.10
   code present:

   ```
   old: verdict.parallel = {verdict: risky, provisional: True}
   new: verdict.parallel = {verdict: ready}
     -> dropped=()  changes=[('parallel', 'risky', 'ready')]
   ```

   v1.10 adds a third path into the same swallow — the straddle
   `continue` — but the sub-cell was already unreachable through a
   moving verdict. **No exit-code consequence in the v1.10 case**: the
   straddle fires exit 3 anyway. In the pre-existing case the verdict's
   own change fires exit 1.

   **Not fixed here**, deliberately: the fix is a change to
   `_diff_verdicts`' precedence, and that precedence cost this wave a
   Critical to get right (both-absent → nothing; one-sided → `dropped`;
   both + straddling → `incomparable`; else score). Reopening it to
   chase a sub-cell with no exit-code consequence is not a fix-wave
   change.

   **What is corrected is the WORDING**, because an invariant stated as
   an equivalence and enforced as an implication is exactly the kind of
   claim this project does not make. Both statements above are in
   committed design specs and are not rewritten; this entry is the
   erratum for both. Corrected statement: *`dropped` names every
   top-level cell measured on exactly one side; the
   `verdict.<name>.provisional` sub-cell is reported only when the
   verdict it rides on did not itself move, so a one-sided provisional
   accompanying a moved or unscored verdict appears in no field at
   all.* `src/assay/diff.py`'s module docstring, which is live source
   prose rather than a design record, is corrected directly.

4. **The version machinery's tests sit at the wrong altitude, in both
   directions at once.** (Recorded, not fixed, by the final fix wave,
   2026-08-19.) Five of the eight tests added for the registry import
   `_straddles` or `_parse_version` directly. `_straddles` has exactly
   one call site, so a refactor that inlines it breaks all five with
   `ImportError` — failing for the wrong reason, loudly, while the
   instrument's behaviour is unchanged. Simultaneously, the two
   mutations that actually matter went almost entirely unseen. Measured
   by the fix wave against the branch as it stood:

   | mutation | before | after |
   |---|---|---|
   | `_straddles` compares versions lexically (`_parse_version` left intact) | **0 failures** of 1037 | 1 |
   | `_straddles`' unparseable branch returns `False` (comparable-by-default, the direction §6 forbids) | 1 failure, a private-helper unit test | 7, six behavioural |

   Brittle to refactor and blind to the mutations that matter is an
   unusual combination, and it has one cause: the tests assert against
   the helpers instead of against `diff_profiles`.

   The fix wave added the behavioural floor —
   `test_the_0_9_0_baseline_every_committed_profile_carries_is_incomparable`
   and `test_an_unidentifiable_instrument_is_incomparable_end_to_end`,
   both driven through `diff_profiles` — so the blindness is closed.
   The five helper-level tests were left in place: they are fast,
   readable, and pin the parser's contract directly, and deleting
   passing tests to satisfy a stylistic preference is not a fix-wave
   change. The recorded debt is the imbalance itself, for whoever next
   touches `_straddles`: if you inline it, the five ImportErrors are
   expected and the two behavioural tests are the ones that must stay
   green.

## Process lessons

1. **A test that pins two fields using two different cells never
   exercises one cell being both.** Task 2's
   `test_dropped_and_incomparable_stay_distinct` used a `patch_editing`
   fixture to pin `dropped` and a `parallel` fixture to pin
   `incomparable`, and passed green over a real Critical defect: the
   straddle guard ran BEFORE the measured/unmeasured determination, so a
   cell measured on exactly ONE side that also straddled a registered
   break landed in `incomparable` instead of `dropped`. The review that
   caught it was asked to check the population PATHS rather than the
   tests themselves. The lesson generalizes past this one bug: when two
   output fields are meant to be mutually exclusive, the test that
   matters is the one where a SINGLE input could land in either — not
   two separate tests that each pin one field with fixtures the other
   field never touches.
2. **Do not assert a test helper's availability from memory of a
   sibling test file.** Three of this wave's four task briefs contained
   a factual error by the controller about the file the task actually
   edits: a snippet that reassigned `_diff_verdicts`'s own loop variable
   (the `name` bound by its `for name in sorted(...)` line, which the
   loop body already builds cell names from — overwriting it would have
   corrupted every cell name for the rest of that iteration); a claim
   that `make_profile`
   (`tests/test_diff.py`) needed a new default added for `probe_version`
   when the payload already carried one (`"0.5.0"`, hardcoded) and only
   needed to be made overridable; and a claim that `_ceiling` "already
   exists in `tests/test_diff.py`" when, at the time the brief was
   checked, it existed only in `tests/test_cli.py`. A fourth stated the
   wrong REASON an existing test survives the change: it survives
   because its cells are unregistered in `SEMANTIC_BREAKS`, not because
   its fixtures share a `probe_version` — they carry none at all, which
   straddles by design (an unparseable-or-absent version straddles
   every break). None of the four was caught by trusting the brief;
   each was caught by opening the actual file the task edits and
   checking the claim against it directly.

   **This entry made the same mistake in its own first draft**, which is
   why it now cites the loop by its `for` line rather than by number: it
   originally read `diff.py:349`, a line number that was accurate when
   the Task 2 brief was written and stale by the time this record was —
   Task 1 had inserted the registry and parser ahead of
   `_diff_verdicts`, moving the loop to `:428`, where `:349` now lands
   inside `_diff_ceiling`. The task review caught it. The compounded
   lesson: a line number is a claim with a shelf life, and in a wave
   that inserts code above the thing it cites, that shelf life is one
   commit. Cite a symbol or a distinctive line of source, which moves
   with the code, rather than a number, which does not.

---

# Carried debt — v1.9 (recorded 2026-08-18, Task 4; extended 2026-08-18
by the final whole-branch fix wave)

This wave's full close-out ledger is not written here yet — that is a
separate, deliberate record following the v1.8/v1.7 convention below.
Item 1 was filed at Task 4's close, ahead of that record, because it
surfaced while closing Task 4 and should not wait for a ledger that had
not been written. Items 2 onward were filed by the final whole-branch
review that precedes merge — one Critical (item 2), two Important
erratum entries (items 2-3 cover both), and four items recorded but
deliberately not fixed (items 4-7, the Minor "park" list) — following
the same not-yet-written-ledger convention.

## Deferred, by area

### Parallel

1. **`probe()` threads no clock or concurrency-runner seam into the
   parallel family.** `probe()` injects `_clock` only into the
   `BudgetMeter` (`run.py:526-530`) — a seam that exists so a
   wall-clock budget ceiling can be tested without a sleeping suite.
   `run.py:682` then constructs `probe_parallel(active, meter,
   baseline_decode_tps=...)` with neither `clock` nor `runner`
   forwarded, even though `probe_parallel` accepts both (`parallel.py`)
   and the family's own unit tests (`tests/test_parallel.py`) use
   exactly those seams to stay clock-free. The gap is real, not
   theoretical: `tests/test_run.py`'s
   `test_full_mode_parallel_verdict_reads_ready_once_overlap_is_scale_free`
   is the only test that pins the full `run.py` -> `compute_verdicts`
   -> `verdict.parallel` chain, and it has no way to script the parallel
   family's lane spans through `probe()`'s public entry point. It works
   around the gap by pacing its in-process fake's calls in real time
   (a local `_RealisticallyPacedBackend`, `tests/test_run.py`) instead
   of injecting synthetic spans — the one test in the suite that
   depends on real time, and it does so because this seam does not
   reach it, not by choice. What would close it: threading `clock`
   and/or `runner` from `probe()` down through to `probe_parallel`, the
   same way `_clock` already reaches `BudgetMeter`. Not done in v1.9,
   deliberately — v1.9's scope was the classification rule
   (`classify_mode`'s fraction, schema v9 → v10), and adding production
   surface to `run.py` to make one test deterministic was judged
   disproportionate to what that test needed. **(v1.9, Task 4)**

   **The empirical fact that motivated the workaround, recorded because
   nobody had written it down:** the house fake's calls return in
   **0.77 microseconds (median)** — faster than this interpreter's own
   `threading.Thread.start()` takes to hand control to a new OS thread
   (measured at **12-71 microseconds** between consecutive lanes' start
   times, this box). *Correction (M5, final fix wave, 2026-08-18): this
   entry originally stated "roughly 10 microseconds" and "roughly
   70-100 microseconds." Independently re-measured — 2000 scripted
   calls for the per-call figure, 300 four-lane trials (900 gaps) for
   the start-time spread — at 0.77 µs median call and a 12-71 µs gap
   range. The conclusion these numbers support is unchanged and holds
   with MORE margin than originally recorded (the fake is faster, and
   the thread-start floor is lower, than first stated); only the
   numbers, which were presented as measurements, needed correcting.*
   Below that floor, `_threaded_runner`'s lanes
   never share the CPU at all: the first thread runs to completion
   before the second is even created, so an unmodified fake's lanes
   produce wall-clock spans that are fully disjoint — genuinely zero
   overlap, not merely under some threshold. No `classify_mode`
   tolerance, the old absolute 0.25 s or the new 0.25-of-span fraction,
   can read two spans that never touch as anything but `serialized`.
   This is why the house fake could never exercise the `parallel`
   branch end to end, under either rule — a fixture-speed ceiling
   orthogonal to which classification rule this project ships, and
   invisible until something tried to pin the fake's rung against a
   real `probe_parallel` call rather than synthetic spans. **(v1.9,
   Task 4)**

### Diff

2. ~~**The diff blind spot: a rule-change flip reads as a silent,
   `--gate`-passing endpoint improvement.** (C1, final fix wave,
   2026-08-18.) The v1.9 spec §3 and this file's own CHANGELOG entry
   originally justified the `tolerance_s` → `overlap_fraction` rename
   with a claim that turned out to be false: that `assay diff` would
   compare `tolerance_s: 0.25` against `tolerance_s: 0.25`, find them
   byte-equal, and report no change across the rule break. Checked
   directly against `src/assay/diff.py`: the module has **zero**
   references to `parallel` as a family name. It never reads
   `tolerance_s`, `overlap_fraction`, or any other field of the
   `parallel` payload — it compares exactly five families (`ceiling`,
   `ceiling_shapes`, `verdicts`, `codecs`, `speed`), and `parallel`
   only enters the comparison indirectly, as one more name inside the
   generic `verdicts` dict `_diff_verdicts` walks.~~

   ~~The real failure is worse than a silently-skipped field, because
   the verdict IS compared — and reads as a genuine finding. Reproduced
   directly against `diff_profiles`/`_diff_exit_code` (same model, same
   hardware tier, only `verdict.parallel` differing — the shape a
   0.10.0-baseline-vs-0.11.0-rerun comparison on a fast endpoint would
   take once the classification rule alone flips a k's `mode` from
   `serialized` to `parallel`):~~

   ```
   comparable: True   identity notes: ()
   verdict.parallel: risky -> ready (improvement, flip)
   dropped: ()
   exit plain: 1      exit --gate: 0
   ```

   ~~`_ladder_direction` (`diff.py`) ranks `ready` above `risky`, so the
   flip scores `improvement`, and `_diff_exit_code` (`cli.py`) only
   fails `--gate` on a `regression` — an improvement passes silently.
   Exit 3 (`dropped`) cannot catch it either: nothing is dropped, both
   sides measured the `parallel` cell, `dropped` means "measured on
   exactly one side" and this is measured on both. So an instrument
   rule change — nothing about the endpoint moved — publishes as a
   verified capability improvement, and a `--gate`-based CI would wave
   it through.~~

   ~~This is NOT the same gap CARRIED-DEBT item 113 already named
   ("`diff` has no version-aware machinery at all"). Item 113 concerns
   whether a cross-schema pair reads exit 3 (it does, IF the newer
   schema measured something new the older one lacks — not "by
   construction"). Every schema bump before v1.9 was purely ADDITIVE:
   a new field, a new family, a new lens — so a pair spanning one of
   those bumps either drops a genuinely-new cell (caught by exit 3,
   item 113's territory) or compares two cells that mean the same thing
   on both sides. **v1.9 is the first bump that changes the MEANING of
   an already-measured cell without changing its name, type, or
   presence.** `verdict.parallel` exists on both sides, is a string on
   both sides, and is measured on both sides — nothing about its shape
   signals that the rule which produced it changed. Item 113's
   diagnosis (exit 3 depends on what was measured, not on schema
   difference alone) is exactly why exit 3 cannot see this: nothing
   was dropped, so by item 113's own accurate account exit 3 correctly
   does not fire — the gap this item names is orthogonal, not a variant.~~

   ~~Also unlike the diff family cells (`ceiling`, `codecs`, `speed`),
   which compare raw measurements diff itself judges by evidence
   strength, `verdict.parallel` is a STRING already reduced by
   `_parallel_verdict`'s own ladder logic before `diff` ever sees it —
   diff has no way to distinguish "the endpoint changed" from "the rule
   that reads the endpoint changed" from a verdict string alone, for
   any family, not just this one. `verdict.parallel` is simply the
   family where this wave made that distinction matter for the first
   time.~~

   ~~The identity gate (`identity_gate`, `diff.py`) does not help either:
   it checks `model.name`, `model.quant`, `model.weights_bytes`,
   `provenance.tier`, and `provenance.emulated` — five fields about
   the HARDWARE and MODEL under test, never `probe_version` or
   `assay_profile_version`. A pair that differs only in which assay
   build measured it passes the identity gate cleanly, by design (the
   gate's whole job is refusing to compare different weights, not
   different instrument versions).~~

   ~~**Not fixed here, deliberately.** Giving `diff` version-awareness is
   a change to its whole contract — CARRIED-DEBT item 113's territory,
   already open, now sharpened by a second concrete instance instead of
   item 113's hypothetical account. The mitigation available today is
   external: a consumer that prechecks `probe_version` +
   `schema_version` before trusting a diff at all — bloomery's drift
   watch does exactly this — is safe from this specific trap, which is
   the real (and now correctly stated) reason the v1.9 rename and
   schema bump matter. A consumer relying on `assay diff --gate` alone,
   with no version precheck of its own, is not protected by anything
   this wave shipped.~~

   ~~**Spec erratum, same item:** the v1.9 design spec's §3
   (`docs/superpowers/specs/2026-08-18-assay-v1.9-scale-free-overlap-design.md`)
   states the false byte-equality justification quoted above. Per this
   project's convention the committed spec is not rewritten after the
   fact; this entry is the correction sitting beside it. CHANGELOG.md's
   v1.9 entry made the same claim and, being in-branch and unpublished,
   is corrected directly rather than by erratum.~~

   **CLOSED in v1.10.** `SEMANTIC_BREAKS` (`diff.py`) now records
   `verdict.parallel`'s v1.9 rule change; a pair straddling it lands in
   the new `DiffResult.incomparable`, never scored, and exit 3 fires on
   either `dropped` or `incomparable`. The reproduction above now reads
   `incomparable: (verdict.parallel,)`, exit 3 under both plain and
   `--gate`, verified end to end by
   `test_an_instrument_rule_change_is_not_published_as_an_improvement`
   (`tests/test_cli.py`).

   **What did NOT change, so this closure is not over-read.**
   `identity_gate` still ignores `probe_version` and
   `assay_profile_version` entirely, by design — a version difference is
   not a different endpoint. `diff` still has no GENERAL
   version-awareness: **item 113's broader observation stays open.**
   This wave closes the one named break the registry lists, not the
   class of the problem; a future release that redefines a different
   cell's meaning without adding a `SEMANTIC_BREAKS` entry for it
   reintroduces exactly this defect, unflagged, for that cell.

   **The v1.10 slice's own record.** `SEMANTIC_BREAKS` grows by one row
   per release that redefines an EXISTING cell's meaning, not per
   release generally. **This paragraph originally continued "every bump
   before v1.9 was additive … which is why one row covers eleven
   releases". That was false**, and is corrected by the v1.10 section's
   Diff item 2 above, which enumerates five earlier releases that
   redefined existing cells and are not registered. The table is small
   because it is INCOMPLETE, not because the history is: any release
   that changes what an already-measured cell means needs its own
   entry, and five that already did are still missing theirs.

3. **Spec §2's "strictly dominates ... changes none it got right" is
   an overclaim, corrected.** (I5, final fix wave, 2026-08-18.)
   Measured counter-examples: `classify_mode([(0, 10), (9.7, 19.7)])`
   and `classify_mode([(0, 10), (9, 19)])` both read `parallel` under
   the retired absolute rule and `serialized` under the new fraction
   rule — a FLIP, not a preserved answer, on a pair the old rule got
   "right" only in the sense that it was internally consistent with
   itself. In both cases the new answer is the better one: 0.3 s and
   1 s of overlap on a 10 s span is 3% and 10%, respectively — nowhere
   near genuinely concurrent — and the old rule called both `parallel`
   purely because 0.3 s and 1 s both clear the flat 0.25 s bar. The
   rule is fine; the domination claim is false. Corrected statement,
   also applied to CHANGELOG.md directly (in-branch, unpublished): the
   new rule corrects every SHORT-lane case the old one got wrong (the
   family this wave was written to fix) AND reclassifies long-lane
   pairs whose overlap is a small fraction of their span (a case the
   old rule was also getting wrong, from the opposite side) — which is
   the intended consequence of making the test relative, not a
   narrower "changes nothing else" guarantee. Per this project's
   convention the spec (§2) is not rewritten; this is the erratum.

### Parked (recorded, not fixed, by the final fix wave, 2026-08-18)

4. **`tests/test_profile.py:1036`'s
   `test_schema_v10_and_the_package_version_move_together` duplicates
   `test_schema_version_and_package_version_move_together`** (line 971
   of the same file), which already asserts `PROFILE_VERSION == 10`,
   `assay.__version__ == "0.11.0"`, the `pyproject.toml` version string,
   AND the README's `assay_profile_version` line — strictly more than
   the duplicate covers. Not removed here: a redundant PASSING test is
   a lower-priority cleanup than the Critical/Important findings this
   wave exists to fix, and removing test coverage is exactly the kind
   of change that deserves its own reviewed diff rather than riding
   along in a fix wave already touching this file in a dozen places.
   **(M3)**

   **CLOSED in v1.10** by the final whole-branch fix wave. Folded into
   `test_schema_version_and_package_version_move_together`, which
   already asserted both of its lines plus the `pyproject.toml` string
   and the README's `assay_profile_version` line, so no coverage was
   lost. Two things changed since this item was parked. First, the
   objection above — that removing a test "deserves its own reviewed
   diff rather than riding along" — was answered by giving it exactly
   that: one commit, that change only. Second, it stopped being a
   merely-redundant passing test: the v1.10 package bump left the
   schema at v10 while `__version__` went 0.11.0 → 0.12.0, so the name
   `..._move_together` came to assert the opposite of what the release
   established. A test name is prose, and a false one is a defect here.
   (Note the line numbers above are stale, as this file's own process
   lesson predicts: the test sat at `:1085` by the time it was removed,
   and its sibling at `:1015`.)
5. **`tests/test_run.py:797` gates the paced fake on
   `seed >= PARALLEL_SEED_BASE`, an open upper bound.** Airtight today —
   every other family's seed base is ≤ 1520 against `PARALLEL_SEED_BASE
   = 1700` — but nothing enforces that relationship going forward: a
   future family seeded at 1700 or above would silently start inheriting
   the paced fake's per-call sleep (real time, deliberately, per this
   file's item 1 above), which would slow that family's tests without
   any signal pointing at why. Not fixed here: the correct fix is
   probably a shared, explicitly-ordered seed registry across all
   families, which is a larger and separately-reviewable change than
   this wave's scope. **(M4)**
6. **Commit `692e0eb` deliberately leaves the suite red until
   `701c521`**, so `git bisect` run across that range can land on a
   commit with failing tests. Not squashed or reordered: this project's
   documents (specs, plans, this file) are amended in place rather than
   silently rewritten after the fact, and the same principle applies
   here — the red window is a true record of TDD's RED step, not an
   accident, and rewriting history to look green throughout would be a
   worse kind of dishonesty than a bisect landing on a known-red commit.
   Recorded so a future bisect run is not surprised by it. **(M6)**
7. **`make_parallel` (`tests/test_profile.py`) is pinned v9-shaped** —
   `tolerance_s`/`tolerance_provenance` real, `overlap_fraction`/
   `overlap_provenance` left at their `None` default — so most
   report/profile tests that build a profile through this fixture
   exercise the pre-v10 render branch by default. The v10 branch has a
   couple of narrow dedicated tests that pass explicit `overlap_*`
   overrides (`test_the_report_renders_a_v10_overlap_fraction` in
   `tests/test_report.py`; `test_a_v10_profile_does_not_emit_the_
   retired_seconds_tolerance` in `tests/test_profile.py`, added by this
   same fix wave for I6) and no DEFAULT-fixture coverage — a future
   change to the general profile/report test suite is still far more
   likely to exercise the v9 path than the v10 one purely by which
   fixture it inherits. Not changed here: `make_parallel`'s own
   docstring explains this is deliberate, because it is shared with
   `tests/test_report.py`'s byte-pinned matrix-page rendering tests,
   which read the literal "0.25" / "chosen-2026-08-17" seconds-tolerance
   text — flipping the fixture's default shape would be a much larger,
   separately-reviewable change touching the byte-pinned page output.
   **(M8)**

# Carried debt — v1.8 (recorded 2026-08-17 at the wave's close)

Known gaps deliberately carried out of v1.8, with the rulings that
carried them. None blocks the release.

**This file is self-contained.** Every deferred item and every ruling
recorded during this wave is written out below; nothing is delegated
to a pointer. The raw process record it was derived from is committed
verbatim beside the plan at
[`superpowers/plans/2026-08-17-assay-v1.8-gate-and-floors-ledger.md`](superpowers/plans/2026-08-17-assay-v1.8-gate-and-floors-ledger.md)
— that is provenance for the derivation, not a place detail hides. The
plan and the binding spec it implements are committed beside it too:
[`plans/2026-08-17-assay-v1.8-gate-and-floors.md`](superpowers/plans/2026-08-17-assay-v1.8-gate-and-floors.md)
and
[`specs/2026-08-17-assay-v1.8-gate-and-floors-design.md`](superpowers/specs/2026-08-17-assay-v1.8-gate-and-floors-design.md).

**Reading the entries.** An item ~~struck through~~ was **closed**
during the wave, and the text that closed it follows in bold. An item
left plain is **open**. Nothing is deleted: an item that was raised
and closed is a different fact from one that was never raised, and a
reader must be able to tell them apart. Item numbers are a single
namespace shared with the v1.7 section below — several source files
now cite them directly (e.g. `CARRIED-DEBT item 18`, `item 29`, `item
74`, `item 101` in code comments and test docstrings) — so a v1.7 item
closed or updated by v1.8 work is edited in place down there rather
than renumbered, and new items pick up where v1.7 left off, at 107.
The `(T<n>)` marks the task that raised it.

---

## What this slice settled

v1.8 shipped two fixes to claims the instrument was making that it
should not have been, and closed the opener v1.7 deliberately left
open. **`diff` gained a fourth exit, `3` ("incomplete")**, firing
whenever any cell was measured on exactly one side, in both plain and
`--gate` mode, and outranking exit `1` at precedence `2 > 3 > 1 > 0` —
a family that vanished between two profiles is not a measured drift,
and exit `1`'s narrower claim (a number moved) could never speak for
it. This came out of the field: bloomery's drift watch exited `0`
under `--gate` on a v8-vs-v4 pair while five measured families went
unmeasured on one side. Underneath it, the **display-layer bug that
would have become an exit-code bug** is fixed — `_diff_verdicts` had
been sending verdicts unmeasured on BOTH sides to `dropped`, so two
byte-equal profiles reported a dropped cell. `dropped` now means
precisely "measured on exactly one side" — the rule the other four
families already kept — which is what makes it safe for the exit code
to read directly. The order the two fixes shipped in was not
incidental, and the wave's own evidence proves it (process lesson 1,
below).

**`verdict.parallel` joined the lens.** v1.7 shipped the family
measurement-only on the ground that a rung invented without a measured
floor is the overclaim the rest of the schema exists to refuse; the
2026-08 campaign's fifteen live profiles removed that ground.
`verdict.parallel` reads the WORST measured k: a refused k or a k with
no ratio reads `unmeasured`; any `serialized` k CAPS the verdict at
`risky`, because a queueing endpoint's per-lane rate looks fine and
only the scheduling fact catches it; otherwise it ladders on
`degradation_ratio` at 0.8 `ready` / 0.5 `risky`. The floors are
**CHOSEN, not derived**, and the lens carries `floor_provenance`
saying so — the `OVERLAP_TOLERANCE_S` idiom. The campaign's thirty
k-readings all sit at 0.995-1.007, far above both floors, so the
cluster sanity-checks the ladder without exercising either boundary
(item 107, below).

Profile schema **v9**, package **0.10.0**, zero new probe calls: the
verdict derives from measurements `parallel` already made, so full and
quick's cost budgets (610/130) are unchanged. The fifteen committed v8
profiles were NOT rescored and the campaign was NOT re-run; the matrix
gained a `parallel` column and the fifteen v8 rows read `unmeasured`
in it, honestly. The campaign's own thirty committed k-readings are
re-laddered by the suite on every run, so a floor that drifts above
the live cluster fails a test rather than a review nobody ran.

Six further hygiene fixes landed alongside the two headline changes:
the campaign wrapper now points at the repo rather than a deleted
v1.7 worktree; the campaign corpus glob is filtered by version key so
a non-profile JSON in that directory is skipped rather than
mis-blamed (item 74, closed); `probe_parallel` refuses a missing
baseline before spending rather than after (item 18, closed); the
three grade-ordering surfaces (`profile`, `report`, `diff`) are now
pinned to agree by test, not coincidence (item 8, closed); the token
meter's honest partial — a family admitted on calls that still dies
mid-run on tokens — has a test (item 29, closed); and `CallRecorder`'s
concurrent-writer case has one too, with an honest finding about what
it does and does not prove (item 101, partially closed — see below).

Twelve tasks, fourteen commits, 986 → 1009 tests.

---

## Standing rulings (do not re-litigate without a recorded amendment)

Every ruling made during the wave, in the order made.

1. **A worktree-durability assertion replaces a checkout-equality one.**
   Task 1's brief asserted `Path(REPO).resolve() == _REPO_ROOT`, where
   `_REPO_ROOT` derives from the test file's own location — under SDD
   that location is the worktree, so the assertion failed there and
   would only pass post-merge. The property actually wanted is "the
   script points at a durable repo root, not an ephemeral worktree,"
   which the equality check does not express. Replacement: `REPO` is
   not under `.worktrees`, exists, is a directory, and contains
   `src/assay/__init__.py`. Cost if wrong: a `REPO` pointing at some
   *other* valid assay checkout would pass, but the worktree
   regression this exists to catch would still fail it. **(T1)**
2. **`REPO` itself stays the canonical `/home/brice/workspace/assay`**,
   unchanged by which checkout happens to be editing the script — that
   is the correct value for a durable script regardless of where the
   edit is made. **(T1)**
3. **The two exit-3 fixture updates and the two schema-8 pins were
   pre-authorized and named in advance**, rather than left for the
   implementer to discover under a rule that forbids loosening the new
   invariant. Removes the only ambiguity a "some tests must legitimately
   fail" mandate creates. **(T3/T5)**
4. **The Task 8 cost-verification step is advisory, not a gate.** The
   brief's `MODES` symbol does not exist (`run.py` defines
   `MODE_PARAMS`); the implementer was told to read the real
   mode-params mapping and complete the comparison rather than treat
   the mismatch as a block, because the step exists to falsify the
   CHANGELOG's "no family costs a call more" claim and any correct
   route to that comparison serves. **(T8)**
5. **A reviewer's "weak second test" minor was upgraded to a fix-loop
   finding**, against the skill's default that minors never re-enter
   the loop. `test_verdicts_keyed_but_unmeasurable_on_both_sides_are_
   not_dropped`, as originally written, deleted the verdict key from
   both sides — the comparison loop never reached the guard it claimed
   to cover, so it passed against the UNFIXED code. The replacement
   uses a key BOTH sides carry with `None` inside it, which does reach
   the guard. This is process lesson 2's second instance. **(T2)**
6. **998 is the correct post-Task-4 test count, not 999.** The plan's
   arithmetic counted two test helpers (`_prow`, `_parallel`) as tests
   alongside the five real ones. The implementer flagged the
   discrepancy instead of inventing a sixth test to hit the predicted
   number. **(T4)**
7. **"Ninety lanes (15 models × k in {2, 4})" is arithmetically wrong**
   — the parenthetical computes to 30, not 90 — and the correct
   phrasing everywhere is *thirty k-readings comprising ninety lanes*
   (k=2 contributes 2 lanes, k=4 contributes 4, so 6 lanes/model × 15
   models). This is a **regression of CARRIED-DEBT item 84**, which
   this project already closed once. It shipped from the controller's
   own spec and Task 8 draft into `src/assay/profile.py:72`, into the
   design spec (item 110, below — recorded as an erratum, not
   rewritten, per this project's convention for committed design
   docs), and into `docs/CARRIED-DEBT.md`'s own item-15 closing text —
   the very file item 84's closure credits with getting this right.
   Closed a second time by a repo-wide grep sweep with a per-hit
   verdict at four sites, so a fifth instance could not hide behind a
   fix that only touched the site a reviewer happened to find. Process
   lesson 4 covers the mechanism. **(T7/T8)**
8. **Task 7's first dispatch attempt failed by editing the wrong repo.**
   A cheaper-tier implementer edited `/home/brice/workspace/assay`
   (the MAIN repo, on `master`) instead of the worktree, then reported
   BLOCKED citing "HEAD is 91adfbe" — master's HEAD, not the
   worktree's. Inspection found both new tests defined TWICE in the
   stray edit (the second pair would have silently shadowed the
   first). The stray edit was discarded with `git checkout --` in the
   main repo and master verified clean at 91adfbe; nothing of value
   was lost, since the canonical test code lived in the brief.
   Re-dispatched on a stronger model with an explicit cd-and-verify
   preamble — the 128-tool-call, no-commit failure mode is what the
   model-selection guidance warns cheap tiers do on multi-step work.
   **(T7)**
9. **The baseline-guard test's brief fixture was wrong; the corrected
   one is what shipped.** `PinnedSpans([])` trips the fake's own
   length assertion before the probe ever reaches the baseline
   arithmetic, and `LaneFake({})` returns lanes with no timings, so
   `per_lane is None` short-circuits the `baseline_decode_tps > 0`
   check and produces a silent `degradation_ratio=None` rather than
   any error. Item 18's actual defect needs lanes that DO report
   timings; verified live, that fixture reaches `TypeError: '>' not
   supported between instances of 'NoneType' and 'int'` with
   `meter.spent.calls == 2` pre-fix, and `ValueError` with
   `meter.spent.calls == 0` post-fix — the second assertion is the
   whole point, since "before the spend" is the property, not merely
   "raises." **(T9)**
10. **The `CallRecorder` concurrency fixture was wrong twice over in
    the brief.** `tests.fakes.ScriptedBackend` rejects any prompt it
    does not recognize, so it cannot drive 200 synthetic calls;
    `test_replay.py`'s own local `ScriptedBackend` pops from a shared
    list and is not thread-safe, so it would fail a concurrency test
    for the wrong reason. The fixture that shipped is a small,
    deliberately STATELESS fake defined inline in the test — no shared
    mutable state, so any lost or corrupt row is unambiguously the
    recorder's fault. **(T11)**
11. **Tasks 10 and 11 were batched into one dispatch** — both small,
    independent, test-only additions touching disjoint files with no
    shared state, saving a full dispatch-and-review cycle. **(T10/T11)**
12. **The recorder-lock test stays; its docstring stops claiming to
    prove the lock.** The implementer honestly reported the
    concurrency test passing with the lock removed. A controller sweep
    at 100 B / 8 KB / 64 KB payloads, with and without the lock, 8
    threads × 15 rows, confirmed it: 120/120 rows, 0 unparseable, all
    six configurations. `CallRecorder._write_row` opens, writes one
    row, and closes per call; on Linux, `write()` to a regular file is
    atomic per-inode and `O_APPEND` makes the offset update atomic, so
    the lock is genuinely redundant for THIS write pattern on THIS
    platform. Decision: keep the test (it pins a real property — every
    row present and whole under concurrency — and would catch a FUTURE
    write-path change, such as a long-lived handle or a row built
    incrementally); fix the docstring so it does not overclaim; record
    item 101 as only PARTIALLY closed. Same shape as
    `OVERLAP_TOLERANCE_S` / item 16 — a guard kept and flagged, not
    retired on evidence that never exercised it. **(T11)**

---

## Deferred, by area

### Carried forward from v1.7, still open

Restated here in full, as the wave's own record of what remained
untouched or unresolved, per this file's self-containment rule. Full
text and any earlier history live at their original item numbers in
the v1.7 section below; nothing here supersedes that text, and nothing
there was struck through unless closed above. (Presented as labelled
paragraphs rather than a renumbered list, deliberately: these item
numbers are non-sequential cross-references, and a native ordered
list here would auto-renumber them to 1-4 on render and silently break
every citation.)

**Item 1 (pool-to-35) — still priced and not taken.** `ready` remains
unreachable non-provisionally under looks {5, 10, 20}: a perfect 20/20
reads `ready` provisional at Wilson lower 0.8389 against the 0.9
floor, and n = 35 is the smallest n at which a perfect cell clears
`ready` undisputed. Brice deferred it again for v1.8, on the same
terms as v1.7: +30 calls on full (15 tasks × 2 turns) over today's 40,
for a rung of decisiveness the schedule does not otherwise reach.
Untouched by any v1.8 task.

**Item 16 (`OVERLAP_TOLERANCE_S`) — still stands on a sanity check,
not a derivation, and it now blocks TWO things instead of one.** The
endpoint that would retire it — one that actually serializes — still
does not exist: fifteen campaign profiles at both k = 2 and k = 4 show
full `n_lanes_ok`, empty `lane_errors`, empty `skipped`, throughout.
The same missing endpoint now ALSO leaves `verdict.parallel`'s new
`serialized` gate unexercised by live data — a rule with a real branch
and zero rows that have ever taken it. `test_no_campaign_row_ever_
read_serialized` pins that absence as a fact the suite states out
loud, rather than one that quietly erodes as campaigns accumulate and
nobody notices the gate has never fired.

**Item 17 (evidence-class strings re-declared) — now declared THREE
times, not two, and the third copy is v1.8's.**
`profile._parallel_verdict` needed the same weakest-first ranking
`parallel._weakest` already used (against `parallel._EVIDENCE_WEAKEST_
FIRST`), and `profile._PARALLEL_EVIDENCE_WEAKEST_FIRST` was added as a
full re-declaration — confirmed byte-identical to
`parallel._EVIDENCE_WEAKEST_FIRST` during Task 4's review. This is a
regression in the literal sense: the fix (one shared public tuple) has
been available and unscheduled since v1.7's close, and v1.8 added to
the duplication it was already carrying rather than resolving it.

**Item 106 (multi-turn chains) — remains untouched.** Every family
through v1.8 still scores a single request/reply or a two-turn tool
exchange. Whether a model holds a plan across N turns — and where it
stops holding it — is a different measurement with its own cost curve
and failure taxonomy, and it needs its own spec. No v1.8 task touched
this.

### Parallel

107. **The parallel floors are CHOSEN and no live row exercises either
     boundary.** Every one of the campaign's thirty k-readings sits at
     `degradation_ratio` 0.995–1.007 against floors of 0.8 (`ready`)
     and 0.5 (`risky`) — a 10× span of single-lane speed (28 → 288
     tok/s) and every reading clears `ready` by a wide margin. The
     condition for deriving rather than choosing them is the same
     missing serialized endpoint item 16 already names: nothing in the
     live cluster has ever approached either number, so the floors are
     a sanity check against the cluster's shape, not a measurement of
     where degradation actually begins. `floor_provenance =
     "chosen-2026-08-17"` says so at every point of use, matching the
     `OVERLAP_TOLERANCE_S` idiom deliberately. **(T4)**
108. **`probe_parallel`'s `baseline_decode_tps` parameter is typed
     `float`, not `float | None`**, although item 18's fix means a
     `None` is now runtime-guarded (raises `ValueError` before any
     spend) rather than reaching a bare `TypeError`. The type hint
     still promises a value the guard proves callers cannot rely on
     being there. Pre-existing since the parameter's introduction; no
     type-checker is configured anywhere in this repo, so nothing
     currently catches the mismatch mechanically. **(T9)**
109. **`_parallel_verdict`'s mode-cap is applied textually AFTER the
     ratio ladder, while its docstring narrates "mode gates first."**
     The Task 4 reviewer proved behavioural equivalence across all six
     truth-table rows by hand, and by mutation: replacing the
     conditional cap with an unconditional assignment makes the
     `serialized` + ratio-0.30 row fail (`'risky' != 'unusable'`),
     confirming the suite genuinely catches the cap-vs-assign bug
     class regardless of source order. The docstring describes rule
     PRIORITY, which the code preserves; only the prose-vs-code
     ordering is cosmetic. **(T4)**

**Item 112 (the 0.25s tolerance cliff is now load-bearing for a
published rung) — recorded, not fixed, by the final fix wave
(2026-08-17).** `classify_mode` calls a pair `serialized` unless
consecutive lanes overlap by MORE than `OVERLAP_TOLERANCE_S`; for
lanes launched together that means each lane must last longer than
0.25s to be called `parallel`. Measured directly: 0.1s
genuinely-concurrent lanes read `serialized`; 0.3s lanes read
`parallel`. Consequences:

- With `DECODE_MAX_TOKENS = 64`, the fastest live campaign model
  (`qwen2.5-coder-1.5b-instruct-q8_0`, 288 tok/s) computes to a
  **0.222s** pure-decode span against the 0.25s tolerance — it read
  `parallel` only because prefill and HTTP pushed the wall span over.
  A faster endpoint on this tier gets `risky` for the fleet question
  while serving every lane at full rate.
- The house fake takes the `serialized` branch on every full-mode run
  — pinned by `test_full_mode_parallel_verdict_is_produced_and_reads_
  risky_not_ready` (M10, the same fix wave): `ScriptedBackend`'s
  in-process lanes return well under 0.25s, so `degradation_ratio`
  reads ~1.0 while `mode` still reads `serialized`, and the verdict
  caps at `risky`.
- Direction of error is UNDER-claim, which is the safe side — but it
  is still a wrong rung on a public page, driven by a constant this
  file already records as unexercised by live data (item 16). This
  shares item 16's retirement condition exactly: an endpoint that
  actually serializes.

  **v1.9 amendment (2026-08-18): this is the entry the v1.9 wave
  exists to close, and it is amended, not struck — the original text
  above is left standing because it is the accurate account of the
  defect as it stood when this item was raised.** `OVERLAP_TOLERANCE_S`
  (0.25 s, absolute) no longer exists; `classify_mode` now reads
  `OVERLAP_FRACTION` (0.25, dimensionless — 25% of the shorter lane's
  span; see `src/assay/parallel.py`). What changed, verified by the
  v1.9 suite (`tests/test_parallel.py`,
  `test_concurrent_lanes_read_parallel_at_every_time_scale`): the
  0.1 s-lane case this item measured directly now reads `parallel`,
  not `serialized`, and so does every genuinely-concurrent duration
  swept from 0.05 s to 1.0 s — the cliff this item named is gone as a
  function of lane DURATION. The 0.222 s pure-decode span this item
  flagged as sitting right at the old edge is now robustly `parallel`
  rather than surviving only because prefill and HTTP padded it. The
  house fake's test was renamed
  (`test_full_mode_parallel_verdict_is_produced_and_reads_risky_not_
  ready` → `test_full_mode_parallel_verdict_reads_ready_once_overlap_
  is_scale_free`, `tests/test_run.py`) and now asserts `ready`, not
  `risky` — the fake's sub-millisecond lanes read `parallel` under the
  new rule where they read `serialized` under the old one, once paced
  to a duration real OS threads can actually overlap at (see this
  file's v1.9 section, item 1, for the fixture-speed ceiling that
  pacing works around).

  What remains open, and is now item 16's open condition exactly
  rather than a variant of it: no live campaign row has ever read
  `serialized` under either rule, so the boundary itself — the point
  where the ratio test actually decides something — is still
  unexercised by live data. **What is NOT supported by anything
  measured** (corrected by I3, same fix wave, after this item's own
  first v1.9 draft overclaimed it): the evidence does not show that no
  live row overlapped by between 0% and 25% of a span. Profiles store
  only the derived `mode`, never the spans (`ParallelRow` has no span
  field), and the transcripts carry no timing — those overlap
  fractions were never recorded and cannot be recovered from anything
  committed. The narrower, actually-supported claim is: no live row
  ever read `serialized` under the RETIRED absolute-seconds rule.
  Retiring `OVERLAP_FRACTION`'s chosen-not-derived flag still needs an
  endpoint that actually serializes under the new rule, and this tier
  has not produced one — item 16's condition, unchanged by this wave.

### Documentation

110. **The v1.8 design spec's own arithmetic is imprecise: "0.995
     minimum over ninety lanes."** `degradation_ratio` is one value
     per k-reading, and the campaign's minimum (0.99526) is the
     minimum over the thirty k-readings, not the ninety lanes — the
     same conflation item 84 closed once and this wave regressed
     (ruling 7). The spec is a committed design document, and per this
     project's convention it is not silently rewritten after the fact;
     this entry is the dated amendment sitting beside the original
     text instead. Correct figure: 0.995 minimum over **thirty
     k-readings**, comprising ninety lanes. Location:
     `docs/superpowers/specs/2026-08-17-assay-v1.8-gate-and-floors-design.md:165`.
     **(T8, ruling 7 amendment)**

**Item 113 (the design spec's exit-3 claim is stronger than the
instrument can guarantee) — erratum recorded by the final fix wave
(I4, 2026-08-17).** Spec §2 states a cross-schema pair reads exit 3
"by construction." `diff` has no version-aware machinery at all; exit
3 is a consequence of which cells were actually measured. Verified: a
v8-vs-v9 pair where the v9 side measured `parallel` drops
`verdict.parallel` and reads exit 3, as the spec claims — but a v9
pair whose `parallel` is `unmeasured` on the newer side too (quick
mode, or full mode where `speed` went unmeasured so `run.py` drops the
family by name) compares byte-clean and exits **0** even across the
same schema bump. Cross-schema-ness alone does not decide it; whether
the newer schema actually measured something new does. The spec is a
committed design document and per this project's convention is not
silently rewritten after the fact; this entry is the dated amendment
sitting beside the original claim instead. Corrected statement (also
applied to the four operator-facing surfaces — `cli.py`'s docstring,
`README.md`, `CHANGELOG.md`, `tests/test_diff.py`'s comment — which
are not spec text and were rewritten directly): a cross-schema pair
reads exit 3 whenever the newer schema actually measured a cell the
older one lacks, not merely because the schemas differ; the same
correction applies to the budget-mode-vs-full-mode claim beside it.
Location:
`docs/superpowers/specs/2026-08-17-assay-v1.8-gate-and-floors-design.md`
§2 (see also the `by construction` phrase repeated at plan
`docs/superpowers/plans/2026-08-17-assay-v1.8-gate-and-floors.md:1068`
and `:1098`, also left unedited as committed process record). **(I4)**

### Test hygiene

**Item 101 (`CallRecorder`'s write lock), updated — full text at its
original number in the v1.7 section below.** The write lock is
exercised by a concurrency test now
(`test_call_recorder_keeps_every_row_whole_under_concurrent_writers`),
but the test's own finding is that the lock is NOT load-bearing for
today's write pattern: `_write_row` does open-append-one-row-close per
call, and on Linux `write()` to a regular file is atomic per-inode
while `O_APPEND` makes the offset update atomic, so rows cannot
interleave with or without the application-level lock. Confirmed at
100 B / 8 KB / 64 KB payloads × 8 threads, with and without the lock:
120/120 rows, 0 unparseable, in all six configurations. The test and
the lock BOTH stay (ruling 12): the test pins a property worth pinning
(every row present and whole under concurrent writers) and would catch
a FUTURE write-path change — a long-lived handle held across calls, or
a row assembled across several writes — that would make the lock
load-bearing; the lock costs nothing and guards a property a refactor
could silently take away. This is recorded the way item 16 is
recorded: a guard kept and flagged, not retired on the strength of
evidence that never exercised it. **(T11)**

~~111. **`tests/test_campaign_script.py:29` asserts both `exists()` and
     `is_dir()` on the same path**, where `is_dir()` alone already
     implies existence — harmless (the redundant check cannot itself
     produce a false pass), and its only effect is a slightly less
     specific failure message if it ever fires. **(T1)**~~ **CLOSED
     by the final fix wave (I3, 2026-08-17): the redundancy turned out
     to be the smaller problem.** The same three assertions —
     `exists()`, `is_dir()`, and a check that
     `src/assay/__init__.py` sits under the asserted `REPO` — pinned
     this machine's own checkout path
     (`/home/brice/workspace/assay`), which is the ONLY test in
     `tests/` that touches a machine-specific filesystem path, against
     a README that promises the suite "runs entirely from scripted
     fakes and recorded transcripts." All three filesystem assertions
     are removed; the test now pins only the portable property it
     exists for — `.worktrees` never appears in the wrapper's `REPO=`
     — on the string alone, with no filesystem access at all.

---

## Process lessons

1. **Sequencing a display-layer fix before the contract that reads
   it.** The false-drop fix (in `_diff_verdicts`) had to land before
   exit 3 could be safe to ship, and the wave's own evidence proves the
   ordering mattered rather than merely sounding prudent: a committed
   live-rerun profile pair
   (`docs/superpowers/evidence/live/granite-code-8b-instruct-q8_0-
   quick.json` against its `live-run2` counterpart) had, pre-fix, a
   `dropped` of exactly `('verdict.long_context',)` — the false drop,
   both sides byte-equal `unmeasured` — and post-fix, `dropped == ()`.
   In the reverse shipping order, this exact pair would have exited 3
   on a comparison that was actually complete: the false positive the
   ordering existed to prevent, demonstrated on real committed
   evidence rather than only a synthetic fixture. The controller's own
   preflight probe got half of this wrong before the Task 3 implementer
   caught it (it had claimed neither exit-3 fixture survived the
   false-drop fix; only one needed to change) — the correction is filed
   as an erratum in the raw ledger, not smoothed over.
2. **A test that cannot fail is worse than no test.** Three tests this
   wave were written, run, and found to pass against code that did NOT
   yet have the property they claimed to guard: the both-sides-absent
   drop test (`test_verdicts_keyed_but_unmeasurable_on_both_sides_are_
   not_dropped`, whose original brief deleted the key from both sides
   so the comparison loop never reached the guard at all — ruling 5),
   the parallel-baseline guard test (whose original brief fixtures
   either tripped an unrelated assertion first or produced a silent
   `None` result that never raised — ruling 9), and the recorder lock
   test (found twice: once for using a fixture that could not drive
   200 synthetic calls or was not thread-safe — ruling 10 — and once,
   even after fixing the fixture, for a docstring that overclaimed the
   lock was proven load-bearing when the test's own evidence said the
   opposite — ruling 12). Every one was caught by demanding a
   demonstration of failure against the unfixed code or an explicit
   payload-and-thread sweep, never by reading the test's text.
3. **A grep for a literal cannot find a semantic reference.** The
   schema bump's preflight probe searched for the literal `== 8` and
   predicted exactly two test updates; five were needed, plus two
   files the grep's own scope never covered
   (`src/assay/__init__.py`'s `__version__`, and a README schema
   string). The one instance the grep could never have found compared
   a frozen campaign corpus against the SYMBOLIC `PROFILE_VERSION`
   rather than a literal number, and had been passing only because
   that constant happened to still equal 8 — it would have silently
   validated a tampered corpus the moment the constant moved to 9. The
   fix pins the frozen corpus to the literal `8` forever, the same way
   the older `_COMMITTED_PROFILES` corpus is pinned to `{1, 2, 3, 4}` —
   a frozen corpus must be pinned to its own permanent schema, never to
   a moving constant, no matter how the pin happens to read today.
4. **A closed wording defect regressed and had to be closed a second
   time, at four sites, because it kept propagating from one document
   into the next.** Item 84's conflation — "ninety lanes (15 models ×
   k in {2, 4})," which computes to 30 — re-entered through the
   controller's own v1.8 design spec, was faithfully transcribed by an
   implementer into `src/assay/profile.py:72`, reached the CHANGELOG
   draft, and then reached `docs/CARRIED-DEBT.md`'s own item-15 closing
   text — the very entry that credits this project with having fixed
   this once already. Three of the four sites were found by three
   different readers (a reviewer, the controller's own audit, and a
   second controller audit that only found the fourth because the
   third had not existed yet at the time of the first sweep) rather
   than by one exhaustive pass. The fix that actually closed it was not
   a series of one-off edits at whichever site a reader happened to
   flag — it was a repo-wide grep with a per-hit verdict, so a fifth
   site could not hide behind a fix that only ever looked where the
   last reader had pointed.

---

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

8. ~~**Three grade-ordering surfaces agree only by coincidence** —
   `profile.py` follows the live `GRADES`; `report.py` and `diff.py`
   freeze the triple. The fix is a cross-surface equality test over a
   deep fixture; nothing fails today if they diverge.~~ **CLOSED in
   v1.8** — `test_the_three_grade_orderings_agree` pins the equality
   over a deep six-grade fixture and pins the order itself (`tiny,
   small, medium, constrained, nested, tabular`). Guard, not a fix:
   nothing was reordered, and nothing failed at the time it was
   written — but a future divergence between the three surfaces now
   fails a test instead of only a review. **(T4/T10)**
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

15. ~~**No verdict floors — and they are now derivable.** v1.7 shipped
    `parallel` as measurement-only on the stated ground that a rung
    invented without a measured floor is the overclaim the rest of the
    schema exists to refuse. That ground is gone: fifteen live rows now
    exist, all reading `mode: parallel` at both k = 2 and k = 4, with
    `degradation_ratio` between 0.995 and 1.007 across a 10× span of
    single-lane speed (28 → 288 tok/s). **The natural v1.8 opener.**~~
    **CLOSED in v1.8** — `verdict.parallel` ladders the worst measured
    k, mode-gated, at chosen floors 0.8/0.5 carrying
    `floor_provenance`. "Derivable" was the optimistic word: the
    cluster sanity-checks a ladder without exercising a boundary, so
    the floors are chosen and say so. The thirty k-readings are
    re-laddered by the suite on every run.
16. **The overlap tolerance sanity check ran, and the flag stands.**
    `OVERLAP_TOLERANCE_S` is 0.25 s; every campaign profile records
    `tolerance_provenance: "chosen-2026-08-17"`. **No real endpoint in
    the campaign read `serialized`** — fifteen models × two k values,
    `n_lanes_ok` full, `lane_errors` empty, `skipped: []` throughout. So
    the constant is consistent with every live row and **exercised at
    its edge by none of them**, which is a weaker claim than a derived
    threshold. Retiring the flag needs an endpoint that actually
    serializes; this tier has not produced one.

    *v1.8 note: the same missing endpoint leaves the new verdict's
    `serialized` gate unexercised by live data too. Both are pinned by
    `test_no_campaign_row_ever_read_serialized`, which asserts the
    absence rather than letting it be forgotten.*

    *v1.9 note (2026-08-18): the constant is renamed and re-typed —
    `OVERLAP_TOLERANCE_S` (0.25 s, absolute) is retired and
    `OVERLAP_FRACTION` (0.25, dimensionless — 25% of the shorter
    lane's span) takes its place, with `overlap_provenance` carrying
    forward the same **chosen, not derived** flag under its own name.
    This item is amended, not closed. What it was waiting for had two
    parts: the old constant was unexercised by live data, AND it could
    read a genuinely concurrent fast endpoint as `serialized` from
    speed alone — the tolerance cliff this file already flagged. v1.9
    removes the second part: §2 of the v1.9 design spec verifies, at
    0.05 s / 0.222 s / 1.0 s, that concurrent lanes now read `parallel`
    at every scale a fixed-seconds tolerance used to fail, while a
    near-miss (2 ms of overlap on a 200 ms span) still correctly reads
    `serialized` — the client-skew guard the tolerance existed to
    provide, preserved. What remains is exactly the first part,
    narrowed to a single boundary: retiring the flag still needs an
    endpoint that actually serializes, and this tier still has not
    produced one. **Correction (I3, final fix wave, 2026-08-18): this
    note originally said "no live row has ever overlapped by between 0%
    and 25% of a span" here. That overstates what is known. Profiles
    store only the derived `mode`, never the spans — `ParallelRow` has
    no span field — and the transcripts carry no timing, so no overlap
    fraction was ever recorded for any live row and none can be
    recovered from anything committed; "0% to 25%" is not a range
    anything measured. The claim the evidence actually supports is
    narrower: no live row has ever read `serialized` under the retired
    absolute-seconds rule. The same overclaim also appears in the
    committed design spec at
    `docs/superpowers/specs/2026-08-18-assay-v1.9-scale-free-overlap-design.md`
    §4 ("no live row has ever overlapped by between 0% and 25% of a
    span"), in the committed plan at
    `docs/superpowers/plans/2026-08-18-assay-v1.9-scale-free-overlap.md`
    (the drafted CHANGELOG block at line 530 and the Step 5 task text at
    line 542, same wording), and in `CHANGELOG.md`'s v1.9 entry; per
    this project's convention the spec and plan are committed process
    record and are not rewritten after the fact — this correction is
    the erratum for both, so the spec's §4 claim and the plan's Step 5
    text should both be read with the same correction applied, and
    CHANGELOG.md is edited directly since it is in-branch and
    unpublished.** The fifteen campaign profiles were left
    exactly as measured — not rescored, no campaign re-run — and still
    carry `tolerance_s` / `tolerance_provenance` under their own
    committed schema version, which the v1.9 renderers read in its own
    terms rather than converting to a fraction nobody measured.

    **Erratum, same note:** the v1.9 spec and plan both wrote "the
    fifteen committed **v9** profiles." They are **v8**. Verified by
    reading `assay_profile_version` out of all fifteen files under
    `docs/superpowers/evidence/tier-enthusiast-2026-08/`: every one
    reads `8`. The v1.8 wave bumped `PROFILE_VERSION` 8 → 9 but
    deliberately never re-ran the campaign (see this same item 16's
    v1.8 note and item 15's closure, above), so no committed profile
    has ever been v9 — and CHANGELOG.md's own v1.8 entry already said
    "v8 profiles" correctly at the time. v1.9 repeats the same
    non-event a schema version later: schema moves to v10, the
    fifteen profiles stay v8, still unrescored. Per this project's
    convention the committed spec and plan are not rewritten; this is
    the dated amendment sitting beside the claim instead. Correct
    phrasing used going forward: *pre-v10 profiles*, which happen to
    be v8.*
17. Evidence-class strings are re-declared here against `speed.py`'s
    inline literals; the fix is a shared tuple in `speed.py`
    (scope-blocked at the time). **(T5)**

    *v1.8 note: this got WORSE, not better. `profile._parallel_verdict`
    needed the same weakest-first ordering to rank a k's evidence class
    and added a THIRD copy, `profile._PARALLEL_EVIDENCE_WEAKEST_FIRST`
    — confirmed byte-identical to `parallel._EVIDENCE_WEAKEST_FIRST` by
    the T4 reviewer. Two duplicates scope-blocked at v1.7's close; v1.8
    added a third rather than collapsing any of them. The fix is still
    the same one shared public tuple, still not taken. (T4)*
18. ~~The `baseline is None` guard is missing — a `TypeError` after
    spend, rather than a clean error.~~ **CLOSED in v1.8** —
    `probe_parallel` raises `ValueError` before any lane runs.
    Non-vacuity shown as a contrast: pre-fix, timed lanes reach
    `TypeError: '>' not supported between instances of 'NoneType' and
    'int'` with `meter.spent.calls == 2` (the spend the guard exists to
    prevent); post-fix, `ValueError` fires with `meter.spent.calls ==
    0`. The brief's first fixture (`LaneFake({})` / `PinnedSpans([])`)
    would not have shown this — timing-free lanes make `per_lane is
    None`, which short-circuits the comparison and hides the defect
    behind a silent `degradation_ratio=None` — so the test uses lanes
    that report timings instead. **(T5/T9)**
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
29. ~~**"What starts, finishes" is a CALL-meter guarantee only**, and
    **no test exercises the honest partial**: an admitted family can
    still die on the token meter mid-run and keep what it measured. The
    docstring says so (ruling 14 fixed the README and CHANGELOG); the
    test does not exist.~~ **CLOSED in v1.8** —
    `test_a_family_admitted_on_calls_can_still_die_on_the_token_meter`
    pins the derived charges (38 + target tokens per rung: 550 / 1612 /
    3698 cumulative over the first three rungs) and asserts a
    2000-token budget cuts at the `2048` rung — named in `skipped`, not
    inferred from an absent row — while `max_calls` stays far out of
    reach, so the cut is unambiguously the token meter's. **(T9/T11)**
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
43. **Quick's margin is 9 calls (121 worst case of 130) — a live
    constraint, not a note.** The deep grades cost quick 15 codec calls
    and the next quick-mode call added anywhere spends the remainder.
    Full has headroom by derivation; quick does not. *Restated at the
    final fix wave: this item was raised as "124 of 130, margin 6", off
    the hand-counted bisection tail that item 91 closed at T8. The
    derived tail is 4 calls for quick, which puts the worst case at 121
    of 130 — the numbers the CHANGELOG and `run.WORST_CASE` carry. The
    constraint the item records is unchanged; only its arithmetic was
    superseded, and a debt file quoting retired figures is a debt file
    a reader has to check twice.* **(T3/T6)**
44. The headroom prose double-counts the bisection: the clean-run
    figure already contained the ladder's own calls, so the worst case
    the prose derived from it was wrong and the cost-narrative clause
    built on that number was false. *Restated at the final fix wave for
    the same reason as item 43: the "546 already contained ~12 ladder
    calls, so the true worst case was ~553" arithmetic this item was
    raised with came from the same hand count its closed companion
    (item 91) retired at T8. The derived terms are a bisection of at
    most 4 calls quick / 8 full over T6's re-measured 552-call clean
    full run, which puts full's worst case at 560 of 610. The
    double-count the item records stands; its figures do not.* **(T3)**
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
61. ~~The published matrix page was not visually opened during T14 — its
    content was verified by parsing, not by eye. The final review takes
    the look.~~ **CLOSED at the final fix wave** — the whole-branch
    review took the look and reported the page sound. **(T14)**
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
74. ~~The campaign corpus glob in `tests/test_profile.py` lacks the
    version-key filter its sibling corpus has, so a non-profile JSON
    landing in that directory would `KeyError` rather than fail a named
    assertion.~~ **CLOSED in v1.8** — `_CAMPAIGN_PROFILES` now filters
    on `"assay_profile_version" in payload`, the same rule
    `_COMMITTED_PROFILES` already used. Confirmed non-narrowing: 15
    filtered, 15 unfiltered, all three campaign-corpus-dependent
    assertions (including Task 7's `len(_CAMPAIGN_PROFILES) == 15`)
    still pass. **(T14/T9)**
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
84. ~~"Thirty lane readings" is loose wording in the CHANGELOG and the
    diffs read-out: thirty is the count of k-readings (15 models × 2 k
    values); the lanes themselves number 90. This file says it
    correctly; those two do not.~~ **CLOSED at the final fix wave** —
    both now say k-readings and name the ninety lanes beside them.
    **(T14)**
85. ~~CARRIED-DEBT.md is linked from nothing. A durable ledger nobody
    can find is halfway to a dead pointer. Deliberately not fixed here
    to keep the fix round's scope tight; ledgered for the final
    review.~~ **CLOSED at the final fix wave** — README's version-history
    paragraph and the v0.9 CHANGELOG entry both point here. **(T14)**
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
101. ~~`CallRecorder`'s write lock is untested — `test_replay` has no
     concurrency case.~~ **PARTIALLY CLOSED in v1.8** — the concurrency
     case now exists
     (`test_call_recorder_keeps_every_row_whole_under_concurrent_writers`,
     8 threads × 25 calls against a stateless fake, 200/200 rows
     present and parseable) but does NOT prove the lock is load-bearing
     — it passed identically with the lock removed, swept at 100 B /
     8 KB / 64 KB payloads × 8 threads, with and without the lock:
     120/120 rows, 0 unparseable, all six configurations. See the v1.8
     section (item 101 update) for the full finding and why the test
     and the lock both stay anyway. **(T6/T11)**
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

---

## Inbound findings — bloomery drift-watch live acceptance (2026-08-17)

Measured against assay 0.9.0 by bloomery's drift-watch boots (evidence:
bloomery `docs/superpowers/evidence/2026-08-17-drift-watch-live.md`,
PR #12). Recorded here because they are assay-instrument facts, not
bloomery defects; v1.8 candidates.

1. ~~**`diff --gate` exits 0 ("no drift beyond noise") on a v8-vs-v4
   pair while five measured families vanish** (`long_output`,
   `tool_calling`, three deep `json_object` cells go unmeasured-on-one-
   side). Literally true under the gate's rules — vanished families are
   unmeasured, not drifted — but a consumer trusting exit codes alone
   reads "nothing changed" across an instrument boundary. bloomery
   guards it with a version precheck that refuses to run the diff at
   all; assay's own gate could rank wholesale family disappearance as
   at least reportable (a distinct exit, or gating on it).~~ **CLOSED
   in v1.8** — exit 3 ("incomplete") fires whenever a cell was measured
   on exactly one side, in both plain and gate mode, and outranks exit
   1. The consumer reading exit codes alone is now told the truth.
2. ~~**`diff` prose falsely reports `dropped: verdict.long_context` for
   equal objects** (reproduced twice, both sides byte-equal on that
   verdict). Prose-only — the gate's exit code is unaffected — but the
   display layer misnames an unchanged field.~~ **CLOSED in v1.8** —
   `_diff_verdicts` adopted the sibling families' rule. It stopped
   being cosmetic the moment exit 3 read `dropped`, which is why it
   landed first.
3. **assay 0.5.0 has no `diff` subcommand; its argparse exits 2**,
   which a consumer maps to "not comparable" — an accidental
   right-answer-for-the-wrong-reason edge, unreachable behind a version
   precheck, recorded so nobody relies on it.
