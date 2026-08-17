# assay v1 live validation — 2026-08-12

Task 12 of the v1 plan, run against Ollama **0.32.4** at
`http://127.0.0.1:11434` (daemon process up since 2026-07-27, unrestarted
throughout), three families, `--quick`, transcripts recorded. Two runs:
**run 1** (`live/`, `*-quick.jsonl`) before the sampler was pinned;
**run 2** (`live-run2/`, `*-quick-run2.jsonl`) at `temperature 0.2`
(`0050bdf`). Every probe completed within budget on all three models,
exit 0, all cells measured — spec §12 criterion 1's mechanics held.

## What reproduced (blind)

- **KV arithmetic, exactly**: qwen **56** KiB/token, granite **144**,
  codegemma **336** — the "codegemma's window costs 6× qwen's" finding,
  re-derived by assay from `/api/show` metadata alone.
  *(Erratum, 2026-08-17: the codegemma figure carries E1 — gemma's
  metadata states `attention.key_length` 256 where v1's derivation read
  192, so codegemma's true cost is **448** KiB/token and the ratio is
  **8×**, not 6×. The reproduction claim stands — assay re-derived
  exactly what the derivation-era code computes — but the number was
  the derivation's, not the model's. qwen and granite are unaffected
  (their metadata states no `key_length`). See
  [`e1-sweep/`](e1-sweep/PROTOCOL.md) and
  [`live/ERRATA.md`](live/ERRATA.md).)*
- **Per-model honest ceilings**: granite verified 4096 / first failure
  4352 (`hard_error`); codegemma 8704 / 9216 (`hard_error`) — real,
  training-context-shaped limits, honestly failed by the daemon
  (`truncate: false` + widened `num_ctx` doing their job).
- **Envelope**: fidelity 1.00 for all three at temp 0.2 (granite 0.90 in
  run 1 — one shape failure at the unpinned temperature).
- **`json_object` discriminates by family**: qwen 15/15, codegemma
  15/15, granite 1–3/45 across both runs. This is the codec the VTT
  consumes, and its verdicts came out exactly as an application would
  need them (`structured_extraction`: qwen/codegemma `ready`, granite
  `unusable`).
- Full suite green, no GPU, 166 tests in 0.11s — criterion 3 met.

## Finding 1: the ~11.5k daemon ceiling is TRANSIENT (criterion 1, partial)

The pinned expectation — ceiling ≈11.5k, mode `missing_stats` — did
**not** appear: qwen's ladder read `none_up_to_cap` at 16384 with intact
counts at every rung. Follow-up under robigo's exact 2026-08-10
conditions (15,792-token prompt, `num_ctx: 32768`, seed 0, temp 0.2):
`prompt_eval_count` present, `done: true`, `done_reason: length` — the
protocol held. **Same daemon process** (up since 07-27), same version,
that failed 40/40 with stats-free 200s two days earlier. The bug is
state-dependent (mechanism unresolved: load history, KV pressure, or
cache state), not version- or restart-dependent.

Two lessons, recorded rather than papered over: (a) an environmental
bug is not a stable target for a success criterion — the criterion
should have named the *behavior class*, not the address it lived at;
(b) this is the strongest argument for what assay is: a capability
profile is a **point-in-time measurement of a serving state**, which is
why provenance carries timestamps and why applications should re-probe,
not cache forever.

Bonus observation from the manual call: at 15.8k the protocol held but
the *output* was degenerate repetition ignoring the front canary — the
`attention_loss` class assay records as model evidence. Quality dies
before protocol does.

## Finding 2: edit-codec landing is INSTRUMENT-DEFINED (criterion 2, part met / part explained)

qwen `search_replace` landed **0/15 in both runs** where robigo's stage 2
— same model, same daemon — measured 100%. Investigated to ground truth,
not adjusted:

- qwen's replies were **semantically correct fixes in every sampled
  probe** — well-formed SEARCH/REPLACE blocks with the right change —
  failing on exactly one formality: the SEARCH lines' leading
  indentation is stripped. Reproduced at temp 0.8 and 0.2, bare and
  inside a fenced directive (one manual call). `whole_file` failed
  byte-equality by editorializing incidentals (rewrote the `# BUG`
  comment, added a blank line) while fixing the defect correctly.
- The two instruments differ in **landing definition**: robigo stage 2
  scores "parses as a patch action + codec applies + result parses as
  Python" (its own docstring: *"whether the edit is semantically right
  is stage 4's question"*). assay v1 scores "applier accepts AND result
  equals expected byte-for-byte" (spec §7). Byte-equality on
  `whole_file` measures compliance-with-incidentals, not edit ability.
- They also differ in **presentation**: robigo's probe presents its
  loop's full action envelope (`patch <path>` + fenced payload +
  system verb list) — and robigo's own history records that template-
  only scored 0/5 where the enveloped form scored 5/5 ("a stage that
  omits the shape it is trying to predict was never measuring the
  model"). Under robigo's shape, today's stage-4 transcripts show qwen
  emitting correctly-indented SEARCH lines; under assay's minimal
  instruction it never did, 30/30.
- **Temperature was falsified as the discriminator** (run 2 unchanged)
  but stays pinned: an unpinned sampler is an uncontrolled instrument
  variable regardless (granite's stray landings moving 0.2→0.0 between
  runs shows the n=5 noise scale).

Not done, deliberately: mutating the probe until the number matches
robigo's. That is the "tune fixtures to raise a landing rate" move the
ancestry bans — a low rate under a named instrument IS the result.

**v1.1 design consequence** (backlog, not patched tonight): the profile
must *name its instrument*, and the codec probe should (a) accept a
consumer-supplied presentation/directive so the landing rate predicts
the caller's actual prompt shape, and (b) offer robigo's
applies-and-parses lens alongside byte-equality. `patch_editing:
unusable` for all three families is true **under this lens** and the
lens belongs in the verdict's name.

## Finding 3: geometry reads VRAM before the model loads

granite and codegemma report `usable window 0 (limited by vram)` because
`free_vram_mib` is read before assay's own probes load the model (qwen,
resident from earlier work, read sanely: 32768 limited by training_ctx).
The residency rule works; the *ordering* is wrong for cold models.
v1.1: read geometry after calibration's first live call, or re-read
`/api/ps` at geometry time. Recorded; not silently corrected in the
artifacts.

## Criterion verdicts (spec §12)

1. Quick probe completes within budget, geometry named its binding
   term, ceiling measured — **met**, except the 11.5k signature which
   is shown transient (Finding 1).
2. Family split — **met** for `json_object`, KV arithmetic, envelope,
   and per-model ceilings; **not comparable** for edit codecs
   (Finding 2: different landing definitions and presentation; cause
   identified and recorded, code behaving exactly as its spec ruled).
3. Suite green, no GPU, under 60s — **met** (166 tests, 0.11s).
4. VTT sufficiency — the `json_object` grades + ceiling + verdicts are
   exactly the fields the VTT adaptations need; formal check moves to
   the VTT-side spec as planned.
