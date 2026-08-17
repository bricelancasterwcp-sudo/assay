# Per-model diffs — `tier-enthusiast` (2026-08-13/15) → `tier-enthusiast-2026-08` (2026-08-17)

Fifteen `assay diff OLD NEW` runs, one per model in the 2026-08-17
re-profile campaign. Each `<slug>.txt` carries the command's own exit
code and both sides' provenance in its header. Nothing here is tidied:
where a number moved for a reason that is not the model, the reason is
named beside it rather than the row being dropped.

**Every one of the fifteen exited 1** — comparable, drift found. None
exited 2 (not comparable) and none exited 4 (unreadable). That is worth
stating because it was not the expectation: the brief anticipated
identity-gate FATALs from old rows that predate hardware-tier marking,
and there are none. Every profile in `../../tier-enthusiast/` declares
`provenance.tier: "enthusiast-16gb"` and `emulated: false`, the two
dry-run-era coder rows included, so the gate passed on all fifteen pairs
and every diff reports real cells.

## Which old profile each new one was compared against

Mapped by **model name**, not by filename — the slug convention changed
between the two directories.

Two models have two old rows apiece. Both were resolved by taking the
**newest old profile by `provenance.started`**, and the loser is named in
that diff's `# note:` header:

| model | chosen old profile | started | rejected | started |
|---|---|---|---|---|
| `qwen2.5-coder:14b-instruct-q4_K_M` | `coder-14b-fixtures-v2-quick.json` (schema v3, probe 0.4.1) | 2026-08-14T06:31:58Z | `qwen2.5-coder-14b-q4-quick.json` (schema v2, probe 0.3.0) | 2026-08-13T22:35:37Z |
| `qwen2.5-coder:7b-instruct-q8_0` | `coder-7b-fixtures-v2-quick.json` (schema v3, probe 0.4.1) | 2026-08-14T06:30:30Z | `qwen2.5-coder-7b-q8-quick.json` (schema v2, probe 0.3.0) | 2026-08-13T22:38:07Z |

The other thirteen map one-to-one (schema v4, probe 0.5.0).

## Read every diff through these four confounds first

The old and new runs differ in more than the calendar. **Nothing in these
files isolates a model change**; they are two measurements taken with
different instruments under different sampling.

1. **Mode: `quick` → `full`.** Two seeds instead of one, `envelope_n`
   30 instead of 10, `loop_runs` 5 instead of 3, codec cells sampled
   sequentially to a cap of 35 instead of a fixed 5.
2. **Fixture set: `codec-fixtures-v2` → `codec-fixtures-v3`.** Every
   codec landing rate below was measured against *different fixtures*.
   `diff` deliberately does not gate on fixture-set name, so it will
   subtract across the change without saying so. A `codec.*` movement is
   a movement in *this instrument's* reading, not a claim that the model
   edits patches better than it did on Friday.
3. **Ceiling cap: 16384 → 32768.** See below — this alone accounts for
   twelve of the fifteen `ceiling.max_verified` "improvements".
4. **Schema v3/v4 → v8.** Families that did not exist then appear in
   every diff's `dropped:` line (`verdict.long_output`,
   `verdict.tool_calling`, the three deep `json_object` grades), plus
   `ceiling_shapes.*` and `verdict.loop_discipline` on the two schema-v3
   coder rows. Dropped is the honest word: nothing was compared.

## Geometry — the E1 direction check

`assay diff` **has no geometry family**. It compares ceiling, shapes,
verdicts, codecs and speed; `geometry.kv_kib_per_token` and
`usable_window` appear in no diff file. The check below was therefore run
directly against the two profiles' `geometry` blocks, and the window law
(`assay.geometry.plan_window`: `(vram_free_mib − 512 MiB) ÷ kv_bytes`,
weights already resident) was replayed by hand.

The four E1-affected models in this directory
(`../../tier-enthusiast/ERRATA.md`; `codegemma` is `live/`-only and is not
in this campaign) **all move in the erratum's direction**, and each new
`kv_kib_per_token` equals the erratum's corrected value *exactly*:

| model | kv old → new | erratum's corrected kv | direction |
|---|---|---|---|
| `deepseek-coder-v2:16b-lite-q5_K_M` | 216 → **324** | 324 | up (over-promise corrected) |
| `qwen3.8:27b` | 216 → **260** | 260 | up (over-promise corrected) |
| `gemma2:9b` | 294 → **336** | 336 | up (over-promise corrected) |
| `mistral-nemo:latest` | 200 → **160** | 160 | **down** — the under-promise case |

Windows are a sanity check, not an identity, because free VRAM was
different on the two nights. Stated three ways:

| model | committed window old → new | vram_free old → new | corrected kv replayed at the **old** profile's own vram | erratum says |
|---|---|---|---|---|
| `deepseek-coder-v2` | 8092 → 9171 | 2219 → 3414 | **5394** | 5394 ✅ |
| `qwen3.8:27b` | 4922 → **3749** | 1552 → 1464 | **4096** | 4096 ✅ |
| `gemma2:9b` | 8192 → 8192 | 7788 → 9318 | **8192** (`training_ctx` binds) | 8192 ✅ |
| `mistral-nemo` | 32711 → **49324** | 6901 → 8219 | **40889** | 40889 ✅ |

Every one of the four reproduces the erratum's corrected window to the
token when the corrected kv is replayed at that profile's *own* recorded
`vram_free_mib`. Read each row:

- **`deepseek-coder-v2` — correct direction despite a window that grew.**
  This is the row that reads backwards at a glance. The window rose
  8092 → 9171 only because free VRAM rose 2219 → 3414 MiB. Held at
  today's VRAM, the *old* (too-small) kv would have promised **13757**
  against the true **9171** — the erratum's 33.3% over-promise, live.
- **`qwen3.8:27b` — the clean case.** kv up (216 → 260 KiB/token),
  window down 4922 → 3749 on essentially unchanged VRAM
  (1552 → 1464 MiB). The old kv would have promised 4513 today against
  the true 3749 — a 16.9% over-promise against the 16.8% the erratum
  filed (the erratum's figure comes off exact byte counts; 16.9% is
  recomputed from the committed, KiB-rounded geometry).
- **`gemma2:9b` — kv corrected, promise unchanged.** `training_ctx` 8192
  binds under both readings, exactly as the erratum predicted. The
  number that was wrong was fixed; the number a reader acts on never
  moved.
- **`mistral-nemo` — grows, and that is correct.** The stated
  `key_length` is *smaller* than the derivation here, so the old profile
  was too conservative. At today's VRAM the old kv would have promised
  39459 against a true 49324: it was under-promising by 20.0% of the
  true window.

**No E1 model moves the wrong way.** Nothing to escalate.

### The unchanged-by-construction models: kv holds exactly

For the eleven models the sweep classified UNAFFECTED or
UNAFFECTED-BY-CONSTRUCTION, `kv_kib_per_token` is not merely inside the
diff's noise discipline — it is **byte-identical** across a probe
version bump, a schema bump and a two-day gap:

`deepseek-r1:14b` 192, `hermes3` 128, `Hermes-4-14B` 160, `llama3.1:8b`
128, `phi4:14b` 200, `qwen2.5-coder:1.5b` 28, `qwen2.5-coder:14b` 192,
`qwen2.5-coder:7b` 56, `qwen3:14b` 160, `qwen3:8b` 144.

**One exception, and it is an absence rather than a disagreement:**
`gemma-4-12b-it-qat-q4_0:latest` committed `geometry: null` in *both*
profiles. It has no kv number on either side, so there is nothing to
hold. (The E1 sweep recorded this model's stated `key_length` 512 against
a derivation of 240 — the largest gap the sweep found, 2.1× — and it is
*still* unreadable at probe 0.9.0. A model whose geometry the instrument
cannot compute at all is a standing gap, not a corrected one.)

Every `usable_window` that moved among these eleven moved because
`vram_free_mib` moved; the four `training_ctx`-bound models
(`qwen2.5-coder:1.5b` 32768, `qwen2.5-coder:7b` 32768, `qwen3:8b` 40960,
`phi4:14b` 16384) show a window unchanged to the token.

## Required read-out: deepseek-r1:14b and Hermes-4-14B

Both come back with `envelope.fidelity` **0.0**, `failures.shape`
**30/30**, `failures.prose` 0, `failures.refusal` 0 — and every codec
cell settled at the schedule's **first look, n=5** (Hermes-4 at 5 and
10). Both then flip `verdict.patch_editing: risky → unusable` in the
diff, and `deepseek-r1` publishes `unusable` on all four of
`structured_extraction`, `patch_editing`, `loop_discipline` and
`tool_calling` in the matrix.

**This is the reasoning-preamble signature, not a capability floor.**
Read it as follows:

- `shape` 30/30 with `prose` 0 and `refusal` 0 means the model answered
  every single time, in the requested register, without refusing — and
  the reply never had the *shape* the envelope asked for. A model that
  could not do the task would show prose failures or refusals. A model
  that emits a chain-of-thought preamble before its answer shows exactly
  this: content present, container wrong, on every attempt.
- The n=5 settling is that same fact seen through the stopping rule, not
  independent evidence. The sequential schedule stops at the first look
  whose Wilson-95 interval decides a rung; 0/5 decides `unusable`
  immediately, so the probe correctly stopped paying for a cell that had
  already answered. Small n here means *decided fast*, not *sampled
  thinly*.
- `provenance.thinking` is `"disabled"` on both new profiles. The
  request asked for no preamble. Getting one anyway is a fact about the
  model's response to that request — which is precisely what the row
  should say — but it is a fact about **formatting compliance**, not
  about whether the model can extract structure or edit a patch.

**What it means for reading their rows:** a consumer choosing a model
for structured extraction or patch editing should read these two rows as
*"unusable through this probe's plain-prompt presentation"*, not as
*"cannot do this task"*. The honest next measurement is the same probe
under a presentation that strips or tolerates a preamble — that is a
lens change, and the profile records `presentation: default-v1` so the
comparison stays legal. Nothing in this campaign licenses the stronger
claim, and the matrix must not be read as making it.

Note the asymmetry that makes this a *signature* rather than a guess:
`Hermes-4-14B` scores `tools.right_tool_rate` 1.0 and
`args_valid_rate` 1.0 (call_rate 0.6, 15 of 40 turns truncated) —
perfectly-formed tool-call JSON from the same model whose envelope shape
fails 30/30. A model that cannot produce structure does not emit
valid tool arguments at 100%.

## Other findings — recorded, not tidied

**1. `ceiling.max_verified: 16384 → 32768` is the CAP moving, not the
model — and the diff calls it an improvement anyway.** Twelve of the
fifteen diffs carry this line as `(improvement, rung-change)`. Both
sides read `failure_mode: none_up_to_cap`: the old quick run stopped at
quick's cap of 16384 without failing, the new full run stopped at full's
cap of 32768 without failing. **Neither run found a ceiling.** The diff
compares the two numbers and scores a capability gain that nobody
measured. This is an instrument finding for a later wave: a
`rung-change` across two runs with different caps and identical
`none_up_to_cap` failure modes is not a comparison, and `diff` has no
term for it today.

**2. `gemma2:9b` `ceiling.failure_mode: hard_error → none_up_to_cap`.**
Same root cause, opposite surface. `ceiling_cap_for` narrows the ladder
by `training_ctx` in every mode **except quick** (`run.py`: `if mode !=
"quick"`). gemma2's `training_ctx` is 8192, so the old quick run marched
the ladder past the model's trained window and got a `hard_error`, while
the new full run narrows to 8192 and reports `none_up_to_cap`. The model
did not get more robust; the ladder stopped asking an illegal question.

**3. `mistral-nemo` — the ceiling "regression" is a daemon telemetry
dropout, not a context loss.** `ceiling.max_verified: 16384 → 3328`,
`failure_mode: none_up_to_cap → missing_stats`, `verdict.long_context:
ready → risky`. The evidence array says exactly what happened: at
est_tokens 4096 **seed 0**, the reply came back with no `tokens_in`,
`tokens_out` or `stop_reason` at all —
`ContractViolation: backend promised token counts but reply lacks
tokens_in, tokens_out, stop_reason`. The ladder read that as a failure,
bisected down, hit the same dropout again at 3584 seed 0, and settled at
3328. **Seed 1 passed 4096 cleanly** (`tokens_in=4019`), and the shape
ladder is byte-identical to the old run at all three shapes
(1664 / 3712 / 7808, `ok_to_shape`) — a model still reading 7808-token
prompts. The old quick run had only seed 0 and passed 4096 fine
(`tokens_in=4001`), so the dropout is transient, not deterministic.
This is the **same daemon-flakiness class** as the campaign's one
transient HTTP 500 (qwen3:14b, rerun clean). It is recorded, not fixed:
the published row says `risky` because that is what was measured, and
this note is where a reader learns why. A ladder that treats a telemetry
gap as a context failure is a second instrument finding for a later
wave.

**4. `qwen3.8:27b` `speed.decode_tps` 23.56 → 28.37 is flagged on the
weakest basis available.** Its `basis` reads `threshold-20pct-assumed` —
+20.4%, just over the assumed threshold, with no per-sample spread on
one of the two sides to justify a real interval. It is the only speed
cell that moved in the whole set; every other model's `decode_tps` and
`prefill_tps` sat within noise. Read as "probably faster", not as a
measured 20%.

**5. Verdict regressions that are not E1 and not caps.**
`loop_discipline: risky → unusable` on `deepseek-coder-v2` and
`qwen3.8:27b`; `structured_extraction: ready → risky` on `gemma2:9b`;
`patch_editing: risky → unusable` on `deepseek-r1` and `Hermes-4-14B`
(the preamble signature above). All five sit downstream of confounds 1
and 2 (fixture-set v2 → v3, and 5 → up-to-35 samples with a Wilson
interval that can now *decide* where the old fixed n=5 could only guess).
Several of the neighbouring `.provisional: True → False` flips say the
same thing from the other side: the new sample is what finally separated
a verdict from its neighbours. **None of these five is evidence that a
model got worse**, and none should be cited as such without a same-mode,
same-fixture re-measurement.

**6. The three coder models and `deepseek-r1` stop the tools family at
n=5 with `composite: 0.0`.** `qwen2.5-coder` 1.5b/7b/14b and
`deepseek-r1:14b` all report `tools.supported: true`, `call_rate: 0.0`,
`n_tasks: 5`. `supported: true` here means the *endpoint* accepted a
tool schema, not that the model used one; 0/5 decides `unusable` at the
first look, so the schedule stopped. Correct behaviour, and a row a
reader can misread as "tools measured over 20 tasks and failed" — it is
five tasks, decided.

**7. The whole parallel family reads `mode: "parallel"` on all 15
models at both k=2 and k=4**, `n_lanes_ok` full, `skipped: []`,
degradation ratios between 0.995 and 1.007. No live endpoint in this
campaign read serialized. That is a fact about the corpus, and it is
carried into `docs/CARRIED-DEBT.md` as the tolerance sanity-check
outcome: `OVERLAP_TOLERANCE_S` was never the binding term across 30 live
lane readings, so its edge remains untested by evidence.
