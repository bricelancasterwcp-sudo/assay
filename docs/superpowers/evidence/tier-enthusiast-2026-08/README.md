# Tier re-profile 2026-08 — enthusiast-16gb, assay 0.9.0 (schema v8)

Profiles measured on real hardware (RTX 5080, 16 GiB; ollama 0.32.13) at
`--full`, tier `enthusiast-16gb`. Written by
[`scripts/campaign-2026-08.sh`](../../../../scripts/campaign-2026-08.sh);
one `<slug>.json` and one `<slug>-transcript.jsonl` per model, slug rule
`[:/] → -`.

This directory holds **all fifteen rows**. The campaign ran 2026-08-17,
08:35:45→11:42:17 −05:00 (13:35:45→16:42:17Z), fifteen models in
weights-ascending order. The run log carries **sixteen `done` lines and
no `skip` line**: fifteen at exit 0 — one per committed row — plus the
one exit-4 attempt described next.
[`campaign-run.log`](campaign-run.log) is the durable timing record and
pins the instrument (`assay_version=0.9.0`, `assay_commit=d9b9792`).

## `qwen3:14b` was run twice; the second run is the row

`qwen3:14b`'s first attempt died at ~call 308 on an HTTP 500 from
ollama's `/api/generate` (exit 4, **no profile written**), and it was
re-run clean from zero rather than resumed. Both the failure and the
rerun are recorded in the run log as a dated `#` comment above the
second `start` — see the line beginning `# rerun qwen3:14b`. The
committed `qwen3-14b.json` is the second, complete run
(16:29:04→16:42:17Z, 793 s).

---

## The dry-run row has been superseded by the campaign's own

**Which row now stands:** the file at
`qwen2.5-coder-1.5b-instruct-q8_0.json` is now the **campaign's** probe
of that tag, not the dry run. Its provenance reads `started`
**2026-08-17T13:35:46Z** → `finished` **13:39:25Z**, which is the run
log's first `start`/`done` pair (`08:35:45`/`08:39:25` −05:00). The
dry-run row carried `started` **13:09:17Z**; those bytes are not in the
tree, they are in git history at commit `d9b9792`. The transcript at the
same slug was overwritten with it — `CallRecorder` truncates the
transcript on open (`src/assay/replay.py`) — so both files at this slug
are the campaign's, from one run.

The two agree where determinism says they must and differ where sampling
says they may: identical `spent` (472 calls, 211 056 prompt tokens) and
identical `calibration.chars_per_token` (5.861313868613139), against
`decode_tps` **286.26** (campaign) vs **286.65** (dry run), a 0.14 %
spread. That is the shape of two clean runs of the same script, not of
one disturbed one.

The section below is kept because it describes what the superseded row
measured and why it was still usable — a limit that is deleted the
moment it stops applying teaches nobody how the next one was caught.

### What happened to the dry-run row (superseded, 13:09:17Z)

`qwen2.5-coder-1.5b-instruct-q8_0.json` **as it stood before the
campaign** was measured on 2026-08-17 (13:09:17→13:12:58Z, 221 s wall,
472 calls, exit 0). **During that run, the model was unloaded from VRAM
once** — a wrapper helper (`unload_all`) was smoke-tested against the
live daemon while the probe was in flight, and ollama reloaded the model
on the next call. Measured cold-load cost for this tag: **1.76 s**.

This is recorded here rather than only in a report because a limit that
does not travel with the evidence is not a limit anyone will find.

#### What it affected

**The wall clock, and only the wall clock.** That profile's provenance
`started`→`finished` span of 221 s included up to 1.76 s (**0.8 %**) of
reload a clean run would not have paid, so 221 s was an upper bound —
which the campaign hour estimate read it as. The campaign's undisturbed
run of the same tag came in at **219 s** (13:35:46→13:39:25Z), inside
that bound.

#### What it did not affect, and why

(All figures in this subsection are the **superseded** row's, at
`d9b9792`.)

Not "no call errored" — a reload stall does not produce an error, it
produces a *slow reply*, so an all-`reply` transcript proves nothing about
timing. The defence is that **the speed family itself shows no stall
signature**, on three independent grounds:

1. **Mechanism.** Every speed number in that profile carried
   `evidence: "server_timings"`, which assay computes as
   `eval_count / eval_duration` from ollama's reply body
   (`src/assay/speed.py`). Ollama reports `load_duration` as a *separate*
   field; model load time is not inside `eval_duration`. Reload latency is
   therefore excluded from these figures by construction, not by luck.

2. **The samples are tight.** `speed.decode_samples` =
   **284.83 / 286.97 / 288.16** tok/s (n=3), a spread of 1.2 %. A reload
   caught inside a timed sample would show up as one conspicuous outlier.
   There is none.

3. **The number went up, not down.** Measured `decode_tps` **286.65** tok/s
   against the historical **271.93** for this same tag
   ([`../tier-enthusiast/qwen2-5-coder-1-5b-instruct-q8_0.json`](../tier-enthusiast/qwen2-5-coder-1-5b-instruct-q8_0.json),
   probe 0.5.0) — **+5.4 %**. The independently-measured parallel family
   agrees: `baseline_decode_tps` 286.65, per-lane **286.77** at k=2 and
   **288.34** at k=4, `degradation_ratio` 1.0004 and 1.0059, `n_lanes_ok`
   2 and 4, and **`lane_errors` empty in both rows**.

The correctness families — codecs, envelope, ceiling, geometry, loop,
long_output, tools — are scored on reply *content*, not on timing, and a
reload changes neither.

### Supersession — done, not pending

The campaign runs this model **first** (it is smallest, and the wrapper
orders by weights ascending), and it did: **the campaign row has
replaced the dry-run row** at this path, exactly as intended. Nothing
was edited to make that happen — the probe was re-run and wrote over
both files, which is the only way committed evidence is ever replaced
here, per [`../tier-enthusiast/ERRATA.md`](../tier-enthusiast/ERRATA.md).
The superseded bytes remain readable at `d9b9792` for anyone checking
this account against them.

## Scope note: this directory is outside the E1 sweep

Erratum E1 (`head_dim` derived instead of read) was fixed in probe
**0.7.0**. These profiles are written by **0.9.0** and cannot carry it, so
they are unswept by construction — see the *Scope bound (amended
2026-08-17)* section of
[`../tier-enthusiast/ERRATA.md`](../tier-enthusiast/ERRATA.md). The
exclusion is enforced by `probe_version` comparison in
`tests/test_e1_sweep.py`, not by a filename list.
