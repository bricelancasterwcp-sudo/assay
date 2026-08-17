# Tier re-profile 2026-08 — enthusiast-16gb, assay 0.9.0 (schema v8)

Profiles measured on real hardware (RTX 5080, 16 GiB; ollama 0.32.13) at
`--full`, tier `enthusiast-16gb`. Written by
[`scripts/campaign-2026-08.sh`](../../../../scripts/campaign-2026-08.sh);
one `<slug>.json` and one `<slug>-transcript.jsonl` per model, slug rule
`[:/] → -`.

This directory currently holds **one row**, the campaign's dry run. The
other fourteen land when the campaign runs.

---

## The dry-run row was disturbed mid-measurement

`qwen2.5-coder-1.5b-instruct-q8_0.json` was measured on 2026-08-17
(13:09:17→13:12:58Z, 221 s wall, 472 calls, exit 0). **During the run, the
model was unloaded from VRAM once** — a wrapper helper (`unload_all`) was
smoke-tested against the live daemon while this probe was in flight, and
ollama reloaded the model on the next call. Measured cold-load cost for
this tag: **1.76 s**.

This is recorded here rather than only in a report because a limit that
does not travel with the evidence is not a limit anyone will find.

### What it affects

**The wall clock, and only the wall clock.** The profile's provenance
`started`→`finished` span of 221 s includes up to 1.76 s (**0.8 %**) of
reload that a clean run would not have paid. Anyone reading an elapsed
time off this row — the campaign hour estimate does exactly that — should
treat 221 s as an upper bound.

### What it does not affect, and why

Not "no call errored" — a reload stall does not produce an error, it
produces a *slow reply*, so an all-`reply` transcript proves nothing about
timing. The defence is that **the speed family itself shows no stall
signature**, on three independent grounds:

1. **Mechanism.** Every speed number in this profile carries
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

### Supersession

The campaign runs this model **first** (it is smallest, and the wrapper
orders by weights ascending), so a real campaign overwrites this row with
an undisturbed one at the same path. That is intended. Until then the row
stands as measured — evidence is not rewritten to suit a later fix, per
[`../tier-enthusiast/ERRATA.md`](../tier-enthusiast/ERRATA.md).

## Scope note: this directory is outside the E1 sweep

Erratum E1 (`head_dim` derived instead of read) was fixed in probe
**0.7.0**. These profiles are written by **0.9.0** and cannot carry it, so
they are unswept by construction — see the *Scope bound (amended
2026-08-17)* section of
[`../tier-enthusiast/ERRATA.md`](../tier-enthusiast/ERRATA.md). The
exclusion is enforced by `probe_version` comparison in
`tests/test_e1_sweep.py`, not by a filename list.
