# E1 sweep — protocol (pre-registered)

**Filed** 2026-08-17, **before** the classifier ran. The sweep settles
the `ERRATA.md` "Not checked" paragraph: every committed profile gets
exactly one classification row against erratum E1 (head_dim derived
instead of read; see `../tier-enthusiast/ERRATA.md`).

## Scope

All committed capability profiles in `docs/superpowers/evidence/`
(any JSON with `assay_profile_version`): 23 files — `live/` (3, v1),
`live-run2/` (3, v1), `tier-enthusiast/` (17: 2×v2, 2×v3, 13×v4) —
over 17 distinct model tags. One v4 profile
(`gemma-4-12b-it-qat-q4_0-latest.json`) committed `geometry: null`;
its row records that no kv number exists to correct, and its model is
still classified for the record.

## Classification rule (fixed by ERRATA.md before this sweep)

Per profile, from its model's metadata:

- **AFFECTED** — the architecture states an `attention.key_length` and
  it differs from `embedding_length // attention.head_count`.
- **UNAFFECTED** — the stated `attention.key_length` equals the
  derivation.
- **UNAFFECTED-BY-CONSTRUCTION** — the metadata omits
  `attention.key_length`; probe 0.7.0's fallback IS the derivation, so
  it reproduces the committed number. (This says the profile does not
  change under 0.7.0 — not that the number is verified against the
  hardware.)
- **UNDETERMINED** — no metadata source reaches the profiled blob
  (model absent, /api/show fails, or the identity gate below fails).
  Recorded as unmeasured, never assumed clean.

Direction is recorded per AFFECTED row: stated > derived means kv was
understated and the window over-promised (the E1 failure mode); stated
< derived means the committed window was conservative — still wrong,
opposite sign.

## Metadata source and identity gate

Source: live `/api/show` from the box's ollama daemon (the same daemon
family that wrote the profiles), captured verbatim into this
directory, plus the `/api/tags` entry read in the same breath.
Extraction runs through `assay`'s own `OllamaNative.model_info()` and
`plan_window` — the current instrument, not a re-implementation.

Tags may have been re-pulled since a profile was written, so a
metadata match to the *profiled* blob must be earned, not assumed.
Identity gate, both parts required for an AFFECTED/UNAFFECTED verdict:

1. `weights_bytes` from today's `/api/tags` equals the profile's
   committed `model.weights_bytes`.
2. Reproduction: feeding the DERIVED head_dim through today's window
   law under the profile's own conditions (below) reproduces the
   committed `kv_kib_per_token`, `usable_window`, and `limited_by`
   exactly. A profile whose numbers the derivation cannot reproduce is
   not explained by E1 alone → **E1-INCONSISTENT**, named for separate
   investigation, never forced into a bucket.

## Recompute conditions (AFFECTED rows only)

Exactly the confirmed-rows procedure: `plan_window` under the
profile's OWN recorded `vram_free_mib`, `user_cap=None`, `kv_bits=16`,
default overhead, `loaded=True` forced (a profile writes its geometry
mid-run with its own model resident). Corrected `kv_kib_per_token` and
`usable_window` are reported beside the committed values; committed
profiles are **not edited**.

## Endpoints

- Complete when all 23 profiles carry exactly one row.
- Deliverables: verbatim show/tags captures + `results.json` here; the
  sweep table replaces ERRATA.md's "Not checked" section (dated
  amendment, old text struck not deleted); if any profile outside
  `tier-enthusiast/` is AFFECTED, an ERRATA pointer is filed in that
  directory (reachability rule); executable tests pin every AFFECTED
  row against the committed profile file itself, mutation-checked
  under the pyc discipline.
- No committed profile JSON is modified. No model is loaded onto the
  GPU (`/api/show` and `/api/tags` are metadata-only).
- Kill criterion: if the daemon is unreachable the sweep does not
  guess from memory or model cards — unreached rows ship as
  UNDETERMINED.
