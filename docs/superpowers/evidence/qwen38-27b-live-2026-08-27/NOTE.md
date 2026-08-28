# qwen3.8:27b — live run under the corrected geometry, 2026-08-27

The model this repository's E2 erratum
([`../tier-enthusiast-2026-08/ERRATA.md`](../tier-enthusiast-2026-08/ERRATA.md))
corrects was absent from the box's Ollama store when the erratum was
filed — its conforming figures were derived by replaying the committed
`/api/show` capture, and the erratum says so ("derived, not
hardware-verified ... no live run of this model has been made under the
corrected code"). This directory is that live run.

## Provenance

- **Re-pulled** 2026-08-27: `ollama pull qwen3.8:27b` → id `22130167c4c2`,
  17,741,872,154 bytes reported by `/api/tags`
  ([`tags-qwen3.8-27b.json`](tags-qwen3.8-27b.json), captured in the same
  breath as the show).
- **Content addressing spot-verified on this blob**: the manifest's
  model-layer digest and `sha256sum` of the blob file agree —
  `f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d`
  (16,810,714,464 bytes). The digest IS the raw GGUF's sha256, so a
  vector can anchor on it self-containedly.
- **Instrument**: assay at `24420f2` (master, the R3/R4/R6 hybrid fix
  shipped), run as
  `assay geometry http://127.0.0.1:11434 --model qwen3.8:27b`, output
  verbatim in [`geometry-qwen3.8-27b.json`](geometry-qwen3.8-27b.json).
- **Metadata**: [`show-qwen3.8-27b.json`](show-qwen3.8-27b.json) is the
  daemon's `/api/show` reply, captured verbatim, same session.

## What the live run establishes

The live metadata states the same hybrid facts the erratum's replay
used — `qwen35.block_count 65`, `nextn_predict_layers 1`,
`full_attention_interval 4`, the `ssm.*` family, `head_count_kv 4`,
`key_length 256` — and the corrected implementation, reading them live,
produces exactly the erratum's conforming per-token and per-context
terms:

| term | live value | erratum's derived value |
|---|---|---|
| `kv_kib_per_token` | **64** | 64 |
| `attention_layer_count` | **16** | 16 |
| `serving_block_count` | **64** | 64 (65 − 1 nextn, R6) |
| `recurrent_state_bytes` | **156,893,184** | 156,893,184 |

## What it does not establish

`usable_window` came back **0, `limited_by: "vram"`** — today's box had
15,162 MiB free against 17.7 GB of weights. That is a condition of this
run, not a fact about the model: the erratum's conforming windows
(12838 / 14246) are arithmetic under **the profiles' own recorded
conditions** (`vram_free_mib` 1464 / 1552 with their residency states),
which no later run can re-occupy. The window law is exactly that
arithmetic, so the live-grounded terms above are the measurable content;
a ceiling-ladder verification of a 12k-token window is not possible on
this box while the weights exceed free VRAM.

## What this closes

The erratum's "live run under the corrected code still owed" clause.
With the terms live-confirmed and the blob sha-anchored, the withheld
`qwen3.8:27b` vector is eligible for the gguf-geometry **v2** set (the
upstream repo's own decision and record — see its
`docs/upstream-errata/2026-08-27-assay-qwen3.8-27b-hybrid-overcharge.md`).
