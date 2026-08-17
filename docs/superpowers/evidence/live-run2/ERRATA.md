# Errata — live-run2/ (v1 validation, run 2)

Profiles are **left exactly as measured**. Read alongside
[`../tier-enthusiast/ERRATA.md`](../tier-enthusiast/ERRATA.md) (erratum
E1) and the sweep that settled this directory
([`../e1-sweep/PROTOCOL.md`](../e1-sweep/PROTOCOL.md),
[`../e1-sweep/results.json`](../e1-sweep/results.json)).

## E1 — codegemma kv understated

[`codegemma-7b-instruct-q8_0-quick.json`](codegemma-7b-instruct-q8_0-quick.json):
same correction as run 1 — stated `attention.key_length` 256 vs derived
192, committed `kv_kib_per_token` 336 → correct **448**,
`usable_window` 0 → **0**. See
[`../live/ERRATA.md`](../live/ERRATA.md).

## Sweep note — qwen geometry was a loaded-GPU reading (no correction)

[`qwen2.5-coder-7b-instruct-q8_0-quick.json`](qwen2.5-coder-7b-instruct-q8_0-quick.json)
failed the sweep's version-keyed replay gate (probe 0.1.0 →
pre-load replay) and is filed E1-INCONSISTENT per protocol, with the
investigation recorded in `results.json`: its geometry reproduces
**exactly** under `loaded=True` — the model was still resident from
run 1 when run 2 read VRAM (4233 MiB free vs run 1's 13684). Its
metadata states no `attention.key_length`, so the kv figure is
identical under either head_dim source: **the committed numbers carry
no E1 error**; only the sweep's replay condition mis-modeled this
row's residency. Nothing to correct — recorded so the gate failure is
never misread as a wrong profile.
