# Errata — live/ (v1 validation, run 1)

Profiles are **left exactly as measured**. Read alongside
[`../tier-enthusiast/ERRATA.md`](../tier-enthusiast/ERRATA.md) (erratum
E1) and the sweep that settled this directory
([`../e1-sweep/PROTOCOL.md`](../e1-sweep/PROTOCOL.md),
[`../e1-sweep/results.json`](../e1-sweep/results.json)).

## E1 — codegemma kv understated

[`codegemma-7b-instruct-q8_0-quick.json`](codegemma-7b-instruct-q8_0-quick.json):
gemma's metadata states `attention.key_length` **256**; probe 0.1.0
derived 192. Committed `kv_kib_per_token` 336 → correct **448**.
`usable_window` 0 → **0** (the weights already exceeded the recorded
free VRAM; the window was honest for the wrong reason at the wrong
per-token price). The validation write-up's "6× qwen" ratio is really
**8×** — amended in place, dated, in
[`../2026-08-12-live-validation.md`](../2026-08-12-live-validation.md).

qwen2.5-coder and granite-code state no `attention.key_length`, so
their committed numbers are what 0.7.0 computes — unaffected by E1.
