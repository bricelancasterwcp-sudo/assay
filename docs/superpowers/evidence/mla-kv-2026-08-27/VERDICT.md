# Verdict — deepseek2 KV bytes/token, mla-kv-2026-08-27

## Verdict

```
H-b
```

(One of `H-a | H-b | H-c | H-c' | no-match`, per `PROTOCOL.md`.)

**H-b = separate K and V widths**, formula `27·16·(192+128)·2`,
**276,480 bytes/token**.

Derived mechanically from the pre-registered decision rule in
`PROTOCOL.md` (committed c379c9f, BEFORE the captures in cf6222e).
Full extraction, verbatim quoted lines and arithmetic: `READING.md`.

---

## Instrument 1 (primary, decisive) — figures per point

| point | KV buffer lines (summed) | log-reported n_ctx | bytes/token |
|-------|--------------------------|--------------------|-------------|
| ctx 2048 | `CUDA0 KV buffer size = 540.00 MiB` (1 line) | 2048 | 540 × 1,048,576 ÷ 2048 = **276,480** |
| ctx 4096 | `CUDA0 KV buffer size = 1080.00 MiB` (1 line) | 4096 | 1080 × 1,048,576 ÷ 4096 = **276,480** |
| ctx 8192 | `CUDA0 KV buffer size = 2160.00 MiB` (1 line) | 8192 | 2160 × 1,048,576 ÷ 8192 = **276,480** |

The cache did not split across backends — exactly one KV buffer line per
load (`grep -c "KV buffer size"` = 1 in each raw log), so each sum is
that single line. Denominators are the log-reported n_ctx, confirmed
three ways per point (`llama_context: n_ctx`, the `N cells` field of the
`llama_kv_cache: size` line, and `new slot, n_ctx`), with `n_slots = 1`
and `1/1 seqs` — no parallel-slot inflation.

**All three points agree exactly: 276,480 bytes/token.**

Structural corroboration from the same lines (not part of the rule): the
K/V split is 165,888 + 110,592 B/token at every point, which reproduces
`27·16·192·2` and `27·16·128·2` — H-b's two halves separately, digit for
digit.

---

## Band arithmetic (±5%, as committed)

| Id | Candidate | Band low (×0.95) | Band high (×1.05) | Measured | In band? |
|----|-----------|------------------|-------------------|----------|----------|
| H-a | 331,776 | 315,187.2 | 348,364.8 | 276,480 | no |
| **H-b** | **276,480** | **262,656.0** | **290,304.0** | **276,480** | **YES** |
| H-c | 31,104 | 29,548.8 | 32,659.2 | 276,480 | no |
| H-c' | 62,208 | 59,097.6 | 65,318.4 | 276,480 | no |

Exactly one candidate matched → that candidate decides. Deviation from
H-b's value: **0.000%** (exact). Bands were not widened, no candidate was
added, no capture was re-run.

---

## Instrument 2 (corroboration only, cannot decide)

Validity: all three `ps-ctx*.txt` report `PROCESSOR = 100% GPU` with the
matching CONTEXT — verified from the files. **`dropped: []`** (empty).

Slope from the `vram-ctx*.txt` before/after deltas
(12,118 / 12,742 / 13,974 MiB above a common 739 MiB baseline):

| segment | delta | tokens | bytes/token | ratio to Instrument 1 |
|---------|-------|--------|-------------|------------------------|
| 2048 → 4096 | 624 MiB | 2048 | 319,488 | 1.156 |
| 4096 → 8192 | 1,232 MiB | 4096 | 315,392 | 1.141 |
| 2048 → 8192 | 1,856 MiB | 6144 | 316,757.33 | 1.146 |

**Consistency: CONSISTENT with Instrument 1.** Same order of magnitude
(10^5, within a factor 1.16), and slope ≥ Instrument 1's 276,480 on every
segment — the over-read direction the protocol pre-registered. The excess
is accounted for by the stated confound: compute buffers grow 116.28 →
364.28 MiB across the range, and KV delta (1,620 MiB) + compute delta
(248 MiB) = 1,868 MiB vs the measured 1,856 MiB, a 12 MiB (0.6%)
residual. Instrument 2 decided nothing; it corroborates only.

---

## Ambient lens

This figure is a property of **this runtime on this box at this time**,
not of the GGUF.

| Facet | Value | Source |
|-------|-------|--------|
| Runtime | ollama **0.32.13** (systemd unit) | `preflight.txt` |
| Engine / runner | **llama runner** — ollama spawned `/usr/local/lib/ollama/llama-server` (`source=server.go:109 msg="using llama-server for model"`, all 3 points). This is ollama's own llama engine path, **not** the protocol's standalone-llama-server contingency lens. | `raw-log-ctx*.txt` |
| Box | brice-X870E-Taichi, **NVIDIA GeForce RTX 5080** (15,839 MiB, 14,832 MiB free), PCI 0000:01:00.0 | `raw-log-ctx*.txt` |
| KV cache type | **f16** — `K (f16)` / `V (f16)` on the kv_cache size lines; `OLLAMA_KV_CACHE_TYPE` unset | `raw-log-ctx*.txt`, `env.txt` |
| Systemd env | `OLLAMA_HOST=0.0.0.0:11434`, `OLLAMA_MODELS=/mnt/extra/ollama-models`; **no** `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_FLASH_ATTENTION`, or `OLLAMA_NUM_PARALLEL` set | `env.txt` |
| Model | `deepseek-coder-v2:16b-lite-instruct-q5_K_M`, id `6065d4880bf9` | `preflight.txt`, `ps-ctx*.txt` |
| Model digest | `sha256:bc286970a24072cf23a4c905f28adb9f6a28c71743b07790185275a86dc72406` — digest OK | `preflight.txt` |
| Arch (log-reported) | deepseek2; 27 layers, 16 heads, key_length 192, value_length 128, kv_lora_rank 512, n_embd_k_gqa 3072, n_embd_v_gqa 2048 | `raw-log-ctx2048.txt` |
| Slots / batch | `-np 1`, `n_slots = 1`, `kv_unified = false`, `-b 512 -ub 512` | `raw-log-ctx*.txt` |
| Flash attention | `flash_attn = auto`, resolved to **FA not on CUDA** (`resolve_fused_ops: ... Flash Attention is assigned to device CPU`), uniform across all 3 points | `raw-log-ctx*.txt` |
| Offload | 28/28 layers to GPU, 100% GPU residency at all 3 points | `raw-log-ctx*.txt`, `ps-ctx*.txt` |
| Capture dates | loads 2026-08-27 23:56:36–23:56:52 −05:00 (CDT); env stamp 2026-08-28T04:54:21+00:00; reading/verdict written 2026-08-28 | `raw-log-ctx*.txt`, `env.txt` |
| Protocol commit | c379c9f (pre-registration) — precedes capture commit cf6222e | git history |

---

## Notes carried into the verdict

- **`load_mode = none` at ctx2048 only.** ollama disabled mmap for that
  load under host memory pressure (`sched.go:1215`); 4096 and 8192 used
  mmap. This concerns the **weights** path, not KV cache sizing — stated
  explicitly rather than ignored, and evidenced: the CUDA0 *model* buffer
  is 11,160.99 MiB at all three points regardless, and ctx2048's KV
  bytes/token (276,480) and K/V split (165,888 / 110,592) are identical
  to the two mmap points. Had mmap perturbed KV allocation, ctx2048 would
  be the outlier; it is not. No point dropped or adjusted. See
  `READING.md` §3 for the full quoted evidence and the honest limit of
  the argument.
- **The blob is MLA, this runtime is not caching the latent.** The log
  reports `kv_lora_rank = 512` / `n_lora_kv = 512`, yet materialises full
  per-head K and V. That is the substance of the H-b result and the
  reason H-c / H-c' are far out of band — recorded as an observation, not
  used to adjust the rule.
- **Contingency did not fire.** Usable KV-buffer lines were present at
  all three points under the pre-registered ollama lens, so the
  standalone-llama-server fallback (which would have needed its own
  ruling) was never reached.
- **Scope.** Per `PROTOCOL.md`, whether a SPEC rule may be written from
  this figure, and in what terms, is the next ruling — this document
  reports the measurement and the pre-registered verdict only.
