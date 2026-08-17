# Errata — tier-enthusiast profiles

Corrections that apply to profiles committed in this directory. The
profiles themselves are **left exactly as measured**: evidence is not
rewritten to suit a later fix. Read this file alongside them.

---

## E1 — `head_dim` derived instead of read: kv and window overstated

**Filed** 2026-08-16 · **Affects** `assay_profile_version` 4 profiles
written by probe 0.5.0 · **Fixed in** probe 0.7.0 (v1.6) ·
**Evidence** [`../tools-anchor/README.md`](../tools-anchor/README.md)

Through v1.4, `head_dim` was DERIVED as
`embedding_length ÷ attention.head_count`. That derivation assumes
attention is sized as embedding width over heads, which is false for any
architecture that sizes attention independently. v1.6 prefers the model
file's **stated** `attention.key_length` and keeps the derivation only as
a fallback for metadata that omits it.

Because `kv_bytes_per_token = 2 × block_count × kv_head_count × head_dim
× bytes/element`, a `head_dim` read too small makes **every** kv number
too small and the planned window correspondingly too large.

### Confirmed affected profiles

Both were re-measured against the live daemon (ollama 0.32.13,
2026-08-16). Feeding the derived `head_dim` back through the current
window law reproduces each committed `usable_window` **exactly**, which
is what identifies the head_dim source as the only thing that changed.

| profile | `head_dim` as written / correct | `kv_kib_per_token` as written → correct | `usable_window` as written → correct |
|---|---|---|---|
| [`deepseek-coder-v2-16b-lite-instruct-q5_K_M.json`](deepseek-coder-v2-16b-lite-instruct-q5_K_M.json) | 128 (derived) / **192** (`deepseek2.attention.key_length`) | 216 → **324** | 8092 → **5394** |
| [`qwen3-8-27b.json`](qwen3-8-27b.json) | 213 (derived) / **256** (`qwen35.attention.key_length`) | 216 → **260** | 4922 → **4096** |

Both corrected windows are computed under **that profile's own recorded
`vram_free_mib`** (2219 and 1552) and its own residency state, so the
only variable changed is `head_dim`.

### The size of it, on two bases

These are the same fact stated two ways, and mixing the two is the easy
mistake — where the VRAM term binds, `usable_window` is inversely
proportional to kv bytes per token, so the second column below is
identically the third.

| profile | window shortfall of the promise<br>(v1.4−v1.6)÷v1.4 | excess over the true window<br>(v1.4−v1.6)÷v1.6 | kv excess per token<br>(v1.6−v1.4)÷v1.4 |
|---|---|---|---|
| deepseek-coder-v2:16b-lite-q5_K_M | **33.3%** | 50.0% | **50.0%** |
| qwen3.8:27b | **16.8%** | 20.2% | **20.2%** |

### Not checked

Every other profile in this directory was written by the same 0.5.0 code
path and may carry the same overstatement. **Only the two above were
re-measured.** A profile is affected exactly when its architecture states
an `attention.key_length` that differs from `embedding_length ÷
attention.head_count`; dense architectures where those agree are
unaffected. Re-profiling the directory under 0.7.0 would settle it and
has not been done.

### What is NOT wrong

Nothing outside `geometry`. The codec, envelope, ceiling, loop, speed and
long-output measurements in these profiles were not computed from
`head_dim` and stand as written. `expert_count` / `expert_used_count` are
absent from these profiles because schema v4 had no such fields — that is
a missing column, not a wrong value.
