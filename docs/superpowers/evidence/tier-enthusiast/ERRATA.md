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

### Not checked ~~(superseded — see the sweep below)~~

Every other profile in this directory was written by the same 0.5.0 code
path and may carry the same overstatement. **Only the two above were
re-measured.** A profile is affected exactly when its architecture states
an `attention.key_length` that differs from `embedding_length ÷
attention.head_count`; dense architectures where those agree are
unaffected. ~~Re-profiling the directory under 0.7.0 would settle it and
has not been done.~~ *Settled 2026-08-17 by the E1 sweep below; original
text left as filed.*

### The sweep (2026-08-17)

~~Every committed profile in the repository~~ *(scope bounded 2026-08-17 —
see below)* — this directory plus `live/`
and `live-run2/` — was classified under a pre-registered protocol
([`../e1-sweep/PROTOCOL.md`](../e1-sweep/PROTOCOL.md)): metadata
captured verbatim from the daemon (ollama 0.32.13), extraction and
window law run through probe 0.7.0 itself, and every verdict earned
through an identity gate (today's blob size equals the committed
`weights_bytes` AND replaying the derived `head_dim` reproduces the
committed geometry exactly). Full table:
[`../e1-sweep/results.json`](../e1-sweep/results.json).

Two more profiles in this directory are AFFECTED:

| profile | `head_dim` derived / stated | `kv_kib_per_token` as written → correct | `usable_window` as written → correct |
|---|---|---|---|
| [`gemma2-9b.json`](gemma2-9b.json) | 224 / **256** (`gemma2.attention.key_length`) | 294 → **336** | 8192 → **8192** (unchanged — `training_ctx` binds under both readings; the kv figure was wrong, the promise held) |
| [`mistral-nemo-latest.json`](mistral-nemo-latest.json) | 160 / **128** (`llama.attention.key_length`) | 200 → **160** | 32711 → **40889** (the profile **under**-promised by 20.0% of the true window) |

mistral-nemo is the sign E1's headline hides: the stated `key_length`
can also be *smaller* than the derivation, in which case every kv
number was too large and the window too conservative. Wrong either
way; over-promise is just the dangerous direction.

The remaining fifteen profiles in this directory settle clean:
`qwen3-14b`, `qwen3-8b`, and the Hermes-4-14B profile are UNAFFECTED
(stated `key_length` equals the derivation); the rest are
UNAFFECTED-BY-CONSTRUCTION (their metadata states no `key_length`, so
0.7.0's fallback reproduces the committed numbers — unchanged under
0.7.0, which is not the same claim as hardware-verified).
[`gemma-4-12b-it-qat-q4_0-latest.json`](gemma-4-12b-it-qat-q4_0-latest.json)
committed `geometry: null`, so it has no kv number to correct — worth
recording anyway: its metadata states `key_length` 512 against a
derivation of 240, the largest gap the sweep found (2.1×).

The `live/` and `live-run2/` codegemma profiles are also AFFECTED
(336 → 448 KiB/token); their errata are filed beside them
([`../live/ERRATA.md`](../live/ERRATA.md),
[`../live-run2/ERRATA.md`](../live-run2/ERRATA.md)).

### Scope bound (amended 2026-08-17)

The sweep sentence above read *"every committed profile in the
repository"*. That was true the day it was filed and stopped being true
the moment a new profile landed — so the claim is bounded here rather
than left to rot. The sweep's 23 rows are every committed profile **in
E1's blast radius**.

E1 was **fixed in probe 0.7.0**. A profile written by 0.7.0 or later
reads the stated `attention.key_length` instead of deriving it, so it
cannot carry this defect, and it joins the tree **unswept by
construction rather than by oversight**. The first such profile is
[`../tier-enthusiast-2026-08/qwen2.5-coder-1.5b-instruct-q8_0.json`](../tier-enthusiast-2026-08/qwen2.5-coder-1.5b-instruct-q8_0.json)
(probe 0.9.0), which brings the repository to 24 profiles, 23 swept.

The bound is **enforced, not asserted**: `tests/test_e1_sweep.py`
recomputes the swept set from the tree by `probe_version` and fails if a
profile written *before* 0.7.0 is ever added without re-running the
sweep. A companion test requires every unswept profile to justify itself
with `probe_version >= 0.7.0`, so a profile with an older version — or
none at all — breaks one test or the other rather than slipping between
them. Original sweep text left as filed;
[`../e1-sweep/PROTOCOL.md`](../e1-sweep/PROTOCOL.md) is pre-registered
and stands untouched.

### What is NOT wrong

Nothing outside `geometry`. The codec, envelope, ceiling, loop, speed and
long-output measurements in these profiles were not computed from
`head_dim` and stand as written. `expert_count` / `expert_used_count` are
absent from these profiles because schema v4 had no such fields — that is
a missing column, not a wrong value.

---

## E2 — hybrid layer geometry: `qwen3-8-27b.json` is wrong a second time

**Filed** 2026-08-27 · **Affects** [`qwen3-8-27b.json`](qwen3-8-27b.json)
in this directory · **Filed in full** beside the newer profile of the
same model:
[`../tier-enthusiast-2026-08/ERRATA.md`](../tier-enthusiast-2026-08/ERRATA.md)

E1 above corrects this profile's `kv_kib_per_token` from 216 to **260**
and its window from 4922 to **4096**, and that correction is right about
`head_dim` and still wrong about the layer count. `qwen3.8:27b` is a
**hybrid**: its metadata states `qwen35.full_attention_interval: 4` and
`qwen35.nextn_predict_layers: 1`, so 16 of its 65 blocks own a kv cache,
not 65 — a further **4.0625×** over-charge — and its 48 recurrent layers
hold 156,893,184 bytes of per-context state that no figure here charges
at all.

| profile | `kv_kib_per_token` | `usable_window` |
|---|---|---|
| as committed (v1.4, derived `head_dim`) | 216 | 4922 |
| after E1 (stated `key_length`, all blocks charged) | 260 | 4096 |
| conforming (E1 + R3/R4/R6) | **64** | **14246** |

Derived under this profile's own recorded `vram_free_mib` (1552) and
residency, not re-measured on hardware. E1's table and percentages are
left exactly as filed: they are correct statements about the head_dim
defect, which is the thing they were computed to measure.

---

## E3 — deepseek2 MLA: kv was R2 arithmetic, not an observed allocation

**Filed** 2026-08-28 · **Affects**
[`deepseek-coder-v2-16b-lite-instruct-q5_K_M.json`](deepseek-coder-v2-16b-lite-instruct-q5_K_M.json)
in this directory and, in the neighbouring directory,
[`../tier-enthusiast-2026-08/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json`](../tier-enthusiast-2026-08/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json)
· **Fixed in** branch `mla-kv-rule`, unreleased (SPEC.md rule R9,
`src/assay/geometry.py`'s R9 branch) · **Evidence**
[`../mla-kv-2026-08-27/`](../mla-kv-2026-08-27/) — protocol `c379c9f`
(pre-registered before any capture), captures `cf6222e`, verdict
`d25af18` · **Contract** gguf-geometry `SPEC.md` rule R9, whose vectors
are vendored at `../../../../tests/data/gguf_geometry_v3/` (v3, from
that repo's master `84f042b`; deepseek re-pinned, the other ten vectors
carried forward byte-identical from v2).

### What was published

`324 KiB/token`, `source: api_show`. This is E1's own correction above —
the row that changed this model's `head_dim` from a derived 128 to the
stated `deepseek2.attention.key_length` 192 — carried forward
unquestioned into the tools-anchor replay
(`../tools-anchor/results.json`) and into a fresh hardware run at probe
0.9.0
([`../tier-enthusiast-2026-08/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json`](../tier-enthusiast-2026-08/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json),
`assay_profile_version` 8), which published `geometry.kv_kib_per_token:
324` carrying the same **hardware-verified** framing every other figure
in that directory earns from having been measured on a live daemon. That
framing was wrong about what kind of number 324 was: it is `2 x
attention_layers x kv_head_count x head_dim x bytes/element` — R2's
arithmetic, run against a metadata field the daemon states — and
deepseek2 is an MLA architecture whose K and V caches are stated at
**different** widths (`deepseek2.attention.key_length` 192 alongside
`deepseek2.attention.value_length` 128, both present in the same show
capture E1 already read half of). "Hardware-verified" described the
*daemon connection*, not the *kv number*: nothing in the v1.4-through-0.9.0
lineage ever read a KV-buffer allocation off the runtime for this model.
It was R1/R2 metadata arithmetic that happened to run against a real
daemon, republished at a newer probe version without becoming a
different kind of claim.

### What the measurement showed

`docs/superpowers/evidence/mla-kv-2026-08-27/` reads ollama 0.32.13's
own KV-buffer log lines under the llama runner (non-FA lens; this box)
at three context points and finds **276,480 bytes/token (270
KiB/token)**, exact and identical at 2048, 4096 and 8192 — the log's own
K/V split (165,888 + 110,592 B/token) independently reproduces both
stated widths digit for digit
(`27 x 16 x 192 x 2 = 165,888`, `27 x 16 x 128 x 2 = 110,592`). Four
candidates were pre-registered before any capture
(`../mla-kv-2026-08-27/PROTOCOL.md`, committed `c379c9f`): `H-a` 331,776
(the dense 2x-head_dim guess this repository's v1/v2 vector pinned),
`H-b` 276,480 (separate K/V widths), `H-c` 31,104 and `H-c'` 62,208 (two
MLA-latent readings). Exactly one candidate landed in the pre-registered
±5% band — **H-b** — deviation from the measured figure **0.000%**. Full
arithmetic and verbatim log lines:
[`../mla-kv-2026-08-27/VERDICT.md`](../mla-kv-2026-08-27/VERDICT.md),
[`READING.md`](../mla-kv-2026-08-27/READING.md).

### Direction — the opposite sign from E1

E1's dominant direction was window-over-promise: a `head_dim` read too
small made every kv number too small and every window computed from it
too large. E3 runs the other way. 324 KiB/token was an **over-charge**
against the true 270, so every `usable_window` computed from it was
**under-promised** — the daemon can serve more context than the
published figure said, on both affected profiles:

| profile | `kv_kib_per_token` published | `usable_window` published (own conditions) | conforming `kv_kib_per_token` | conforming `usable_window` (same conditions) |
|---|---|---|---|---|
| [`deepseek-coder-v2-16b-lite-instruct-q5_K_M.json`](deepseek-coder-v2-16b-lite-instruct-q5_K_M.json) (this dir, v1.4 committed 216; E1-corrected 324 at `vram_free_mib` 2219, per `../tools-anchor/results.json`'s `v16_recomputed_under_v14_conditions`) | 324 | 5394 | **270** | **6473** |
| [`../tier-enthusiast-2026-08/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json`](../tier-enthusiast-2026-08/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json) (probe 0.9.0, published as measured, `vram_free_mib` 3414) | 324 | 9171 | **270** | **11006** |

Both conforming columns are derived under each profile's own recorded
`vram_free_mib` and residency, through today's extractor
(`OllamaNative.model_info()` -> `geometry.py`'s R9 branch) — not
re-measured a second time on hardware; the 276,480 B/token figure itself
is the one hardware measurement this erratum rests on, and every window
above is arithmetic over it. `docs/superpowers/evidence/e1-sweep/results.json`'s
deepseek row carries the same re-pin (`corrected.kv_kib_per_token: 270`,
`corrected.usable_window: 6473`, `window_shortfall_pct_of_committed_promise:
20.0`, down from 33.3 — the shortfall itself shrank, because less of
E1's original head_dim promise turns out to have been wrong).

### What is NOT wrong

`deepseek-r1-14b.json`, the other deepseek-family profile in both
directories, is unaffected: its architecture states no
`attention.value_length` distinct from its `key_length`, so R9 never
fires for it and its kv figure (192 KiB/token) is unchanged. Nothing
outside `geometry.kv_kib_per_token` and the `usable_window` it feeds
moves for either affected profile — `expert_count`/`expert_used_count`
(64/6) are unaffected, kv is expert-invariant by R2, unchanged by R9.

### Corrected value

| quantity | published | corrected |
|---|---|---|
| `kv_bytes_per_token` | 331,776 | **276,480** |
| `kv_kib_per_token` | 324 | **270** |

Both affected profiles stand exactly as committed — evidence is not
rewritten to suit a later fix, per this file's own header. This erratum,
`docs/superpowers/evidence/mla-kv-2026-08-27/`, and SPEC.md's R9 are the
correction; `tests/test_geometry_conformance.py`'s vendored v3 vector and
`tests/test_e1_sweep.py` / `tests/test_geometry.py`'s replay tests pin
the corrected figures as checked arithmetic rather than prose.
