# Errata — tier-enthusiast-2026-08 profiles

Corrections that apply to profiles committed in this directory. The
profiles themselves are **left exactly as measured**: evidence is not
rewritten to suit a later fix. Read this file alongside them.

---

## E2 — hybrid layer geometry: every block charged as an attention layer

**Filed** 2026-08-27 · **Affects** `qwen3.8-27b.json` (probe 0.9.0,
`assay_profile_version` 8) and, in the neighbouring directory,
[`../tier-enthusiast/qwen3-8-27b.json`](../tier-enthusiast/qwen3-8-27b.json)
(probe 0.5.0, v4) · **Fixed on** branch `geometry-conformance`, unreleased
(MERGE HOLD) · **Upstream record**
`~/workspace/gguf-geometry/docs/upstream-errata/2026-08-27-assay-qwen3.8-27b-hybrid-overcharge.md`,
filed against assay by the gguf-geometry contract work · **Contract**
gguf-geometry `SPEC.md` rules R3, R4 and R6, whose vectors are vendored
at `tests/data/gguf_geometry_v1/`

Through probe 0.9.0, `kv_bytes_per_token` charged
`block_count` attention layers — every block the file states. That is
correct for a dense architecture and wrong for a **hybrid** one, which
interleaves attention and recurrent layers and states its period in the
metadata. `qwen3.8:27b` (architecture `qwen35`) is the one such model in
this repository's committed evidence, and its own metadata says so
(verbatim capture:
[`../e1-sweep/show-qwen3.8-27b.json`](../e1-sweep/show-qwen3.8-27b.json)):

```
qwen35.block_count               65
qwen35.nextn_predict_layers      1     <- R6: an MTP layer, not a serving layer
qwen35.full_attention_interval   4     <- R3: 1 layer in 4 owns a kv cache
qwen35.ssm.conv_kernel           4     <- R4: the other 48 layers are recurrent
qwen35.ssm.group_count           16
qwen35.ssm.inner_size            6144
qwen35.ssm.state_size            128
qwen35.attention.head_count_kv   4
qwen35.attention.key_length      256
```

so `serving_block_count = 65 − 1 = 64`, `attention_layer_count = 64 ÷ 4
= 16`, and `2 × 16 × 4 × 256 × 2 = 65,536` bytes = **64 KiB/token**. The
published 260 KiB/token is `2 × 65 × 4 × 256 × 2`: the raw block count,
a **4.0625×** over-charge. This is the same defect class bloomery
measured and fixed in its turn 5 (all 40 blocks of Qwen3.6-35B-A3B
charged where 10 are attention layers, 4.00×), which is why the contract
has a rule against it.

A second term is missing rather than wrong: the 48 recurrent layers hold
`156,893,184` bytes (149.625 MiB) of per-context state (R4), and no
published figure here charges anything for it.

### Affected profiles

| profile | `kv_kib_per_token` as written → conforming | `usable_window` as written → conforming | `recurrent_state_bytes` as written → conforming |
|---|---|---|---|
| [`qwen3.8-27b.json`](qwen3.8-27b.json) | 260 → **64** | 3749 → **12838** | absent → **156893184** |
| [`../tier-enthusiast/qwen3-8-27b.json`](../tier-enthusiast/qwen3-8-27b.json) | 216 → **64** (E1 had already corrected 216 → 260; this correction is on top of that) | 4922 → **14246** | absent → **156893184** |

Each conforming window is computed under **that profile's own recorded
`vram_free_mib`** (1464 and 1552) and its own residency state, so the
only variables changed are the layer count and the new recurrent charge.

### Direction, and its honest status

This one **under**-promises: the corrected window is ~3.4× the published
one, because the over-charged cache term dominates the newly charged
recurrent term. Wrong either way — E1's mistral-nemo row was the same
shape — but this is not a direction that has cost anyone context they
were promised.

The conforming figures above are **derived, not hardware-verified**.
They come from replaying the committed `/api/show` capture through the
corrected implementation, which is deterministic given that capture and
that VRAM reading; no live run of this model has been made under the
corrected code. A re-measurement would supersede them, and until one
happens the upstream record cited above keeps `qwen3.8-27b` **withheld**
from the gguf-geometry v1 vector set rather than pinning a number nobody
measured.

### Enforced, not asserted

Two tests recompute this erratum from the committed bytes rather than
trusting the table:
`tests/test_e1_sweep.py::test_the_sweeps_hybrid_row_carries_a_second_defect`
and
`tests/test_geometry.py::test_the_anchors_hybrid_row_carries_the_overcharge`.
Both drive the verbatim capture through the shipped extractor, assert
the 4.0625 ratio and the recurrent term, and would fail if either the
metadata or the rule moved. The rules themselves are pinned against the
frozen gguf-geometry vectors in
`tests/test_geometry_conformance.py`.

### What is NOT wrong

Nothing outside `geometry`, and nothing in any other profile in this
directory. The other fourteen models here state no
`full_attention_interval` — checked against the same verbatim captures,
not assumed — so their kv figures are unchanged by this fix, and every
non-geometry family (codecs, envelope, ceiling, loop, speed,
long_output, tools, parallel) was never computed from a layer count.

The published matrix (`docs/matrix/`) is built from this directory's
rows and **still shows 260 KiB/token and a 3749-token window for this
model**. The profile is not rewritten, so neither is the page's number.
What changed on 2026-08-27 is that the page now *says so*: the matrix
build reads a machine-readable sidecar
([`errata/matrix-errata.json`](errata/matrix-errata.json)) and renders a
flag beside each figure this erratum supersedes, linking back to this
file, with a note stating that the value below it is left exactly as
measured.

The sidecar is the machine-readable half and this file is the human
one; the build never parses the markdown. Prose is written for a reader
and changes shape whenever someone improves it, and a build that scraped
it would stop flagging things the day a heading moved.

The page annotates and never substitutes. Writing 64 into the matrix
would be the same edit this file exists to avoid, performed one layer
out — where a reader comparing the page against the profile beside it
could not see that it had happened.

### Decision provenance

Filing this rather than editing the two profiles was ruled **before** the
fix work began, not chosen inside it. The ruling is recorded in the
gguf-geometry SDD ledger,
`~/workspace/gguf-geometry/.superpowers/sdd/2026-08-27-gguf-geometry/progress.md`,
under "Hold release + Tasks 8-10 (2026-08-27)":

> Ruling (pre-ruled Task 9 STOP clause): committed hybrid-affected assay
> artifacts EXIST (qwen3.8-27b: e1-sweep results + tier-enthusiast
> ERRATA line 36 + tier-enthusiast-2026-08 profile + matrix). Per the
> E1-erratum precedent, Task 9 appends an ERRATA.md row (profiles left
> as committed) citing gguf-geometry's
> docs/upstream-errata/2026-08-27-assay-qwen3.8-27b-hybrid-overcharge.md;
> NO matrix rebuild (Pages publish = outward-facing, Brice's);
> branch-only.

That ledger is deliberately untracked in its own repository (SDD
workspaces stay local in public repos), which is why this is a quoted
citation by path rather than a link — the same shape of citation that
repository's `.gitignore` carries for the same reason.

What the ruling settles: leave the profiles as measured, file the
correction beside them, do not rebuild the matrix, keep the work on a
branch. What it does **not** settle, and what no ruling here can: the
publication decisions. Rebuilding the matrix and pushing anything remain
Brice's, and the same ledger's approval record of 2026-08-27 covers
merging this branch to assay master — approving a merge is not
approving a rebuild, and this file states the gap rather than closing
it quietly.

**Update, 2026-08-27, after the merge to master.** The rebuild was then
directed as its own decision, which is the gap above being closed rather
than the ruling being reinterpreted — the quoted ledger text stands as
written, and it was written when no rebuild had been asked for. The
matrix was regenerated with the errata-aware build described in "What is
NOT wrong": every published figure is unchanged (a rebuild does not
correct a profile, and this one was byte-identical to its predecessor
outside the flags it added), and the two superseded figures now carry
one. The Pages publish — pushing master — is not part of that change and
remains Brice's.
