# Pre-registered protocol — deepseek2 KV bytes/token under ollama 0.32.13

Written and committed BEFORE any capture. Subject: what this runtime
actually allocates per KV-cache token for
deepseek-coder-v2:16b-lite-instruct-q5_K_M (GGUF sha256 bc286970…2406).
Lens: ollama 0.32.13 (systemd unit), this box (RTX 5080 16 GiB), KV
cache type f16 (OLLAMA_KV_CACHE_TYPE unset), point-in-time. This is a
property of the runtime-on-this-box, not of the GGUF.

## Candidates

| Id | Model of the cache | Formula | B/token |
|----|--------------------|---------|---------|
| H-a | k-width for both K and V (current pin) | 2·27·16·192·2 | 331,776 |
| H-b | separate K and V widths | 27·16·(192+128)·2 | 276,480 |
| H-c | MLA latent, per-layer (no head factor) | 27·(512+64)·2 | 31,104 |
| H-c′ | MLA latent + transposed copy | 2·27·(512+64)·2 | 62,208 |

H-c′ is llama.cpp's historical MLA variant caching a transposed copy of
the latent; a derived implementation variant, not a post-hoc hedge.

## Decision rule

Instrument 1's bytes/token matching exactly ONE candidate within ±5%
decides. Bands are disjoint (closest pair 20% apart). No match → no
rule is written; the figure is recorded and the SPEC gap stays open.
The point estimate decides: no re-run, no extension, no added candidate
after a number is seen. A failed load / missing log line / OOM is
infrastructure, recorded as such, never data.

## Instrument 1 (primary): server-log KV buffer lines

Load with explicit options.num_ctx; slice journalctl for the load;
quote VERBATIM every KV-cache buffer/size line (all backends — if the
cache splits across CUDA0/CPU, the reading is the SUM of the lines).
bytes/token = summed KV bytes ÷ the n_ctx THE LOG REPORTS (not the
requested one; parallel slots inflate effective n_ctx). Every "because"
in the reading is a quoted line.

## Instrument 2 (corroboration only): VRAM slope

num_ctx ∈ {2048, 4096, 8192}; nvidia-smi before/after each load.
Validity per point: ollama ps reports 100% GPU, else the point is named
in dropped. Stated confound: compute buffers also grow with ctx, so the
slope over-reads KV. Pre-registered consequence: instrument 2 CANNOT
decide alone; it must only be consistent with instrument 1 (same order
of magnitude, slope ≥ instrument 1's figure).

## Contingency

If ollama 0.32.13's logs surface no usable KV-buffer lines, fallback is
the same blob under a standalone llama-server — a DIFFERENT NAMED LENS;
whether a rule may be written from it is Brice's ruling, not a swap.

## Ambient record (filled at capture, committed with the evidence)

ollama version; systemd env (OLLAMA_KV_CACHE_TYPE, OLLAMA_FLASH_ATTENTION,
OLLAMA_NUM_PARALLEL); runner engine as logged; GPU + free VRAM before
each point; ollama ps residency per point; timestamps.
