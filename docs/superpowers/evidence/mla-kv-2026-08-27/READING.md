# Reading — deepseek2 KV bytes/token, mla-kv-2026-08-27

Applies the decision rule pre-registered in `PROTOCOL.md` (committed
c379c9f, BEFORE the captures in cf6222e) to the raw files captured in
Task 2. Every claim below is backed by a verbatim line quoted from a
named source file. No re-runs, no band widening, no added candidates.

Verdict is in `VERDICT.md`.

---

## 0. Runner identification

The protocol's Instrument 1 reads "server-log KV buffer lines" under
ollama 0.32.13. Which engine actually served the load matters, because
the protocol's contingency (§Contingency) treats a standalone
llama-server as a DIFFERENT NAMED LENS. Here the runner is llama-server
**invoked by ollama as its own subprocess** — that is ollama 0.32.13's
normal llama engine path, not the fallback lens. Quoted per point:

`raw-log-ctx2048.txt:2`
```
Aug 27 23:56:36 brice-X870E-Taichi ollama[1502099]: time=2026-08-27T23:56:36.676-05:00 level=INFO source=server.go:109 msg="using llama-server for model" model=/mnt/extra/ollama-models/blobs/sha256-bc286970a24072cf23a4c905f28adb9f6a28c71743b07790185275a86dc72406
```

`raw-log-ctx4096.txt:1`
```
Aug 27 23:56:43 brice-X870E-Taichi ollama[1502099]: time=2026-08-27T23:56:43.816-05:00 level=INFO source=server.go:109 msg="using llama-server for model" model=/mnt/extra/ollama-models/blobs/sha256-bc286970a24072cf23a4c905f28adb9f6a28c71743b07790185275a86dc72406
```

`raw-log-ctx8192.txt:1`
```
Aug 27 23:56:51 brice-X870E-Taichi ollama[1502099]: time=2026-08-27T23:56:51.185-05:00 level=INFO source=server.go:109 msg="using llama-server for model" model=/mnt/extra/ollama-models/blobs/sha256-bc286970a24072cf23a4c905f28adb9f6a28c71743b07790185275a86dc72406
```

All three lines come from `source=server.go:109` inside the
`ollama[1502099]` systemd unit — ollama chose and spawned the runner. The
spawn line names the binary and the flags ollama picked
(`raw-log-ctx2048.txt:3`, abbreviated only where marked `…`; the `-c`
value is the requested num_ctx, which is *not* the denominator — see §2):

```
Aug 27 23:56:36 brice-X870E-Taichi ollama[1502099]: time=2026-08-27T23:56:36.676-05:00 level=INFO source=llama_server.go:431 msg="starting llama-server" cmd="/usr/local/lib/ollama/llama-server --model /mnt/extra/…/sha256-bc286970… --port 45739 --host 127.0.0.1 --no-webui --offline -c 2048 -np 1 --log-verbosity 4 --no-log-prefix --no-log-timestamps --no-jinja --chat-template chatml --load-mode none --flash-attn auto -b 512 -ub 512"
```

**Contingency did NOT fire.** The protocol's contingency is reserved for
"if ollama 0.32.13's logs surface no usable KV-buffer lines". Usable KV
buffer lines are present at all three points (§1). This is the
pre-registered primary lens, read as pre-registered.

---

## 1. Instrument 1 (primary) — KV buffer lines, verbatim

### Buffer-line inventory (is the cache split across backends?)

The protocol requires the SUM of KV buffer lines if the cache splits
across backends. It does not split. Every `buffer size` line in each
log, with the systemd prefix stripped for readability (grep
`"buffer size"`, complete — nothing omitted):

`raw-log-ctx2048.txt`
```
load_tensors:        CUDA0 model buffer size = 11160.99 MiB
load_tensors:    CUDA_Host model buffer size =   137.50 MiB
llama_context:  CUDA_Host  output buffer size =     0.39 MiB
llama_kv_cache:      CUDA0 KV buffer size =   540.00 MiB
sched_reserve:      CUDA0 compute buffer size =   104.27 MiB
sched_reserve:  CUDA_Host compute buffer size =    12.01 MiB
```

`raw-log-ctx4096.txt`
```
load_tensors:   CPU_Mapped model buffer size =   137.50 MiB
load_tensors:        CUDA0 model buffer size = 11160.99 MiB
llama_context:  CUDA_Host  output buffer size =     0.39 MiB
llama_kv_cache:      CUDA0 KV buffer size =  1080.00 MiB
sched_reserve:      CUDA0 compute buffer size =   188.27 MiB
sched_reserve:  CUDA_Host compute buffer size =    16.01 MiB
```

`raw-log-ctx8192.txt`
```
load_tensors:   CPU_Mapped model buffer size =   137.50 MiB
load_tensors:        CUDA0 model buffer size = 11160.99 MiB
llama_context:  CUDA_Host  output buffer size =     0.39 MiB
llama_kv_cache:      CUDA0 KV buffer size =  2160.00 MiB
sched_reserve:      CUDA0 compute buffer size =   340.27 MiB
sched_reserve:  CUDA_Host compute buffer size =    24.01 MiB
```

`grep -c "KV buffer size"` returns **1** for each of the three files.
Exactly one KV buffer line per load, all on CUDA0. The sum over KV
buffer lines is therefore that single line's value at each point. The
other buffers (`model`, `output`, `compute`) are not KV cache and are
excluded — the protocol scopes Instrument 1 to KV-cache buffer/size
lines.

### Per-point verbatim KV block + denominator

**ctx 2048** — `raw-log-ctx2048.txt:170, 184–187, 208`
```
Aug 27 23:56:37 brice-X870E-Taichi ollama[1502099]: llama_context: n_ctx                 = 2048
Aug 27 23:56:37 brice-X870E-Taichi ollama[1502099]: llama_kv_cache:      CUDA0 KV buffer size =   540.00 MiB
Aug 27 23:56:37 brice-X870E-Taichi ollama[1502099]: llama_kv_cache: size =  540.00 MiB (  2048 cells,  27 layers,  1/1 seqs), K (f16):  324.00 MiB, V (f16):  216.00 MiB
Aug 27 23:56:37 brice-X870E-Taichi ollama[1502099]: llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 192
Aug 27 23:56:37 brice-X870E-Taichi ollama[1502099]: llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 128
Aug 27 23:56:37 brice-X870E-Taichi ollama[1502099]: slot   load_model: id  0 | task -1 | new slot, n_ctx = 2048
```

**ctx 4096** — `raw-log-ctx4096.txt:169, 183–186, 207`
```
Aug 27 23:56:44 brice-X870E-Taichi ollama[1502099]: llama_context: n_ctx                 = 4096
Aug 27 23:56:44 brice-X870E-Taichi ollama[1502099]: llama_kv_cache:      CUDA0 KV buffer size =  1080.00 MiB
Aug 27 23:56:44 brice-X870E-Taichi ollama[1502099]: llama_kv_cache: size = 1080.00 MiB (  4096 cells,  27 layers,  1/1 seqs), K (f16):  648.00 MiB, V (f16):  432.00 MiB
Aug 27 23:56:44 brice-X870E-Taichi ollama[1502099]: llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 192
Aug 27 23:56:44 brice-X870E-Taichi ollama[1502099]: llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 128
Aug 27 23:56:45 brice-X870E-Taichi ollama[1502099]: slot   load_model: id  0 | task -1 | new slot, n_ctx = 4096
```

**ctx 8192** — `raw-log-ctx8192.txt:169, 183–186, 207`
```
Aug 27 23:56:52 brice-X870E-Taichi ollama[1502099]: llama_context: n_ctx                 = 8192
Aug 27 23:56:52 brice-X870E-Taichi ollama[1502099]: llama_kv_cache:      CUDA0 KV buffer size =  2160.00 MiB
Aug 27 23:56:52 brice-X870E-Taichi ollama[1502099]: llama_kv_cache: size = 2160.00 MiB (  8192 cells,  27 layers,  1/1 seqs), K (f16): 1296.00 MiB, V (f16):  864.00 MiB
Aug 27 23:56:52 brice-X870E-Taichi ollama[1502099]: llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 192
Aug 27 23:56:52 brice-X870E-Taichi ollama[1502099]: llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 128
Aug 27 23:56:52 brice-X870E-Taichi ollama[1502099]: slot   load_model: id  0 | task -1 | new slot, n_ctx = 8192
```

### Denominator discipline

The protocol: "bytes/token = summed KV bytes ÷ the n_ctx THE LOG REPORTS
(not the requested one; parallel slots inflate effective n_ctx)."

Three independent log statements of the effective n_ctx agree at every
point, so the denominator is unambiguous:

| point | `llama_context: n_ctx` | `cells` in the kv_cache size line | `new slot, n_ctx` | denominator used |
|-------|------------------------|-----------------------------------|-------------------|------------------|
| 2048  | 2048 | 2048 | 2048 | **2048** |
| 4096  | 4096 | 4096 | 4096 | **4096** |
| 8192  | 8192 | 8192 | 8192 | **8192** |

Parallel-slot inflation did not occur. `-np 1` on the spawn line, and
(`raw-log-ctx2048.txt:206`, same shape at the other two points):
```
Aug 27 23:56:37 brice-X870E-Taichi ollama[1502099]: srv    load_model: initializing, n_slots = 1, n_ctx_slot = 2048, kv_unified = 'false'
```
`n_slots = 1` and `n_ctx_slot == n_ctx`, and the kv_cache size line
reports `1/1 seqs`. The log-reported n_ctx happens to equal the
requested num_ctx here; the denominator is nonetheless taken from the
log lines, per protocol, not from the request.

### Arithmetic, digit by digit

`1 MiB = 1024 × 1024 = 1,048,576 bytes.`

**ctx 2048**
```
summed KV bytes = 540.00 MiB (single CUDA0 line)
540 × 1,048,576   = 566,231,040 bytes
566,231,040 ÷ 2048 = 276,480 bytes/token
```
check: 276,480 × 2048 = 566,231,040 ✓

**ctx 4096**
```
summed KV bytes = 1080.00 MiB (single CUDA0 line)
1080 × 1,048,576    = 1,132,462,080 bytes
1,132,462,080 ÷ 4096 = 276,480 bytes/token
```
check: 276,480 × 4096 = 1,132,462,080 ✓

**ctx 8192**
```
summed KV bytes = 2160.00 MiB (single CUDA0 line)
2160 × 1,048,576    = 2,264,924,160 bytes
2,264,924,160 ÷ 8192 = 276,480 bytes/token
```
check: 276,480 × 8192 = 2,264,924,160 ✓

**Instrument 1 result: 276,480 bytes/token at all three points, exactly
and identically.** The three points agree, as the decision rule requires.

### K/V split — an independent structural check

The kv_cache size lines also break the total into K and V. This is not
required by the decision rule, but it constrains *which* model of the
cache is being read, so it is recorded:

| point | K (f16) | K bytes/token | V (f16) | V bytes/token |
|-------|---------|---------------|---------|---------------|
| 2048 | 324.00 MiB | 324 × 1,048,576 ÷ 2048 = **165,888** | 216.00 MiB | 216 × 1,048,576 ÷ 2048 = **110,592** |
| 4096 | 648.00 MiB | 648 × 1,048,576 ÷ 4096 = **165,888** | 432.00 MiB | 432 × 1,048,576 ÷ 4096 = **110,592** |
| 8192 | 1296.00 MiB | 1296 × 1,048,576 ÷ 8192 = **165,888** | 864.00 MiB | 864 × 1,048,576 ÷ 8192 = **110,592** |

165,888 + 110,592 = 276,480 ✓ (matches the total at every point)

Decomposed against the architecture the log itself reports
(`raw-log-ctx2048.txt`, `llama_model_loader` / `print_info` lines):
```
llama_model_loader: - kv   2:                      deepseek2.block_count u32              = 27
llama_model_loader: - kv   6:             deepseek2.attention.head_count u32              = 16
llama_model_loader: - kv   7:          deepseek2.attention.head_count_kv u32              = 16
llama_model_loader: - kv  14:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  15:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  16:           deepseek2.attention.value_length u32              = 128
print_info: n_embd_k_gqa          = 3072
print_info: n_embd_v_gqa          = 2048
```
```
K: 27 layers × 16 heads × 192 (key_length) × 2 bytes (f16)
   27 × 16 = 432 ; 432 × 192 = 82,944 ; 82,944 × 2 = 165,888  ✓ matches measured K
V: 27 layers × 16 heads × 128 (value_length) × 2 bytes (f16)
   27 × 16 = 432 ; 432 × 128 = 55,296 ; 55,296 × 2 = 110,592  ✓ matches measured V
```
Equivalently via the log's own aggregate widths:
`(n_embd_k_gqa 3072 + n_embd_v_gqa 2048) × 2 bytes × 27 layers`
`= 5120 × 2 = 10,240 ; 10,240 × 27 = 276,480` ✓

So the measured total is not merely numerically near a candidate — the
K and V halves independently reproduce the separate-width model, digit
for digit. Note also `kv_lora_rank = 512` / `print_info: n_lora_kv = 512`:
the blob *is* an MLA architecture, and this runtime is nevertheless
materialising full per-head K and V rather than caching the 512+64
latent. That observation is recorded, not used to adjust the rule.

---

## 2. Instrument 2 (corroboration only) — VRAM slope

Per protocol: Instrument 2 **cannot decide**. Its only job is to be
consistent with Instrument 1 — same order of magnitude, and slope ≥
Instrument 1's figure (the stated confound is that compute buffers also
grow with ctx, so the slope over-reads KV).

### Validity condition — checked from the ps files myself

The protocol's validity condition is "ollama ps reports 100% GPU, else
the point is named in dropped." Read directly from the ps files:

`ps-ctx2048.txt`
```
NAME                                          ID              SIZE     PROCESSOR    CONTEXT    UNTIL              
deepseek-coder-v2:16b-lite-instruct-q5_K_M    6065d4880bf9    12 GB    100% GPU     2048       2 minutes from now    
```
`ps-ctx4096.txt`
```
deepseek-coder-v2:16b-lite-instruct-q5_K_M    6065d4880bf9    13 GB    100% GPU     4096       2 minutes from now    
```
`ps-ctx8192.txt`
```
deepseek-coder-v2:16b-lite-instruct-q5_K_M    6065d4880bf9    14 GB    100% GPU     8192       2 minutes from now    
```

All three report `PROCESSOR = 100% GPU`, and each reports the CONTEXT
matching its point (2048 / 4096 / 8192).

**`dropped: []`** — empty. No point fails the residency condition, so no
point is dropped.

### Deltas

Each `vram-ctxN.txt` is: line 1 = before (`used MiB, free MiB`), line 2 =
after, then the per-process table. Verbatim:

`vram-ctx2048.txt`
```
739 MiB, 15101 MiB
12857 MiB, 2983 MiB
pid, process_name, used_gpu_memory [MiB]
905038, /usr/bin/ptyxis, 29 MiB
906744, /usr/bin/lact, 52 MiB
3643459, /usr/local/lib/ollama/llama-server, 12110 MiB
```
`vram-ctx4096.txt`
```
739 MiB, 15101 MiB
13481 MiB, 2359 MiB
pid, process_name, used_gpu_memory [MiB]
905038, /usr/bin/ptyxis, 29 MiB
906744, /usr/bin/lact, 52 MiB
3643691, /usr/local/lib/ollama/llama-server, 12734 MiB
```
`vram-ctx8192.txt`
```
739 MiB, 15101 MiB
14713 MiB, 1127 MiB
pid, process_name, used_gpu_memory [MiB]
905038, /usr/bin/ptyxis, 29 MiB
906744, /usr/bin/lact, 52 MiB
3643979, /usr/local/lib/ollama/llama-server, 13966 MiB
```

Per-point delta (after − before):
```
ctx 2048: 12,857 − 739 = 12,118 MiB   (llama-server process alone: 12,110 MiB)
ctx 4096: 13,481 − 739 = 12,742 MiB   (llama-server process alone: 12,734 MiB)
ctx 8192: 14,713 − 739 = 13,974 MiB   (llama-server process alone: 13,966 MiB)
```
The before reading is identical (739 MiB) at all three points — each
point started from an unloaded GPU, so the deltas are comparable. The
global delta and the per-process figure differ by a constant 8 MiB at
every point, so the slope is identical either way; global deltas are
used below.

### Slope

```
2048 → 4096: (12,742 − 12,118) = 624 MiB over (4096 − 2048) = 2048 tokens
             624 × 1,048,576 = 654,311,424 bytes
             654,311,424 ÷ 2048 = 319,488 bytes/token

4096 → 8192: (13,974 − 12,742) = 1,232 MiB over (8192 − 4096) = 4096 tokens
             1,232 × 1,048,576 = 1,291,845,632 bytes
             1,291,845,632 ÷ 4096 = 315,392 bytes/token

2048 → 8192: (13,974 − 12,118) = 1,856 MiB over (8192 − 2048) = 6144 tokens
             1,856 × 1,048,576 = 1,946,157,056 bytes
             1,946,157,056 ÷ 6144 = 316,757.33 bytes/token
```

### Consistency with Instrument 1

- **Same order of magnitude?** Yes. 315,392 / 316,757 / 319,488 vs
  Instrument 1's 276,480 — all 10^5, all within a factor 1.16.
- **Slope ≥ Instrument 1's figure?** Yes, on every segment:
  319,488 ≥ 276,480 (ratio 1.156); 315,392 ≥ 276,480 (ratio 1.141);
  316,757 ≥ 276,480 (ratio 1.146).

**Instrument 2 is CONSISTENT with Instrument 1**, in the exact direction
the protocol pre-registered (over-read, never under-read).

The over-read is fully accounted for by the pre-registered confound. The
compute buffers, quoted above, do grow with ctx:
```
ctx 2048: CUDA0 104.27 + CUDA_Host 12.01 = 116.28 MiB
ctx 4096: CUDA0 188.27 + CUDA_Host 16.01 = 204.28 MiB
ctx 8192: CUDA0 340.27 + CUDA_Host 24.01 = 364.28 MiB
```
```
2048 → 8192 predicted growth = KV delta + compute delta
  KV delta      = 2160 − 540      = 1,620.00 MiB
  compute delta = 364.28 − 116.28 =   248.00 MiB
  predicted total                 = 1,868.00 MiB
  measured global delta           = 1,856 MiB
  residual = 1,868 − 1,856 = 12 MiB  (0.6% of 1,856 — allocator granularity
                                      and nvidia-smi's MiB rounding)
```
That decomposition is corroboration, not a second reading: Instrument 2
still cannot decide, and does not need to. Instrument 1 decides alone.

---

## 3. Recorded anomaly — `load_mode = none` at ctx2048 only

Carried from Task 2's review, and stated explicitly rather than passed
over in silence.

ctx2048 loaded with mmap **disabled**; 4096 and 8192 used mmap. The
reason is in the log (`raw-log-ctx2048.txt:1`, present only in that
file):
```
Aug 27 23:56:36 brice-X870E-Taichi ollama[1502099]: time=2026-08-27T23:56:36.676-05:00 level=INFO source=sched.go:1215 msg="disabling mmap for llama-server load due to host memory pressure" model=/mnt/extra/…/sha256-bc286970… model_size="11.0 GiB" loaded_mmap_size="0 B" headroom="7.5 GiB" system_free="16.1 GiB" system_total="29.9 GiB" predicted_vram="11.5 GiB" available_vram="14.5 GiB"
```
Consequently `--load-mode none` appears on the ctx2048 spawn line (§0)
but on neither of the others, and the tensor-load line differs:
```
raw-log-ctx2048.txt:160  load_tensors: loading model tensors, this can take a while... (load_mode = none)
raw-log-ctx4096.txt:159  load_tensors: loading model tensors, this can take a while... (load_mode = mmap)
raw-log-ctx8192.txt:159  load_tensors: loading model tensors, this can take a while... (load_mode = mmap)
```
It also shows in the *model* buffer lines: ctx2048 reports
`CUDA_Host model buffer size = 137.50 MiB` where the other two report
`CPU_Mapped model buffer size = 137.50 MiB` — same 137.50 MiB, different
host-allocation path.

**Assessment: this concerns the weights path, not KV cache sizing.**
Saying so is not hand-waving it away; the evidence for the claim is that
every KV-relevant quantity is bit-identical across the mmap boundary:
- `CUDA0 model buffer size = 11160.99 MiB` — identical at all three
  points, mmap or not.
- the KV bytes/token derived at ctx2048 (load_mode=none) is **276,480**,
  the identical figure derived at ctx4096 and ctx8192 (load_mode=mmap).
- the K/V split per token (165,888 / 110,592) is identical across the
  boundary too.

If the mmap difference perturbed KV allocation at all, ctx2048 would be
the outlier. It is not — it lands on the same figure to the byte. Note
also the honest limit of that argument: it shows the anomaly did not
*affect this measurement*, not that mmap can never matter. Nothing is
dropped or adjusted on account of it; all three points stand as data.

Also recorded, uniform across all three points and therefore part of the
lens rather than a per-point anomaly: `llama_context: flash_attn = auto`
resolved to Flash Attention **not** running on the CUDA device —
```
resolve_fused_ops: layer 0 is assigned to device CUDA0 but Flash Attention is assigned to device CPU (usually due to missing support)
```
(present in all three logs, 11 occurrences each). The 276,480 figure is
therefore a reading of the non-FA path. Since it is constant across the
three points it cannot confound the slope, but it is a lens condition a
future FA-enabled build could change.

---

## 4. Infrastructure vs data

Per protocol, "a failed load / missing log line / OOM is infrastructure,
recorded as such, never data." `infrastructure-notes.txt` records one
such event: a `journalctl --since` timezone mismatch that produced three
empty log files on the first attempt. Those empty files were not
committed and no content from that attempt was spliced in; the committed
raw files are a fresh complete set. All three loads in the committed set
succeeded (28/28 layers offloaded, 100% GPU, KV lines present). No
committed point is infrastructure.

---

## 5. Decision-rule application

Rule (verbatim from `PROTOCOL.md`): "Instrument 1's bytes/token matching
exactly ONE candidate within ±5% decides. Bands are disjoint (closest
pair 20% apart). No match → no rule is written."

Instrument 1 = **276,480 bytes/token**, identical at 2048, 4096 and 8192,
so the "all three points must agree on the candidate" condition holds
trivially — it is one number, not three near numbers.

| Id | Candidate B/token | −5% bound | +5% bound | 276,480 in band? |
|----|-------------------|-----------|-----------|------------------|
| H-a | 331,776 | 331,776 × 0.95 = 315,187.2 | 331,776 × 1.05 = 348,364.8 | **OUT** (276,480 < 315,187.2) |
| H-b | 276,480 | 276,480 × 0.95 = 262,656.0 | 276,480 × 1.05 = 290,304.0 | **IN** (exact centre) |
| H-c | 31,104 | 31,104 × 0.95 = 29,548.8 | 31,104 × 1.05 = 32,659.2 | **OUT** (276,480 > 32,659.2) |
| H-c' | 62,208 | 62,208 × 0.95 = 59,097.6 | 62,208 × 1.05 = 65,318.4 | **OUT** (276,480 > 65,318.4) |

Exactly one candidate matches: **H-b**. Deviation from H-b is 0.000% —
the measurement is not merely inside the band, it is the candidate value
to the byte.

→ Verdict **H-b**. Written up in `VERDICT.md`.
