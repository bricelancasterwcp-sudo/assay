# The degeneracy anchor

Live capture behind the long-output floors in `src/assay/long_output.py`.
Result in one line: **`ZLIB_FLOOR` is now derived (0.20 -> 0.2557);
`DISTINCT_FLOOR` could not be derived and stays assumed at 0.30, because
the degenerate and healthy clusters OVERLAP on that metric.**

## Provenance

| | |
|---|---|
| Captured | 2026-08-15 |
| Daemon | ollama **0.32.13** at `http://127.0.0.1:11434` |
| Hardware | NVIDIA RTX 5080, 16 GiB |
| Probe | `assay.long_output.probe_long_output`, unmodified |
| Task | `enumeration-v1` (`LONG_OUTPUT_TASK`) — the probe's own prompt, verbatim |
| Rungs | 512 / 1024 / 2048 / 4096 (`RUNGS`), one call each |
| Seeds | 1100, 1101, 1102, 1103 (`_LONG_SEED` + rung index) |
| Temperature | 0.2 (`PROBE_TEMPERATURE`) |
| `num_ctx` | unset — daemon default, exactly as `run.py` calls the probe |
| Models | 7, one at a time, unloaded between runs |
| Calls | 28 (7 models x 4 rungs) |

The probe was driven directly rather than through `assay probe`, wrapped in
`CallRecorder` — 4 calls per model instead of ~93 — because the anchor needs
the long-output family and nothing else. Prompt, seeds, rungs, temperature and
`num_ctx` are all the probe's own; nothing was tuned to provoke degeneracy, and
no prompt other than `_PROMPT` was sent.

Transcripts: `*-longoutput.jsonl`, one JSONL row per call, `CallRecorder`
format (replayable through `CallReplayer`). Human labels with the arithmetic
per sample: `labels.json`.

## What was captured

Three small models were run to hunt degeneracy (the brief's cap is 2-3) and
four larger ones to supply healthy same-task output.

| model | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| qwen2.5-coder:0.5b-instruct-q8_0 | **D** | **D** | **D** | **D** |
| qwen2.5-coder:1.5b-instruct-q8_0 | **D** | **D** | **D** | **D** |
| qwen2.5-coder:1.5b (q4_K_M) | healthy | **D** | **D** | healthy |
| qwen2.5-coder:7b-instruct-q8_0 | healthy | healthy | healthy | healthy |
| gemma2:9b | healthy | healthy | healthy | healthy |
| llama3.1:8b | healthy | healthy | healthy | healthy |
| mistral-nemo:latest | healthy | healthy | healthy | healthy |

**Degeneracy was found**, unambiguously and without prompting for it. The
0.5b locks onto one sentence and emits it for the rest of the generation:

```
 1. The sun is the only natural light source in the world.
 2. The Earth is the only planet in our solar system.
 3. The moon is the only planet in our solar system.
 4. The moon is the only planet in our solar system.
 ...
38. The moon is the only planet in our solar system.
```

The q4 1.5b is the ladder shape the family exists to find: clean at 512,
looping at 1024 and 2048, clean again at 4096 (where it happened to stop
after 138 tokens). Degeneracy is per-generation, not a switch that flips
once at a size.

### Labelling

Samples were sorted into clusters by **reading them**, recorded in
`labels.json` alongside `duplicate_item_fraction` — the share of numbered
list items that restate an earlier item word for word. That fraction is a
labelling aid for audit only; it is not computed anywhere in `src/` and is
not a third metric. The clusters do not touch on it: **every healthy sample
duplicates 0% of its items, every degenerate sample duplicates >= 50%**, so
no sample sat near the labelling boundary.

Labelling by `distinct_n_ratio`/`zlib_ratio` and then deriving those floors
from the labels would have been circular — the floors would have reproduced
whatever they were already set to.

## Derivation

Rule (brief Step 2): per metric, the floor is the **midpoint of the gap
between the degenerate cluster's best value and the healthy cluster's worst
value**; on overlap the floor stays assumed.

The healthy cluster is every real healthy reply available, not only the
enumeration ones: the 248 committed code replies of `docs/evidence-transcripts/`
(>= 50 words) are real healthy output of a repetitive genre, they are what the
spec §3 amendment guard already holds the floors to, and a floor that flags
them is a floor that false-positives on real data. Both genres are therefore
in the denominator.

### zlib_ratio — DERIVED

```
degenerate best        0.236194   qwen2.5-coder:0.5b-instruct-q8_0 seed 1103 (4096)
healthy worst, enum    0.454239   qwen2.5-coder:7b-instruct-q8_0   seed 1103 (4096)
healthy worst, code    0.275208   sweep-hermes3-latest.jsonl
healthy worst, all     0.275208
                       ---------------------------------------------------
separated:             band (0.236194, 0.275208)
midpoint               (0.236194 + 0.275208) / 2 = 0.255701
ZLIB_FLOOR             0.2557        (4 dp, strictly inside the band)
```

### distinct_n_ratio — NOT DERIVED, STAYS ASSUMED

```
degenerate best        0.612745   qwen2.5-coder:1.5b seed 1101 (1024)
healthy worst, enum    0.935780   qwen2.5-coder:7b-instruct-q8_0 seed 1103
healthy worst, code    0.595238   sweep-hermes3-latest.jsonl
healthy worst, all     0.595238
                       ---------------------------------------------------
OVERLAP:               0.612745 (degenerate) > 0.595238 (healthy)
DISTINCT_FLOOR         0.30, unchanged and still assumed
```

A qwen2.5-coder:1.5b reply whose items 7-20 are the identical sentence scores
**0.6127**, higher than a healthy hermes3 code reply at **0.5952**. No single
genre-agnostic distinct-n floor separates them, so none was derived. This is
the metric's real limit, not a gap in the capture: a looping model that pads
each repeated line with a changing number keeps its 4-gram diversity up, which
is exactly the collapse `zlib_ratio` exists to catch instead (long_output's
module docstring says so, and here is the measurement).

Had the enumeration genre been taken alone, the distinct midpoint would have
been (0.6127 + 0.9358) / 2 = **0.7743** — and that floor would have flagged
real healthy code replies. Deriving on one genre and shipping the number as a
genre-agnostic threshold is the mistake this section refuses to make.

## What the derivation bought, and what it cost

Against the 28 captured enumeration samples plus the 248 committed code
replies:

| floors | degenerate caught | healthy false-positives |
|---|---|---|
| assumed (0.30 / 0.20) | **8 / 10** | 0 / 266 |
| derived (0.30 / 0.2557) | **10 / 10** | 0 / 266 |

The assumed `ZLIB_FLOOR` of 0.20 **missed two genuinely degenerate replies** —
0.5b seed 1103 (zlib 0.2362) and q4 1.5b seed 1101 (zlib 0.2260, the one with
14 identical lines out of 20) — and caught two more only barely, at 0.19756
and 0.19972. Its sensitivity was closer to luck than to calibration. The
derived floor catches all ten.

The cost is false-positive headroom, and it is the honest bad news here:

| floor | worst healthy | headroom |
|---|---|---|
| ZLIB_FLOOR 0.20 (assumed) | 0.275208 | 1.376x |
| ZLIB_FLOOR 0.2557 (derived) | 0.275208 | **1.076x** |
| DISTINCT_FLOOR 0.30 (assumed) | 0.595238 | 1.984x |

The derived floor sits 7.6% below the worst healthy reply ever recorded here
and 8.3% above the best degenerate one. That band is narrow because the two
genres nearly touch: repetitive-but-healthy code compresses almost as well as
looping prose. A future code reply slightly more repetitive than any of the
248 would false-positive, and if one appears the floor must yield to it — it
is a derived number, but derived from 276 samples, not from a population.
Both edges are pinned in `tests/test_long_output.py` so the margin cannot be
eaten silently.

For completeness, the 1110 committed replies shorter than 50 words (outside
the guard's window) score no lower than 0.3401 zlib and 0.6087 distinct, so
they clear the derived floor too.

## Threshold provenance

```python
THRESHOLDS_PROVENANCE = "derived-2026-08-15 (zlib only; distinct still assumed)"
```

The string no longer starts with `assumed`, so Task 9's forced-provisional cap
on long_output verdicts releases and those verdicts now follow the normal
rules. That is deliberate, and it rests on this: **every one of the ten
degenerate samples is flagged by the derived zlib floor alone.** The still-
assumed distinct floor is strictly redundant on this data — the only sample
below it (0.5b seed 1101, distinct 0.2969) is flagged by zlib at 0.0781 as
well — so the verdict's sensitivity rests on the derived number, while the
assumed one is a conservative backstop with 1.98x headroom that has never
fired on healthy output.

The mixed state travels in the string itself, which means it travels in every
profile's `verdicts.long_output.lens.thresholds`. A reader who quotes the
verdict cannot avoid reading that half the instrument is still assumed. The
cap machinery is untouched and still works: `tests/test_profile.py` pins that
an `assumed`-prefixed provenance forces `provisional` back to True.

## Reproducing

```bash
python - <<'PY'
from assay.backends.ollama import OllamaNative
from assay.budget import Budget, BudgetMeter
from assay.long_output import probe_long_output
from assay.replay import CallRecorder

live = OllamaNative("http://127.0.0.1:11434", "qwen2.5-coder:0.5b-instruct-q8_0",
                    timeout=900.0)
out = probe_long_output(
    CallRecorder(live, "0.5b.jsonl"),
    BudgetMeter(Budget(max_calls=50, max_prompt_tokens=10**7)),
    ceiling_max=None,
)
print([(r.target_tokens, r.zlib_ratio, r.degenerate) for r in out.rungs])
PY
```

Sampling is seeded but not bit-reproducible across daemon or driver versions;
the committed transcripts are the record, and the acceptance tests read them
rather than re-calling the daemon, so the suite stays offline and GPU-free.
