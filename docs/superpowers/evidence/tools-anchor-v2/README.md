# scripted-tools-v2 live anchor — llama3.1:8b, 2026-08-17

The v1 anchor ([`../tools-anchor/`](../tools-anchor/)) pins the **v1**
tools instrument — five tasks, fixed n — against four live endpoints.
This directory does the same job for the instrument that shipped in
v1.7: **`scripted-tools-v2`**, twenty tasks walked under the sequential
look schedule `TOOLS_LOOK_SCHEDULE` = 5 / 10 / 20.

| | |
|---|---|
| Model | `llama3.1:8b` (Q4_K_M) |
| Captured | 2026-08-17, 13:45:52→13:52:37Z |
| Endpoint | ollama 0.32.13, `http://127.0.0.1:11434` |
| Probe | assay 0.9.0 at commit `d9b9792` |
| Rows | 40 `chat_tools` (20 tasks × 2 turns) |

## Where these bytes came from

**Not a bespoke capture.** They are the `chat_tools` slice of a *whole
full-mode probe run* — the tier re-profile campaign's llama3.1:8b row,
written by [`scripts/campaign-2026-08.sh`](../../../../scripts/campaign-2026-08.sh):

- `tools-llama3.1-8b.jsonl` — the 40 `chat_tools` lines of
  [`../tier-enthusiast-2026-08/llama3.1-8b-transcript.jsonl`](../tier-enthusiast-2026-08/llama3.1-8b-transcript.jsonl),
  **verbatim**: whole lines, copied, never reserialized. The 332
  `generate` rows of the other families are the only thing dropped.
- `results.json` — the `Tools` values obtained by **replaying** those
  lines through the unmodified probe (`CallReplayer` +
  `probe_tools(look_schedule=TOOLS_LOOK_SCHEDULE)`), plus the capture
  metadata. No value here was typed by hand; each one either came out of
  the replay or was read off the committed profile.

That provenance buys a check the v1 anchor cannot have. Because the rows
are a slice of a run that ALSO wrote
[`../tier-enthusiast-2026-08/llama3.1-8b.json`](../tier-enthusiast-2026-08/llama3.1-8b.json),
the replayed values and that profile's `tools` block are the same
measurement reached by two roads, and the suite pins them to each other.
The anchor and the profile cannot drift apart without a test failing.

## The measurement

```
supported true · call_rate 1.0 · right_tool_rate 1.0 · args_valid_rate 1.0
composite 1.0 · result_use_rate 0.4
n_tasks 20 · n_turns 40 · n_truncated 1 · n_stop_unreported 0
stopping_rule "wilson95-looks-5-10-20"
```

Two things are worth reading off this row rather than past it:

1. **The schedule ran to the end.** `stopping_rule` names the looks, and
   `n_tasks` is 20 — the composite never cleared a rung early, so the
   endpoint bought the widest n the pool offers. A 20/20 composite still
   reads *provisional*: the profile's `tool_calling` verdict is
   `ready`, `provisional: true`, `interval95: [0.839, 1.0]` — nothing
   below n=35 clears the 0.9 floor non-provisionally, and the instrument
   says so rather than rounding up to a certainty the pool cannot buy.
2. **Calling and comprehending are different facts.** Perfect protocol
   (`call_rate`, `right_tool_rate`, `args_valid_rate` all 1.0) sits
   beside `result_use_rate` **0.4** — this endpoint emits well-formed
   calls every time and quotes the tool's answer back less than half the
   time. That gap is exactly what T2 exists to see.

## What the suite holds this to

In [`tests/test_tools.py`](../../../../tests/test_tools.py), all read
from the committed bytes with no daemon, no GPU, no network:

- `test_the_tools_v2_anchor_capture_is_committed_whole` — the metadata
  matches the LIVE constants (`TOOLS_INSTRUMENT`, `TOOLSET_NAME`,
  `TOOLS_LOOK_SCHEDULE`), so a pool or schedule edit invalidates this
  anchor instead of being re-described by it.
- `test_every_committed_v2_capture_replays_to_its_recorded_values` — the
  acceptance test: re-derive by replay and compare, over **every** field
  of `Tools`, so a value pruned out of `results.json` stops being
  checked loudly rather than quietly.
- `test_the_v2_anchor_and_its_campaign_profile_cannot_drift_apart` — the
  replay against the campaign profile's `tools` block, field for field.
- `test_the_v2_anchor_rows_are_the_campaign_transcript_verbatim` — the
  "extracted" claim, held to the bytes: the committed lines must equal
  the campaign transcript's `chat_tools` lines in file order. A tidied
  or re-recorded copy fails here even when it replays identically.
- `test_the_same_bytes_under_v1s_rule_are_a_different_measurement` — why
  `look_schedule` is recorded rather than assumed. The first ten of
  these rows are also a valid v1 run (the pools share tasks 0–4
  verbatim), and replayed with no schedule the probe reports `fixed-n`,
  `n_tasks` 5, `n_truncated` 0. The schedule decides the numbers.

## Rules

These bytes are evidence. They are not edited to suit a later fix; if
the instrument changes, this anchor is superseded by a new capture, per
[`../tier-enthusiast/ERRATA.md`](../tier-enthusiast/ERRATA.md).
