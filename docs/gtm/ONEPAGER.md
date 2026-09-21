# Assay — one-pager

**Admission control for locally served LLMs.**

Before your agent, IDE, or appliance spends GPU time on a model, Assay answers: **ready**, **risky**, or **unusable** — for the *jobs you care about*, on the *endpoint you actually run*.

---

## The problem

Local stacks fail in ways leaderboards don’t measure:

- The server **silently truncates** the prompt and answers confidently.
- HTTP 200 with **missing stats** — looks healthy, contract is broken.
- Model A lands search/replace edits; Model B lands **0%** of the same codec.
- Tool calling / loop discipline collapse under multi-step use.

IQ benches won’t tell you. Shipping work to the model will — the expensive way.

---

## What Assay is

A **stdlib-only** Python CLI/library that probes a local endpoint (Ollama / OpenAI-compatible) and emits a **versioned capability profile**:

- Context **geometry** and real **prompt ceiling**
- Format / **codec landing** (edit formats, JSON shapes, …)
- Speed, loop discipline, tools, long output, parallel lanes
- Machine-readable **verdicts** (`ready` / `risky` / `unusable` / …)

Plus offline tools:

- `assay diff` — did anything move beyond noise?
- `assay cover` — does this candidate **cover** my floor?
- `assay report` — matrix HTML from N profiles

**Assay measures instrument fitness, not intelligence.**

Published enthusiast-tier matrix:  
https://bricelancasterwcp-sudo.github.io/assay/matrix/

---

## Who it’s for

| Audience | Use |
|----------|-----|
| Agent / IDE authors | Refuse models that can’t land your edit codec |
| Appliance / local AI OS (e.g. bloomery) | Admit workloads only to measured-capable endpoints |
| Platform / ML ops | CI gate: candidate profile must cover frozen floor |
| Enthusiasts | Pick a local model that won’t waste an evening |

---

## 30-second path (target after PyPI)

```bash
pip install llm-assay   # distribute name TBD — see checklist
assay probe http://127.0.0.1:11434 --model qwen2.5-coder:7b --quick --json profile.json
assay cover floors/agent-edit.json profile.json
```

Exit `0` → covered. Nonzero → don’t ship that model for that job.

---

## Positioning (say this / not that)

| Say | Don’t say |
|-----|-----------|
| Admission control for local endpoints | “Another LLM benchmark” |
| Ready / risky / unusable for *this job* | “Smarter model ranking” |
| Versioned profiles + errata | “One true leaderboard” |
| Works offline on profiles you already have | “Cloud eval SaaS” (v1) |

---

## Business (open core)

- **Free (MIT):** probe, diff, cover, report, published matrices.
- **Paid later (optional):** fleet dashboards, continuous regression, hosted multi-machine matrix, support SLAs.
- **Never:** paywall the measurements that make the brand credible.

---

## Place in the stack

- **Assay** — can this endpoint do the job?
- **Sensorium** — what did the program *actually* do when it ran?
- **Bloomery** — appliance OS that shouldn’t lie about VRAM/KV — pins Assay for honesty.

Assay is the trust layer. Keep the core; ship install + CI gate next; GTM after Sensorium’s Show HN lands.
