# Floors

A **floor** is a frozen Assay profile that states the minimum capability bar for a job
(for example agent search/replace edits). `assay cover <floor> <candidate>` asks whether
the candidate meets every measured cell the floor established.

## Discipline

- Floors are **frozen**. Do not silently rewrite a committed floor to make CI green.
- If a floor was wrong, add an erratum note and cut a new floor file (new name or version).
- Floor and candidate must come from a **comparable instrument** (same probe generation /
  schema family). Crossed models are allowed; crossed instruments are refused.
- Produce floors with a real `assay probe` on reference hardware, then commit the JSON.

## Starter floor in this repo

`agent-edit.json` is a **starter** floor copied from the enthusiast-tier evidence set
(`qwen2.5-coder:14b-instruct-q4_K_M`, quick profile) so CI and docs have a real artifact.
Treat it as an example bar, not a product SLA, until a dedicated floor campaign replaces it.

Companion candidate for CI: `examples/cover-ci/candidate.json`
(`qwen2.5-coder:7b-instruct-q8_0`), which covers this floor under current cover rules.
