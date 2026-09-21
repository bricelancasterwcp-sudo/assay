# Assay quickstart

Admission control for a **local** LLM endpoint: measure, then ask whether a candidate covers your floor.

## Install (from checkout)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
assay --help
```

PyPI distribute name will be **`llm-assay`** (the name `assay` is taken). Until publish:

```bash
pip install -e /path/to/assay
```

## Probe one model

```bash
# Ollama example — adjust URL/model
assay probe http://127.0.0.1:11434 --model qwen2.5-coder:7b-instruct-q8_0 --quick --json profile.json
```

## Cover a floor

```bash
assay cover floors/agent-edit.json profile.json
# exit 0 = covered; nonzero = not covered / refused / incomplete / infra
```

CI example uses `floors/agent-edit.json` and `examples/cover-ci/candidate.json`.

## Read next

- [ONEPAGER.md](./ONEPAGER.md) — positioning
- [pypi-and-action-checklist.md](./pypi-and-action-checklist.md) — ship checklist
- Root [README.md](../../README.md) — full instrument docs
