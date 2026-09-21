# Assay — PyPI publish + `cover` GitHub Action checklist

Draft 2026-09-20. Engineering next steps after positioning: admission control for local LLM endpoints.

## Blocker: PyPI name

`assay` on PyPI is already taken (Brandon Rhodes, `0.0`). Do **not** try to claim it.

**Recommended published name:** `llm-assay` (import/CLI stay `assay` via package config if desired, or rename CLI later).

Alternates: `local-assay`, `endpoint-assay`. Check `https://pypi.org/project/<name>/` before locking.

Also reserve the matching GitHub Topics / docs URL slug.

---

## A. PyPI publish checklist

### Package shape
- [ ] Decide distribute name (`llm-assay`) vs import name (`assay`) — keep import `assay` to avoid breaking bloomery `PYTHONPATH` consumers.
- [ ] Bump / confirm `pyproject.toml`: name, version, description, `requires-python`, MIT, project.urls (Homepage, Matrix, Source, Changelog).
- [ ] Add `[project.urls]` and long description from a **short** README section or `docs/gtm/QUICKSTART.md` (full README can stay deep).
- [ ] Confirm `assay` console script entry point still works: `assay = "assay.cli:main"` (or equivalent).
- [ ] `pip install -e .` and `pip install -e .[dev]` still green; wheel builds with `python -m build`.

### Artifacts
- [ ] `python -m build` → sdist + wheel.
- [ ] Twine check: `twine check dist/*`.
- [ ] Dry-run against TestPyPI: `twine upload --repository testpypi dist/*`.
- [ ] Install from TestPyPI in a clean venv and run: `assay --help`, `assay cover --help`.

### Release
- [ ] Tag `v0.14.0` (or next) matching package version; CHANGELOG already narrates meaning.
- [ ] GitHub Release with install one-liner and link to matrix.
- [ ] Upload to real PyPI (Trusted Publisher / API token — prefer OIDC trusted publishing from GitHub Actions).
- [ ] Verify: `pip install llm-assay` then `python -c "import assay; print(assay.__version__)"`.

### Post-publish
- [ ] Update README install block to PyPI first, clone second.
- [ ] Update bloomery pin docs from `PYTHONPATH=.../assay/src` → `pip install llm-assay==…` when ready.
- [ ] Add PyPI badge + matrix badge to README top.

---

## B. GitHub Action: `assay cover` gate

### Goal
Fail CI when a **candidate** profile does not cover a committed **floor** profile — admission control in PRs / nightly hardware jobs.

### Non-goals
- Running live `assay probe` on GitHub-hosted runners (no GPU / no local Ollama by default).
- Replacing nightly/real-hardware campaigns.

### Recommended workflow shapes

**1. Offline cover gate (default, cheap)**  
Inputs: `floor.json` (committed requirement) + `candidate.json` (artifact from a prior probe job or developer machine).

```yaml
# .github/workflows/assay-cover.yml
name: assay-cover
on:
  workflow_dispatch:
  pull_request:
    paths: ['floors/**', 'profiles/**', '.github/workflows/assay-cover.yml']

jobs:
  cover:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install assay
        run: pip install llm-assay==0.14.0   # pin
      - name: Cover floor
        run: |
          assay cover floors/agent-edit.json profiles/candidate.json
        # exit 0 covered, 1 not covered, 2 refused, 3 incomplete, 4 infra
```

**2. Composite action (publishable)**  
`bricelancasterwcp-sudo/assay-cover-action` with inputs:
- `floor` (path)
- `candidate` (path)
- `assay-version` (default pin)
- `fail-on` (`not-covered` | `incomplete` | `any-nonzero`)

Map exit codes explicitly in the action so CI summaries are readable.

**3. Optional self-hosted probe + cover**  
Self-hosted runner with Ollama:
1. `assay probe … --json profiles/$MODEL.json --quick` (or budget)
2. `assay cover floors/….json profiles/$MODEL.json`
3. Upload profile artifact + cover JSON.

Keep this **separate** workflow so PRs don’t depend on GPU availability.

### Floor files to invent (product)
Commit 1–3 floors under `floors/` with clear names, e.g.:
- `floors/agent-search-replace.json` — patch_editing ready
- `floors/tool-calling-min.json` — tool_calling ready
- `floors/chat-usable.json` — chat_speed + envelope floor

Document how floors are produced (probe once on reference hardware, freeze, never silently rewrite — same errata discipline as the matrix).

### Action checklist
- [ ] Pin `llm-assay` version in the workflow.
- [ ] Surface `assay cover --json` as a CI artifact.
- [ ] Annotate job summary: covered / not covered / refused / incomplete.
- [ ] Example repo or `examples/cover-ci/` in assay itself.
- [ ] README section: “Use as a CI gate”.

---

## C. Order of work (suggested)

1. Pick distribute name (`llm-assay`) and confirm free on PyPI.
2. QUICKSTART one-pager (see sibling doc) + README install trim.
3. Trusted Publisher + first PyPI upload.
4. `assay-cover` workflow example in-tree; then optional composite action.
5. Point bloomery at the package pin.

## D. Explicit non-work

- Don’t paywall `probe`.
- Don’t add cloud endpoint probing in v1.
- Don’t rename the scientific README into marketing mush — add a short path *above* it.
