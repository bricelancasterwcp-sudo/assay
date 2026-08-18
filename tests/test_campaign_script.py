"""The campaign wrapper's durable paths.

CARRIED-DEBT closed here: the wrapper was pinned to
`.worktrees/v17`, a worktree deleted when v1.7 closed, so every rerun
died at its own preflight. A script committed to the repository must
point at the repository — an ephemeral path in a durable artifact is
the same defect class as a debt file that delegates to a worktree.
"""

import pathlib
import re

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts/campaign-2026-08.sh"


def _assignment(name: str) -> str:
    text = _SCRIPT.read_text(encoding="utf-8")
    match = re.search(rf"^{name}=(.+)$", text, re.MULTILINE)
    assert match is not None, f"{name} is not assigned in {_SCRIPT.name}"
    return match.group(1).strip()


def test_the_wrapper_points_at_the_repository_not_a_worktree():
    repo = _assignment("REPO")
    assert ".worktrees" not in repo, (
        "a committed script must not depend on a worktree: worktrees are "
        "removed at the close of the wave that made them")
    assert pathlib.Path(repo).exists() and pathlib.Path(repo).is_dir()
    assert (pathlib.Path(repo) / "src/assay/__init__.py").exists()


def test_the_wrapper_uses_the_repo_venv():
    assert _assignment("ASSAY") == '"$REPO/.venv/bin/assay"'
