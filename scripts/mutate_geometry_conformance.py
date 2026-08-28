#!/usr/bin/env python3
"""Mutation check for ``tests/test_geometry_conformance.py``.

The conformance harness is only worth its report if each assertion is
load-bearing: a test that passes whether or not the code is correct
measures nothing. This script proves the assertions bite by breaking ONE
line of the code (or one byte of the vendored data) each mutant is aimed
at, and requiring the named test selection to go red.

Run it as ``python scripts/mutate_geometry_conformance.py`` from the
repository root (``scripts/`` is not a package, same pin as
``build_matrix.py``). It lives in the tree so the mutation claim in a
task report is re-runnable by a reviewer rather than taken on trust.

Method, and why each part is there:

* **Baseline first.** A mutant that "fails" a selection which was already
  failing proves nothing, so every case runs the selection unmutated and
  requires green before mutating.
* **Unique anchor.** Each mutation is an exact-string replace whose
  anchor must occur exactly once; a second occurrence would make the edit
  ambiguous and the kill unattributable.
* **Byte-for-byte restore, verified.** The original text is written back
  in a ``finally``, and the file's sha256 before and after must match —
  not just "we wrote something back".
* **Post-restore re-run.** The selection must be green again afterwards.
  A mutant is only KILLED on green -> red -> green.
* **Stale-bytecode purge.** ``__pycache__`` is removed before *every*
  run and ``PYTHONDONTWRITEBYTECODE=1`` is set. A same-length edit
  written inside the same second as the original leaves CPython's
  mtime+size cache validation happy, and the interpreter then runs the
  *old* bytecode — which silently turns every mutant into a survivor.
* **Clean tree in, clean tree out.** The worktree must be free of
  uncommitted changes at the start (otherwise a previous interrupted run
  could be mistaken for the baseline) and is checked again at the end.
  A dirty tree at exit is a hard error naming the files.
* **Scrubbed environment.** ``PYTEST_ADDOPTS`` and an inherited
  ``PYTHONPATH`` are dropped, so an ambient setting cannot quietly
  deselect the tests being measured.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

DEFAULT_RUNNER = (
    "uv run --offline --with pytest --python 3.12 python -m pytest"
)

TESTMOD = "tests/test_geometry_conformance.py"

#: The four E1-corpus vectors whose metadata states ``attention.key_length``
#: and whose head_dim would differ if it were derived instead of read.
E1_AFFECTED = (
    "codegemma-7b-instruct-q8_0",
    "deepseek-coder-v2-16b-lite-instruct-q5_K_M",
    "gemma2-9b",
    "mistral-nemo-latest",
)


def cases(root: pathlib.Path) -> list[tuple[str, pathlib.Path, str, str, list[str]]]:
    """(label, file, original, mutant, pytest selection) per mutant."""
    ollama = root / "src/assay/backends/ollama.py"
    geometry = root / "src/assay/geometry.py"
    testfile = root / TESTMOD
    data = root / "tests/data/gguf_geometry_v3"

    return [
        (
            "M1 head_dim: ignore stated key_length, always derive (R1 / erratum E1)",
            ollama,
            '    key_length = _arch_value(arch_info, "attention.key_length")\n'
            "    if key_length is not None:\n        return key_length\n",
            '    key_length = _arch_value(arch_info, "attention.key_length")\n'
            "    if key_length is not None and False:\n        return key_length\n",
            [f"{TESTMOD}::test_kv_interpretation_conforms[{m}]" for m in E1_AFFECTED],
        ),
        (
            "M2 kv_bytes_per_token: guess 0 instead of refusing (R8)",
            geometry,
            "    if any(part is None for part in parts):\n        return None\n",
            "    if any(part is None for part in parts):\n        return 0\n",
            [
                f"{TESTMOD}::test_kv_interpretation_conforms"
                "[gemma-4-12b-it-qat-q4_0-latest]"
            ],
        ),
        (
            "M3 expert_count: absent keys become 0, not None (R5)",
            ollama,
            '            expert_count=_arch_value(arch_info, "expert_count"),\n',
            '            expert_count=_arch_value(arch_info, "expert_count") or 0,\n',
            [
                f"{TESTMOD}::test_expert_fields_conform"
                "[qwen2.5-coder-7b-instruct-q8_0]"
            ],
        ),
        (
            "M4 plan_window: drop the fixed-overhead subtraction (R7 term)",
            geometry,
            "                (vram_free_mib - overhead_mib) * _MIB\n",
            "                vram_free_mib * _MIB\n",
            [
                f"{TESTMOD}::test_window_law_conforms"
                "[qwen2.5-coder-7b-instruct-q8_0-0-budget]"
            ],
        ),
        (
            "M5 plan_window: max instead of min over candidate terms (R7 law)",
            geometry,
            "    limited_by, usable_window = min(candidates, key=lambda term: term[1])\n",
            "    limited_by, usable_window = max(candidates, key=lambda term: term[1])\n",
            [f"{TESTMOD}::test_window_law_conforms"],
        ),
        (
            "M6 plan_window: rename the budget term (limited_by mapping)",
            geometry,
            '            candidates.append(("vram", kv_fit))\n',
            '            candidates.append(("accelerator", kv_fit))\n',
            [
                f"{TESTMOD}::test_window_law_conforms"
                "[qwen2.5-coder-7b-instruct-q8_0-0-budget]"
            ],
        ),
        (
            "M7 vendored vector edited (structural pin, not the manifest pin)",
            data / "gemma2-9b.json",
            '"gemma2.block_count": 42',
            '"gemma2.block_count": 41',
            [f"{TESTMOD}::test_the_vendored_copy_is_the_set_the_manifest_names"],
        ),
        (
            "M8 vendored MANIFEST edited (sha pin)",
            data / "MANIFEST.json",
            '"set_version": "v3"',
            '"set_version": "v2"',
            [f"{TESTMOD}::test_vendored_manifest_is_pinned"],
        ),
        (
            "M9 harness collects nothing (empty-suite guard)",
            testfile,
            '        if path.name != "MANIFEST.json"\n',
            '        if path.name == "MANIFEST.json"\n',
            [f"{TESTMOD}::test_the_suite_actually_covers_the_frozen_set"],
        ),
        (
            # Turns M8 into a no-op mutant (its "mutation" becomes its own
            # original), which is the way a mutation table rots without
            # anyone noticing: the case still runs, still reports, and
            # proves nothing. The suite's guard test must catch it.
            #
            # Mutating this very file works because the guard test re-reads
            # it from disk while this already-imported process keeps running
            # the original table. The anchor is spelled as a concatenation
            # on purpose: written as one literal it would occur twice in
            # this file — here and in M8 — and an ambiguous anchor is
            # refused before anything is edited.
            "M10 a harness case's mutant equals its original (no-op-mutant guard)",
            root / "scripts/mutate_geometry_conformance.py",
            '"set_version": ' + '"v2"',
            '"set_version": "v3"',
            [f"{TESTMOD}::test_the_mutation_harness_still_aims_at_real_lines"],
        ),
        (
            # The historical regression itself: charge the raw serving
            # count instead of dividing by the stated interval. The
            # vector's `must_not_equal` 81920 is exactly what this
            # produces, so the ban assertion fires as well as the value.
            "M11 attention_layer_count: charge every block, not 1 in N (R3)",
            geometry,
            "    layers = serving_blocks // full_attention_interval\n",
            "    layers = serving_blocks\n",
            [
                f"{TESTMOD}::test_attention_layer_count_conforms"
                "[qwen3.6-35b-a3b-reap48-ours-q4km]",
                f"{TESTMOD}::test_kv_interpretation_conforms"
                "[qwen3.6-35b-a3b-reap48-ours-q4km]",
            ],
        ),
        (
            # R6 is pinned by the serving-count assertion ALONE on this
            # corpus: 41 // 4 and 40 // 4 are both 10, so dropping the
            # subtraction leaves the trap vector's kv figure correct and
            # only its block count wrong — which is precisely the state
            # that produced an unloadable GGUF.
            "M12 serving_block_count: keep the MTP layer as a serving block (R6)",
            geometry,
            "    return block_count - mtp_layer_count\n",
            "    return block_count\n",
            [
                f"{TESTMOD}::test_serving_block_count_conforms"
                "[qwen3.6-35b-a3b-reap48-mtp-trap]"
            ],
        ),
        (
            "M13 recurrent_state_bytes: drop the conv-state term (R4)",
            geometry,
            "    per_layer = conv_state + state_size * inner_size\n",
            "    per_layer = state_size * inner_size\n",
            [f"{TESTMOD}::test_recurrent_state_conforms"],
        ),
        (
            # R9 (MLA, separate widths), guard flip. deepseek's vector is
            # the only one of the 11 whose stated value_length (128)
            # differs from its head_dim (192) — the only vector this
            # branch touches — so it is the whole selection.
            "M14 R9 guard: flip != to == (MLA, separate widths)",
            geometry,
            "    if info.value_length is not None and "
            "info.value_length != info.head_dim:\n",
            "    if info.value_length is not None and "
            "info.value_length == info.head_dim:\n",
            [
                f"{TESTMOD}::test_kv_interpretation_conforms"
                "[deepseek-coder-v2-16b-lite-instruct-q5_K_M]"
            ],
        ),
        (
            "M15 R9 sum: drop the + info.value_length term (MLA)",
            geometry,
            "        return layers * info.kv_head_count * "
            "(info.head_dim + info.value_length) * (kv_bits // 8)\n",
            "        return layers * info.kv_head_count * "
            "(info.head_dim) * (kv_bits // 8)\n",
            [
                f"{TESTMOD}::test_kv_interpretation_conforms"
                "[deepseek-coder-v2-16b-lite-instruct-q5_K_M]"
            ],
        ),
    ]


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def purge_pycache(root: pathlib.Path) -> None:
    for path in root.rglob("__pycache__"):
        if ".venv" in path.parts:
            continue
        shutil.rmtree(path, ignore_errors=True)


def git_dirty(root: pathlib.Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def run(root: pathlib.Path, runner: list[str], selection: list[str]):
    purge_pycache(root)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTEST_ADDOPTS", "PYTHONPATH", "VIRTUAL_ENV")
    }
    env["PYTHONPATH"] = "src"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [*runner, *selection, "-q", "--no-header"],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
    )


def last_line(result: subprocess.CompletedProcess) -> str:
    lines = (result.stdout or result.stderr).strip().splitlines()
    return lines[-1] if lines else "<no output>"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="repository root to mutate (default: this script's repo)",
    )
    parser.add_argument(
        "--runner",
        default=DEFAULT_RUNNER,
        help=f"pytest invocation, quoted (default: {DEFAULT_RUNNER!r})",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    runner = shlex.split(args.runner)

    dirty = git_dirty(root)
    if dirty:
        print("REFUSING: worktree is not clean; a baseline cannot be trusted.")
        print(dirty)
        return 2
    print(f"root:   {root}")
    print(f"runner: {' '.join(runner)}")
    print()

    survivors: list[str] = []
    for label, path, original, mutant, selection in cases(root):
        source = path.read_text()
        if source.count(original) != 1:
            print(f"[ERROR] {label}: anchor occurs {source.count(original)}x, need 1")
            survivors.append(label)
            continue
        before = sha256(path)

        baseline = run(root, runner, selection)
        path.write_text(source.replace(original, mutant))
        try:
            result = run(root, runner, selection)
        finally:
            path.write_text(source)
        purge_pycache(root)
        restored_sha = sha256(path)
        after = run(root, runner, selection)

        killed = (
            baseline.returncode == 0
            and result.returncode != 0
            and after.returncode == 0
            and restored_sha == before
        )
        status = "KILLED" if killed else "SURVIVED"
        if not killed:
            survivors.append(label)
        print(f"[{status}] {label}")
        print(f"    file:     {path.relative_to(root)}")
        print(f"    baseline: rc={baseline.returncode}  {last_line(baseline)}")
        print(f"    mutant:   rc={result.returncode}  {last_line(result)}")
        print(f"    restored: rc={after.returncode}  {last_line(after)}")
        print(f"    sha256:   {before[:16]}… -> {restored_sha[:16]}… "
              f"({'identical' if restored_sha == before else 'CHANGED'})")

    print()
    dirty = git_dirty(root)
    print(f"worktree after the run: {dirty or 'clean'}")
    print("SURVIVORS:", ", ".join(survivors) if survivors else "none")
    return 1 if (survivors or dirty) else 0


if __name__ == "__main__":
    sys.exit(main())
