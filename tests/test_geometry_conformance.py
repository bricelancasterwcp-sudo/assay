"""gguf-geometry v1 conformance: assay's extraction against the contract.

The vectors under ``tests/data/gguf_geometry_v1/`` are a vendored,
sha-pinned copy of the frozen v1 set from the gguf-geometry repo (rules
R1-R8 in that repo's SPEC.md). Every ``expected`` value in them was
measured on real hardware and cited there; nothing in this file computes
geometry, and nothing here may be edited to make a red go green — a red
is a divergence between assay and the contract, i.e. a work order.

Each vector's ``metadata`` block IS the ``model_info`` object Ollama's
/api/show returns, so the code under test is assay's own interpretation
seam, ``OllamaNative.model_info()``, followed by ``geometry.py``. A
metadata-to-``ModelInfo`` mapping written here instead would be a fourth
implementation of the contract, and this suite would then be measuring
the test rather than assay. No daemon, no GPU, no network: the transport
is replaced, the interpretation is not.

Known reds at authoring time (2026-08-27), both hybrid vectors:
``qwen3.6-35b-a3b-reap48-ours-q4km`` (R3: every block charged as an
attention layer where ``full_attention_interval`` says 1 in 4 is) and
``qwen3.6-35b-a3b-reap48-mtp-trap`` (R3 + R6: the MTP layer counted as a
serving block). geometry.py has no hybrid handling; the fix is a real
assay change, not a test change.

Three quantities the vectors state have no assay surface to assert
against at all yet — ``attention_layers`` and ``serving_block_count``
(R3/R6) and ``recurrent_state_bytes`` (R4). They are absent from this
suite rather than asserted against an invented field name; both R3 and
R6 still surface here indirectly, as the kv figure they distort.
"""

import hashlib
import importlib.util
import json
import pathlib

import pytest

from assay.backends.ollama import OllamaNative
from assay.geometry import kv_bytes_per_token, plan_window

DATA = pathlib.Path(__file__).resolve().parent / "data" / "gguf_geometry_v1"

VENDORED_MANIFEST_SHA = (
    "d50e7c36ea714cfc837fc9bc0f8b3d1a573bb3a031dd7fbe776e3a252ee29ddc"
)
"""sha256 of the vendored MANIFEST.json, pinned at vendoring.

Re-vendoring a newer vector set is a deliberate act with a visible diff
(this constant changes), never something that happens by a stray copy.
"""

_MIB = 1024 * 1024

_LIMITED_BY = {
    # The contract's term names (R7) -> assay's name for the same term.
    # assay calls the accelerator-budget term "vram"; the contract calls
    # it "budget" because a consumer's budget need not be VRAM.
    "budget": "vram",
    "training_ctx": "training_ctx",
    "user_cap": "user_cap",
}


def vectors() -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted(DATA.glob("*.json"))
        if path.name != "MANIFEST.json"
    ]


VECTORS = vectors()


def vector_backend(vec: dict, *, loaded: bool = False) -> OllamaNative:
    """`OllamaNative` serving one vector's metadata as an /api/show body.

    /api/tags replays empty (weights size is not a term any vector
    states); /api/ps replays the residency the caller forces, because
    the window scenarios all state ``weights_bytes: 0`` and assay reaches
    that only through the residency rule.
    """
    name = vec["id"]

    def http_post(url, payload):
        assert url.endswith("/api/show") and payload == {"model": name}
        return 200, {"model_info": vec["metadata"]}

    def http_get(url):
        if url.endswith("/api/tags"):
            return 200, {"models": []}
        assert url.endswith("/api/ps")
        return 200, {"models": [{"model": name}] if loaded else []}

    return OllamaNative(
        "http://vectors", name, http_post=http_post, http_get=http_get
    )


def test_vendored_manifest_is_pinned():
    actual = hashlib.sha256((DATA / "MANIFEST.json").read_bytes()).hexdigest()
    assert actual == VENDORED_MANIFEST_SHA


def test_the_vendored_copy_is_the_set_the_manifest_names():
    """Every vendored file matches its manifest sha, and none is missing.

    The manifest pin above catches an edited manifest; this catches a
    half-copied set, an edited vector, or an extra file smuggled in
    beside the frozen ones.
    """
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    assert manifest["set_version"] == "v1"
    on_disk = {path.name for path in DATA.glob("*.json")} - {"MANIFEST.json"}
    assert on_disk == set(manifest["files"])
    for name, expected_sha in manifest["files"].items():
        actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        assert actual == expected_sha, f"vendored {name} is not the frozen file"


@pytest.mark.parametrize("vec", VECTORS, ids=lambda v: v["id"])
def test_kv_interpretation_conforms(vec):
    info = vector_backend(vec).model_info()
    kv = kv_bytes_per_token(info)

    if vec["expected"].get("refuses"):
        assert kv is None, "R8: unknown geometry must refuse, not guess"
        return

    banned = vec.get("must_not_equal", {}).get("kv_bytes_per_token", [])
    assert kv not in banned, (
        f"banned historical value: {vec.get('must_not_equal', {}).get('note')}"
    )
    assert kv == vec["expected"]["kv_bytes_per_token"]


@pytest.mark.parametrize(
    "vec",
    [v for v in VECTORS if not v["expected"].get("refuses")],
    ids=lambda v: v["id"],
)
def test_expert_fields_conform(vec):
    """R5: absent expert keys are None, never 0 — a dense model is not a
    0-expert MoE."""
    info = vector_backend(vec).model_info()
    experts = vec["expected"]["experts"]

    if experts is None:
        assert info.expert_count is None
        assert info.expert_used_count is None
        return
    assert info.expert_count == experts["count"]
    assert info.expert_used_count == experts["used"]


def window_scenarios():
    for vec in VECTORS:
        for index, scenario in enumerate(vec["expected"].get("windows", [])):
            yield pytest.param(
                vec,
                scenario,
                id=f"{vec['id']}-{index}-{scenario['limited_by']}",
            )


@pytest.mark.parametrize("vec,scenario", list(window_scenarios()))
def test_window_law_conforms(vec, scenario):
    """R7: the window law over the terms each scenario states.

    Term mapping onto assay's ``plan_window``: the contract's
    ``budget_bytes`` (free accelerator bytes) is assay's
    ``vram_free_mib``, and ``fixed_overhead_bytes`` is its
    ``overhead_mib`` — assay subtracts the second from the first, which
    is the scenarios' own arithmetic. ``weights_bytes: 0`` is reached
    through the residency rule (a loaded model's weights are already
    outside free VRAM), so the backend replays the model as resident.
    """
    terms = scenario["terms"]
    assert terms["weights_bytes"] == 0, "mapping assumes a resident model"
    assert terms["budget_bytes"] % _MIB == 0, "budget must be MiB-exact"
    assert terms["fixed_overhead_bytes"] % _MIB == 0, "overhead must be MiB-exact"

    info = vector_backend(vec, loaded=True).model_info()
    # The scenario's stated inputs must be what assay itself read, or the
    # window number below would be right or wrong for the wrong reason.
    assert info.loaded is True
    assert info.training_ctx == terms["training_ctx"]
    assert kv_bytes_per_token(info) == terms["kv_bytes_per_token"]

    geometry = plan_window(
        info,
        vram_free_mib=terms["budget_bytes"] // _MIB,
        user_cap=terms["user_cap"],
        overhead_mib=terms["fixed_overhead_bytes"] // _MIB,
    )

    assert geometry is not None
    assert geometry.usable_window == scenario["usable_window"], scenario["note"]
    assert geometry.limited_by == _LIMITED_BY[scenario["limited_by"]]


def test_the_suite_actually_covers_the_frozen_set():
    """A conformance suite that silently collected nothing would pass.

    Pins the counts the vendored set carries: 10 vectors, one of them the
    R8 refusal, and the three window scenarios. If a re-vendor changes
    them, this fails and the numbers are re-read deliberately.
    """
    assert len(VECTORS) == 10
    assert sum(1 for v in VECTORS if v["expected"].get("refuses")) == 1
    assert len(list(window_scenarios())) == 3


def test_the_mutation_harness_still_aims_at_real_lines():
    """`scripts/mutate_geometry_conformance.py` must not rot silently.

    That script is the evidence that the assertions above are
    load-bearing, and it finds its targets by exact string. A refactor
    that reworded one of those lines would leave a harness whose anchors
    no longer match — discovered only when someone next runs it, long
    after the assertions stopped being proven. This is the same check the
    harness makes at run time, paid for on every suite run instead.

    Loaded from its path, not imported: `scripts/` is not a package, the
    same pin `test_matrix_build.py` keeps.
    """
    repo = pathlib.Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "mutate_geometry_conformance.py"
    spec = importlib.util.spec_from_file_location("mutate_conformance", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cases = module.cases(repo)
    assert len(cases) == 10, "a mutant was added or dropped — say so deliberately"
    for label, path, original, mutant, selection in cases:
        assert path.is_file(), f"{label}: target {path} is gone"
        assert path.read_text().count(original) == 1, (
            f"{label}: anchor no longer occurs exactly once in {path.name} — "
            "the mutation would be unattributable"
        )
        assert original != mutant, f"{label}: mutant is identical to the original"
        assert all(
            item.startswith(f"tests/{pathlib.Path(__file__).name}::")
            for item in selection
        ), f"{label}: selection has drifted off this module"
