"""gguf-geometry v2 conformance: assay's extraction against the contract.

The vectors under ``tests/data/gguf_geometry_v2/`` are a vendored,
sha-pinned copy of the current v2 set, taken byte-for-byte from
gguf-geometry master ``7f858c8`` on 2026-08-27 (rules R1-R8 in that
repo's SPEC.md). Every ``expected`` value in them was measured on real
hardware and cited there; nothing in this file computes geometry, and
nothing here may be edited to make a red go green — a red is a
divergence between assay and the contract, i.e. a work order.

One set is vendored, not a pile of them: that repo's consumer model is a
single current directory, and v2 contains v1's content — eight of v1's
ten vectors carried forward with identical shas, two (gemma-4,
deepseek-coder-v2) stating metadata keys their v1 copies cited but
omitted with no ``expected`` value moved, plus the eleventh,
``qwen3.8-27b``. So ``tests/data/gguf_geometry_v1/`` was deleted in the
same commit that added this set; nothing it asserted stopped being
asserted.

``qwen3.8-27b`` is the case v1 withheld, and it is this repository's own
defect coming back as a contract: the published 260 KiB/token figure it
bans (266240 bytes, all 65 raw blocks charged) is what
``docs/superpowers/evidence/tier-enthusiast-2026-08/`` carried until
erratum E2, and its ``expected`` values are what assay's fixed extractor
read live off the daemon holding that blob
(``docs/superpowers/evidence/qwen38-27b-live-2026-08-27/``). It passed
on the first run of this suite against the v2 set, with no change to any
production module — the vector arriving from outside and landing on the
numbers assay already produces is the point of vendoring it.

Each vector's ``metadata`` block IS the ``model_info`` object Ollama's
/api/show returns, so the code under test is assay's own interpretation
seam, ``OllamaNative.model_info()``, followed by ``geometry.py``. A
metadata-to-``ModelInfo`` mapping written here instead would be a fourth
implementation of the contract, and this suite would then be measuring
the test rather than assay. No daemon, no GPU, no network: the transport
is replaced, the interpretation is not.

Reds at authoring time (2026-08-27), both hybrid vectors:
``qwen3.6-35b-a3b-reap48-ours-q4km`` (R3: every block charged as an
attention layer where ``full_attention_interval`` says 1 in 4 is) and
``qwen3.6-35b-a3b-reap48-mtp-trap`` (R3 + R6: the MTP layer counted as a
serving block). Both were divergences in ``geometry.py``, which had no
hybrid handling at all, and both were closed by a real assay change —
``ModelInfo.attention_layer_count`` / ``mtp_layer_count`` /
``recurrent_state_bytes`` and the rules behind them — never by a test
change.

The three quantities that had no assay surface to assert against when
this module was written — ``attention_layers`` and
``serving_block_count`` (R3/R6) and ``recurrent_state_bytes`` (R4) —
have one now and are asserted directly below, on the three vectors whose
evidence states them (two under v1, and ``qwen3.8-27b`` as of v2).
"""

import hashlib
import importlib.util
import json
import pathlib

import pytest

from assay.backends.ollama import OllamaNative
from assay.geometry import kv_bytes_per_token, plan_window, serving_block_count

DATA = pathlib.Path(__file__).resolve().parent / "data" / "gguf_geometry_v2"

VENDORED_MANIFEST_SHA = (
    "06da801b5dc57fedbfd42555c377c1cd3b6b8fb3c549cd2d367396447fc15116"
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
    assert manifest["set_version"] == "v2"
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


def stating(field: str) -> list:
    """The vectors whose ``expected`` block states ``field``.

    A vector states a quantity only where its evidence measured one, so
    each rule below is asserted exactly on the corpus that carries it —
    never on a default standing in for a measurement.
    """
    return [v for v in VECTORS if field in v["expected"]]


@pytest.mark.parametrize("vec", stating("attention_layers"), ids=lambda v: v["id"])
def test_attention_layer_count_conforms(vec):
    """R3: the layers that own a kv cache, never the raw block count."""
    info = vector_backend(vec).model_info()

    assert info.attention_layer_count == vec["expected"]["attention_layers"]


@pytest.mark.parametrize("vec", stating("serving_block_count"), ids=lambda v: v["id"])
def test_serving_block_count_conforms(vec):
    """R6: MTP layers are counted by the converter and do not serve."""
    info = vector_backend(vec).model_info()
    serving = serving_block_count(info.block_count, info.mtp_layer_count)

    banned = vec.get("must_not_equal", {}).get("serving_block_count", [])
    assert serving not in banned, (
        f"banned historical value: {vec.get('must_not_equal', {}).get('note')}"
    )
    assert serving == vec["expected"]["serving_block_count"]


@pytest.mark.parametrize("vec", stating("recurrent_state_bytes"), ids=lambda v: v["id"])
def test_recurrent_state_conforms(vec):
    """R4: the ssm term is charged per context, never defaulted to 0."""
    info = vector_backend(vec).model_info()

    assert info.recurrent_state_bytes == vec["expected"]["recurrent_state_bytes"]


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

    Pins the counts the vendored set carries: 11 vectors, one of them the
    R8 refusal, and the three window scenarios. If a re-vendor changes
    them, this fails and the numbers are re-read deliberately. The v1->v2
    re-vendor is exactly that event: 10 -> 11 vectors and 2 -> 3 stating
    each hybrid rule, all of it the one new ``qwen3.8-27b``, whose
    scenarios add no window (v2 states none for it; R7 is still carried
    by ``qwen2.5-coder-7b-instruct-q8_0`` alone).

    The per-rule selections are pinned for the same reason: each of R3,
    R4 and R6 is stated by exactly the three hybrid vectors, and a
    selection that silently emptied would report green having asserted
    nothing about the rule it names.
    """
    assert len(VECTORS) == 11
    assert sum(1 for v in VECTORS if v["expected"].get("refuses")) == 1
    assert len(list(window_scenarios())) == 3
    assert len(stating("attention_layers")) == 3
    assert len(stating("serving_block_count")) == 3
    assert len(stating("recurrent_state_bytes")) == 3
    assert sum(
        1 for v in VECTORS if "serving_block_count" in v.get("must_not_equal", {})
    ) == 2


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
    assert len(cases) == 13, "a mutant was added or dropped — say so deliberately"
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
