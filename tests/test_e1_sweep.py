"""The E1 sweep, held to arithmetic rather than to prose.

Every claim the sweep filed (docs/superpowers/evidence/e1-sweep/) is
replayed here from the committed bytes: the verbatim /api/show bodies
and /api/tags entries drive assay's own extraction and window law, and
the corrections are checked against the committed profile FILES, never
against a transcription of their numbers. No daemon, no GPU, no
network.
"""

import json
import pathlib

import pytest

from dataclasses import replace

from assay.backends.ollama import OllamaNative, _arch_value
from assay.geometry import plan_window

EVIDENCE = pathlib.Path(__file__).resolve().parents[1] / (
    "docs/superpowers/evidence")
SWEEP = EVIDENCE / "e1-sweep"
RESULTS = json.loads((SWEEP / "results.json").read_text())


def sweep_backend(name: str) -> OllamaNative:
    """`OllamaNative` served the committed sweep captures for `name`.

    /api/show is the capture, verbatim; /api/tags replays the entry
    recorded in results.json in the same breath; /api/ps replays empty
    — residency at replay time is irrelevant, each check FORCES the
    residency its row's conditions name.
    """
    record = RESULTS["models"][name]
    show = json.loads((SWEEP / record["show"]).read_text())

    def http_post(url, payload):
        assert url.endswith("/api/show") and payload == {"model": name}
        return 200, show

    def http_get(url):
        if url.endswith("/api/tags"):
            return 200, {"models": [record["tags_entry"]]}
        assert url.endswith("/api/ps")
        return 200, {"models": []}

    return OllamaNative("http://sweep", name, http_post=http_post,
                        http_get=http_get)


def row_loaded(row: dict) -> bool:
    # The protocol's version-keyed replay condition (footnote 1): probe
    # 0.1.0 read geometry pre-load; 0.3.0+ post-load.
    return not row["probe_version"].startswith("0.1.")


def affected_rows() -> list[dict]:
    return [r for r in RESULTS["profiles"]
            if r["classification"] == "AFFECTED"]


def test_the_sweep_covers_every_committed_profile_exactly_once():
    """Scope completeness: a 24th profile must break this sweep's claim.

    The sweep says "every committed profile classified". If a profile
    is ever added without re-running the sweep, the claim silently
    rots — so the claim is recomputed from the tree.
    """
    on_disk = []
    for path in sorted(EVIDENCE.glob("**/*.json")):
        if path.is_relative_to(SWEEP):
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "assay_profile_version" in data:
            on_disk.append(str(path.relative_to(EVIDENCE)))
    swept = [r["file"] for r in RESULTS["profiles"]]
    assert sorted(swept) == sorted(on_disk)
    assert len(on_disk) == 23


@pytest.mark.parametrize("name", sorted(RESULTS["models"]),
                         ids=lambda n: n)
def test_the_model_classification_replays_from_the_committed_bytes(name):
    """The protocol's rule, recomputed — never trusted as a label."""
    record = RESULTS["models"][name]
    arch_info = json.loads(
        (SWEEP / record["show"]).read_text())["model_info"]

    stated = _arch_value(arch_info, "attention.key_length")
    embedding = _arch_value(arch_info, "embedding_length")
    head_count = _arch_value(arch_info, "attention.head_count")
    derived = embedding // head_count if embedding and head_count else None

    assert stated == record["stated_key_length"]
    assert derived == record["derived_head_dim"]
    if stated is None:
        expected = "UNAFFECTED-BY-CONSTRUCTION"
    elif stated == derived:
        expected = "UNAFFECTED"
    else:
        expected = "AFFECTED"
    assert record["classification"] == expected
    # The extraction the probe actually ships agrees with the raw read:
    # stated wins when present, the derivation only fills its absence.
    info = sweep_backend(name).model_info()
    assert info.head_dim == (stated if stated is not None else derived)


@pytest.mark.parametrize("row", affected_rows(), ids=lambda r: r["file"])
def test_each_affected_row_is_checked_against_the_profile_itself(row):
    """Identity gate + correction, against the committed FILE.

    Feeding the DERIVED head_dim through today's window law under the
    row's own conditions must reproduce the committed geometry exactly
    (that is what pins head_dim as the only thing that changed), and
    the STATED head_dim must land on the corrected numbers the sweep
    filed. The profile stands as written throughout.
    """
    profile = json.loads((EVIDENCE / row["file"]).read_text())
    geometry = profile["geometry"]
    assert profile["model"]["name"] == row["model"]
    for field in ("kv_kib_per_token", "usable_window", "limited_by",
                  "vram_free_mib"):
        assert geometry[field] == row["committed"][field], field

    record = RESULTS["models"][row["model"]]
    # Identity gate part 1: today's blob is the committed blob.
    assert record["tags_entry"]["size"] == profile["model"]["weights_bytes"]

    info = sweep_backend(row["model"]).model_info()
    loaded = row_loaded(row)
    conditions = dict(vram_free_mib=geometry["vram_free_mib"],
                      user_cap=None)

    derived_geo = plan_window(
        replace(info, head_dim=record["derived_head_dim"], loaded=loaded),
        **conditions)
    assert derived_geo.kv_kib_per_token == geometry["kv_kib_per_token"]
    assert derived_geo.usable_window == geometry["usable_window"]
    assert derived_geo.limited_by == geometry["limited_by"]

    corrected = plan_window(replace(info, loaded=loaded), **conditions)
    assert corrected.kv_kib_per_token == row["corrected"]["kv_kib_per_token"]
    assert corrected.usable_window == row["corrected"]["usable_window"]
    assert corrected.limited_by == row["corrected"]["limited_by"]

    # Direction, from the integers: over-promised rows cost MORE cache
    # and never gain window; the conservative row is the mirror.
    if record["direction"] == "window-over-promised":
        assert corrected.kv_kib_per_token > geometry["kv_kib_per_token"]
        assert corrected.usable_window <= geometry["usable_window"]
    else:
        assert record["direction"] == "window-conservative"
        assert corrected.kv_kib_per_token < geometry["kv_kib_per_token"]
        assert corrected.usable_window > geometry["usable_window"]


def test_the_sweep_found_both_signs_and_a_promise_that_held():
    """The three shapes E1 takes, pinned so none is quietly dropped.

    gemma2: the kv figure was wrong while the WINDOW held (training_ctx
    binds under both readings) — shortfall 0.0 is a real row, not a
    rounding artifact. mistral-nemo: stated < derived, the committed
    window UNDER-promised. codegemma: over-promised kv on a window that
    was already 0.
    """
    by_file = {r["file"]: r for r in affected_rows()}

    gemma2 = by_file["tier-enthusiast/gemma2-9b.json"]
    assert gemma2["committed"]["kv_kib_per_token"] == 294
    assert gemma2["corrected"]["kv_kib_per_token"] == 336
    assert (gemma2["corrected"]["usable_window"]
            == gemma2["committed"]["usable_window"] == 8192)
    assert gemma2["corrected"]["limited_by"] == "training_ctx"
    assert gemma2["corrected"][
        "window_shortfall_pct_of_committed_promise"] == 0.0

    nemo = by_file["tier-enthusiast/mistral-nemo-latest.json"]
    assert RESULTS["models"]["mistral-nemo:latest"]["direction"] == (
        "window-conservative")
    assert nemo["committed"]["kv_kib_per_token"] == 200
    assert nemo["corrected"]["kv_kib_per_token"] == 160
    assert nemo["committed"]["usable_window"] == 32711
    assert nemo["corrected"]["usable_window"] == 40889
    # Conservative rows publish no shortfall-of-promise: that ratio
    # names an over-promise and would be a value that looks like a
    # measurement of one.
    assert "window_shortfall_pct_of_committed_promise" not in nemo["corrected"]

    for directory in ("live", "live-run2"):
        codegemma = by_file[
            f"{directory}/codegemma-7b-instruct-q8_0-quick.json"]
        assert codegemma["committed"]["kv_kib_per_token"] == 336
        assert codegemma["corrected"]["kv_kib_per_token"] == 448
        assert (codegemma["corrected"]["usable_window"]
                == codegemma["committed"]["usable_window"] == 0)


def test_the_confirmed_erratum_rows_replayed_to_their_published_figures():
    """The sweep reproduces the two rows E1 was filed on — its known
    answers. A sweep that disagreed with the erratum it extends would
    be measuring something else."""
    by_file = {r["file"]: r for r in affected_rows()}
    deepseek = by_file[
        "tier-enthusiast/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json"]
    assert deepseek["corrected"]["kv_kib_per_token"] == 324
    assert deepseek["corrected"]["usable_window"] == 5394
    assert deepseek["corrected"][
        "window_shortfall_pct_of_committed_promise"] == 33.3
    qwen35 = by_file["tier-enthusiast/qwen3-8-27b.json"]
    assert qwen35["corrected"]["kv_kib_per_token"] == 260
    assert qwen35["corrected"]["usable_window"] == 4096
    assert qwen35["corrected"][
        "window_shortfall_pct_of_committed_promise"] == 16.8


def test_the_inconsistent_row_is_residency_not_a_wrong_profile():
    """The gate failure is real, its explanation is checked, and the
    row was not forced into a bucket.

    Pre-load replay does NOT reproduce the committed geometry (that IS
    the recorded failure); loaded=True reproduces it exactly; and the
    model states no key_length, so kv is identical under either
    head_dim source — the committed numbers carry no E1 error.
    """
    (row,) = [r for r in RESULTS["profiles"]
              if r["classification"] == "E1-INCONSISTENT"]
    assert row["file"] == "live-run2/qwen2.5-coder-7b-instruct-q8_0-quick.json"
    (investigation,) = RESULTS["investigations"]
    assert investigation["file"] == row["file"]
    assert investigation["finding"] == "residual-residency"

    profile = json.loads((EVIDENCE / row["file"]).read_text())
    geometry = profile["geometry"]
    record = RESULTS["models"][row["model"]]
    assert record["stated_key_length"] is None
    info = sweep_backend(row["model"]).model_info()
    assert info.head_dim == record["derived_head_dim"]
    conditions = dict(vram_free_mib=geometry["vram_free_mib"],
                      user_cap=None)

    pre_load = plan_window(replace(info, loaded=False), **conditions)
    assert pre_load.usable_window != geometry["usable_window"]

    resident = plan_window(replace(info, loaded=True), **conditions)
    assert resident.kv_kib_per_token == geometry["kv_kib_per_token"]
    assert resident.usable_window == geometry["usable_window"]
    assert resident.limited_by == geometry["limited_by"]


def test_no_geometry_is_a_named_absence_not_a_clean_bill():
    """None-vs-zero at the row level: the geometry-less profile is
    filed NO-GEOMETRY even though its MODEL shows the largest stated
    /derived gap the sweep found — there is simply no committed kv
    number for that gap to have corrupted."""
    (row,) = [r for r in RESULTS["profiles"]
              if r["classification"] == "NO-GEOMETRY"]
    assert row["file"] == "tier-enthusiast/gemma-4-12b-it-qat-q4_0-latest.json"
    profile = json.loads((EVIDENCE / row["file"]).read_text())
    assert profile["geometry"] is None
    record = RESULTS["models"][row["model"]]
    assert record["classification"] == "AFFECTED"
    assert record["stated_key_length"] == 512
    assert record["derived_head_dim"] == 240
    assert "corrected" not in row


def test_every_row_earned_its_verdict_through_the_identity_gate():
    for row in RESULTS["profiles"]:
        if row["classification"] in ("AFFECTED", "UNAFFECTED",
                                     "UNAFFECTED-BY-CONSTRUCTION"):
            assert row["identity_size_matches"] is True, row["file"]
            assert row["derivation_reproduces_committed"] is True, row["file"]


def test_the_errata_are_filed_beside_the_profiles_they_correct():
    """Reachability: each directory with a wrong number carries its own
    correction, and the amended finding names the corrected figures."""
    tier = (EVIDENCE / "tier-enthusiast/ERRATA.md").read_text()
    for name, corrected in (("gemma2-9b.json", "336"),
                            ("mistral-nemo-latest.json", "40889")):
        assert name in tier, name
        assert corrected in tier

    for directory in ("live", "live-run2"):
        errata = EVIDENCE / directory / "ERRATA.md"
        assert errata.exists()
        text = errata.read_text()
        assert "codegemma-7b-instruct-q8_0-quick.json" in text
        assert "448" in text

    validation = (EVIDENCE / "2026-08-12-live-validation.md").read_text()
    assert "448" in validation and "8×" in validation
