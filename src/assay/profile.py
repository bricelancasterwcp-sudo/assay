"""Versioned capability profile, verdicts, render (spec §8).

One JSON document; every field a measurement, a None named in
``provenance.dropped``, or provenance. The schema is self-policing:
constructing a Profile with a None family that ``dropped`` does not
name is a ValueError — unmeasured must always be named, never silent.

None-vs-zero at the verdict layer: unmeasured inputs yield
``"unmeasured"``, never ``"unusable"`` — a consumer must be able to
tell "assay could not measure this" from "the model failed".
"""

import dataclasses
import json
from dataclasses import dataclass

from assay.ceiling import CallEvidence, Ceiling
from assay.codecs import Landing
from assay.envelope import Envelope
from assay.geometry import Geometry

PROFILE_VERSION = 2

_FAMILIES = ("geometry", "ceiling", "envelope", "codecs")
_GRADE_FOR_VERDICTS = "small"
_READY_THRESHOLD = 0.9
_RISKY_THRESHOLD = 0.6
_LONG_CONTEXT_TOKENS = 16384
_TRUNCATION_GUARD_TOKENS = 4096
_HONEST_MODES = frozenset({"hard_error", "none_up_to_cap"})
_LYING_MODES = frozenset({"silent_truncation", "missing_stats"})


@dataclass(frozen=True)
class Profile:
    assay_profile_version: int  # == PROFILE_VERSION
    probe_version: str
    endpoint: dict  # {"kind", "base_url", "autodetected"}
    model: dict  # {"name", "quant", "weights_bytes", "training_ctx"}
    geometry: Geometry | None
    ceiling: Ceiling | None
    envelope: Envelope | None
    codecs: dict[str, dict[str, Landing]] | None
    verdicts: dict[str, dict]
    provenance: dict  # started/finished/mode/seeds/budget/spent/calibration/dropped

    def __post_init__(self) -> None:
        dropped = self.provenance.get("dropped") or []
        named = {str(entry).split(":", 1)[0].strip() for entry in dropped}
        for family in _FAMILIES:
            if getattr(self, family) is None and family not in named:
                raise ValueError(
                    f"{family} is None but provenance.dropped does not name it"
                )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, payload: dict) -> "Profile":
        return cls(
            assay_profile_version=payload["assay_profile_version"],
            probe_version=payload["probe_version"],
            endpoint=payload["endpoint"],
            model=payload["model"],
            geometry=_geometry_from(payload["geometry"]),
            ceiling=_ceiling_from(payload["ceiling"]),
            envelope=_envelope_from(payload["envelope"]),
            codecs=_codecs_from(payload["codecs"]),
            verdicts=payload["verdicts"],
            provenance=payload["provenance"],
        )


def _geometry_from(payload: dict | None) -> Geometry | None:
    return None if payload is None else Geometry(**payload)


def _ceiling_from(payload: dict | None) -> Ceiling | None:
    if payload is None:
        return None
    data = dict(payload)
    data["evidence"] = tuple(CallEvidence(**entry) for entry in data["evidence"])
    return Ceiling(**data)


def _envelope_from(payload: dict | None) -> Envelope | None:
    return None if payload is None else Envelope(**payload)


def _codecs_from(
    payload: dict | None,
) -> dict[str, dict[str, Landing]] | None:
    if payload is None:
        return None
    return {
        codec: {grade: Landing(**cell) for grade, cell in grades.items()}
        for codec, grades in payload.items()
    }


def _small_landing(
    codecs: dict[str, dict[str, Landing]] | None, codec: str,
    *, lens: str = "byte_equality",
) -> float | None:
    """The .small landing rate under the named lens, or None wherever it
    was not measured."""
    if codecs is None:
        return None
    cell = codecs.get(codec, {}).get(_GRADE_FOR_VERDICTS)
    if cell is None:
        return None
    if lens == "applies_and_parses":
        return cell.lands_applies  # None when the cell exists but n == 0
    return cell.lands


def _truncates_below_4k(ceiling: Ceiling | None) -> bool:
    return (
        ceiling is not None
        and ceiling.failure_mode == "silent_truncation"
        and ceiling.first_failure is not None
        and ceiling.first_failure < _TRUNCATION_GUARD_TOKENS
    )


def _ladder(lands: float | None, *, ready_blocked: bool = False) -> str:
    if lands is None:
        return "unmeasured"
    if lands >= _READY_THRESHOLD and not ready_blocked:
        return "ready"
    if lands >= _RISKY_THRESHOLD:
        return "risky"
    return "unusable"


def _long_context(ceiling: Ceiling | None) -> str:
    if ceiling is None:
        return "unmeasured"
    if ceiling.failure_mode in _LYING_MODES:
        return "risky"  # the daemon lies past the edge
    if (
        ceiling.max_verified is not None
        and ceiling.max_verified >= _LONG_CONTEXT_TOKENS
        and ceiling.failure_mode in _HONEST_MODES
    ):
        return "ready"
    return "unmeasured"


def compute_verdicts(
    geometry: Geometry | None,
    ceiling: Ceiling | None,
    envelope: Envelope | None,
    codecs: dict[str, dict[str, Landing]] | None,
    *,
    presentation: str = "default-v1",
) -> dict[str, dict]:
    """Spec §8 verdict rules, v2: every verdict NAMES ITS LENS.

    The 2026-08-12 live validation measured the same model at 0% and
    100% edit landing under two instruments; a verdict quoted without
    its lens is not a model property. Each entry is
    ``{"verdict": ..., "lens": {...}}`` where the lens states the
    landing definition, the presentation (``default-v1`` or the
    consumer's ``custom`` directive), the pinned sampler, and — for
    long_context — the evidence class.

    ``patch_editing`` is judged under the **applies-and-parses** lens:
    an application accepting a patch validates the result by running
    it, so byte-equality's compliance-with-incidentals is the wrong
    predictor there. The raw byte-equality column stays in
    ``codecs`` for consumers who want the stricter number.

    Unmeasured inputs -> "unmeasured", never worse. ``geometry`` and
    ``envelope`` inform no verdict but are part of the stable signature.
    """
    del geometry, envelope  # no verdict consumes them
    sampler = {"temperature": 0.2}
    patch_rates = [
        rate
        for codec in ("search_replace", "whole_file")
        if (rate := _small_landing(codecs, codec,
                                   lens="applies_and_parses")) is not None
    ]
    counts = None if ceiling is None else ceiling.counts_available
    return {
        "structured_extraction": {
            "verdict": _ladder(
                _small_landing(codecs, "json_object"),
                ready_blocked=_truncates_below_4k(ceiling),
            ),
            "lens": {"landing": "json_valid_required_keys",
                     "presentation": presentation, **sampler},
        },
        "patch_editing": {
            "verdict": _ladder(max(patch_rates) if patch_rates else None),
            "lens": {"landing": "applies_and_parses",
                     "presentation": presentation, **sampler},
        },
        "long_context": {
            "verdict": _long_context(ceiling),
            "lens": {"evidence": ("counts+canary" if counts
                                  else "canary_only" if counts is not None
                                  else "unmeasured")},
        },
    }


def _show(value: object) -> str:
    return "unmeasured" if value is None else str(value)


def _render_geometry(geometry: Geometry | None) -> str:
    if geometry is None:
        return "geometry   unmeasured"
    vram = (
        f"{geometry.vram_free_mib} MiB free"
        if geometry.vram_free_mib is not None
        else "vram unmeasured"
    )
    return (
        f"geometry   {geometry.kv_kib_per_token} KiB/token"
        f" | usable window {geometry.usable_window}"
        f" (limited by {geometry.limited_by}) | {vram}"
    )


def _render_ceiling(ceiling: Ceiling | None) -> str:
    if ceiling is None:
        return "ceiling    unmeasured"
    return (
        f"ceiling    max verified {_show(ceiling.max_verified)}"
        f" | first failure {_show(ceiling.first_failure)}"
        f" | mode {ceiling.failure_mode}"
    )


def _render_envelope(envelope: Envelope | None) -> str:
    if envelope is None or envelope.fidelity is None:
        return "envelope   unmeasured"
    failures = envelope.failures
    return (
        f"envelope   fidelity {envelope.fidelity:.2f} (n={envelope.n})"
        f" | prose {failures.get('prose', 0)}"
        f" shape {failures.get('shape', 0)}"
        f" refusal {failures.get('refusal', 0)}"
    )


def _render_codecs(codecs: dict[str, dict[str, Landing]] | None) -> list[str]:
    if codecs is None:
        return ["codecs     unmeasured"]
    lines = ["codecs           " + "".join(g.ljust(12) for g in ("tiny", "small", "medium"))]
    for codec, grades in codecs.items():
        cells = []
        for grade in ("tiny", "small", "medium"):
            cell = grades.get(grade)
            if cell is None or cell.lands is None:
                cells.append("-".ljust(12))
            else:
                cells.append(f"{cell.lands:.2f} (n={cell.n})".ljust(12))
        lines.append(f"  {codec.ljust(15)}{''.join(cells)}")
    return lines


def render_table(profile: Profile) -> str:
    """Human view of a profile. Unmeasured is SAID, never shown as 0."""
    endpoint = profile.endpoint
    model = profile.model
    detected = "autodetected" if endpoint.get("autodetected") else "forced"
    lines = [
        f"assay profile v{profile.assay_profile_version}"
        f" (probe {profile.probe_version})",
        f"endpoint   {endpoint.get('kind')} {endpoint.get('base_url')} ({detected})",
        f"model      {model.get('name')}"
        f"  quant={_show(model.get('quant'))}"
        f"  training_ctx={_show(model.get('training_ctx'))}",
        "",
        _render_geometry(profile.geometry),
        _render_ceiling(profile.ceiling),
        _render_envelope(profile.envelope),
        *_render_codecs(profile.codecs),
        "",
        "verdicts   "
        + " | ".join(f"{name}: {entry['verdict']}"
                     for name, entry in profile.verdicts.items()),
        "lenses     "
        + " | ".join(
            f"{name}: " + ",".join(f"{k}={v}" for k, v in entry["lens"].items())
            for name, entry in profile.verdicts.items()),
    ]
    dropped = profile.provenance.get("dropped") or []
    if dropped:
        lines.append("dropped    " + "; ".join(str(entry) for entry in dropped))
    return "\n".join(lines)
