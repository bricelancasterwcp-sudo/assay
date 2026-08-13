"""Probe orchestrator (spec §9, plan Task 11).

Order: detect/build backend (recorded when asked) -> model_info ->
geometry (pure, no model calls) -> calibrate -> ceiling -> envelope ->
codecs -> verdicts -> Profile.

Budget discipline: ``budget`` has NO default — a library consumer
burning a user's GPU time must say how much (spec §9). Any probe
family hitting BudgetExhausted stops the remaining families; every
unfinished family is None and named in ``provenance.dropped`` — never
partial-pretending-complete. If the budget dies before ANY family
completed, BudgetExhausted propagates (the CLI maps it to exit 2).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from assay import __version__
from assay.backends import Backend, detect_backend
from assay.backends.base import PROBE_TEMPERATURE
from assay.backends.ollama import OllamaNative
from assay.backends.openai_compat import OpenAICompat
from assay.budget import Budget, BudgetMeter
from assay.ceiling import Calibration, Ceiling, calibrate, probe_ceiling
from assay.codecs import probe_codecs
from assay.envelope import probe_envelope
from assay.errors import BudgetExhausted
from assay.geometry import free_vram_mib, plan_window
from assay.profile import PROFILE_VERSION, Profile, compute_verdicts
from assay.replay import CallRecorder


@dataclass(frozen=True)
class ModeParams:
    seeds: tuple[int, ...]
    envelope_n: int
    codecs_n_per_cell: int


MODE_PARAMS = {
    "quick": ModeParams(seeds=(0,), envelope_n=10, codecs_n_per_cell=5),
    "full": ModeParams(seeds=(0, 1), envelope_n=30, codecs_n_per_cell=10),
}

_QUICK_CEILING_CAP = 16384
_FULL_CEILING_CAP = 32768


def ceiling_cap_for(
    mode: str, training_ctx: int | None, window_cap: int | None
) -> int:
    """The ladder cap: mode table, then the user's cap if tighter."""
    if mode == "quick":
        cap = _QUICK_CEILING_CAP
    else:
        cap = min(training_ctx or _FULL_CEILING_CAP, _FULL_CEILING_CAP)
    if window_cap is not None:
        cap = min(cap, window_cap)
    return cap


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _endpoint_kind(live: Backend, forced: str | None) -> str:
    if forced is not None:
        return forced
    if isinstance(live, OllamaNative):
        return "ollama"
    if isinstance(live, OpenAICompat):
        return "openai"
    return "injected"  # a test override: named honestly, never guessed


def _calibration_payload(calibration: Calibration | None) -> dict | None:
    if calibration is None:
        return None
    return {
        "chars_per_token": calibration.chars_per_token,
        "counts_available": calibration.counts_available,
        "deterministic": calibration.deterministic,
    }


def probe(
    base_url: str,
    model: str,
    *,
    budget: Budget,
    mode: str = "quick",
    backend: str | None = None,
    record: Path | None = None,
    window_cap: int | None = None,
    _backend_override: Backend | None = None,
) -> Profile:
    """Run the full probe suite against one endpoint; return a Profile.

    ``backend`` forces the kind ("ollama" | "openai"); else auto-detect.
    ``record`` wraps the backend in a CallRecorder writing a transcript.
    ``window_cap`` is the user's context cap: it bounds both geometry's
    user_cap term and the ceiling ladder (spec §5).
    """
    if mode not in MODE_PARAMS:
        raise ValueError(f"unknown mode: {mode!r} (expected 'quick' or 'full')")
    params = MODE_PARAMS[mode]
    started = _utc_now()

    autodetected = backend is None and _backend_override is None
    live = (
        _backend_override
        if _backend_override is not None
        else detect_backend(base_url, model, forced=backend)
    )
    kind = _endpoint_kind(live, backend)
    active: Backend = CallRecorder(live, Path(record)) if record is not None else live
    meter = BudgetMeter(budget)
    dropped: list[str] = []

    info = active.model_info()  # InfrastructureError propagates (spec §3)

    # Geometry is pure arithmetic over metadata: no model calls, no budget.
    geometry = plan_window(
        info, vram_free_mib=free_vram_mib(), user_cap=window_cap
    )
    if geometry is None:
        dropped.append(
            "geometry: kv arithmetic or training_ctx unavailable "
            f"(source={info.source})"
        )

    calibration: Calibration | None = None
    ceiling: Ceiling | None = None
    envelope = None
    codecs = None
    budget_death: BudgetExhausted | None = None

    try:
        calibration = calibrate(active, meter, seed=params.seeds[0])
        ceiling = probe_ceiling(
            active,
            meter,
            cap_tokens=ceiling_cap_for(mode, info.training_ctx, window_cap),
            seeds=params.seeds,
            calibration=calibration,
        )
    except BudgetExhausted as exc:
        budget_death = exc
    if ceiling is None:
        dropped.append("ceiling: budget exhausted before any ladder call")
        if calibration is None:
            dropped.append("calibration: budget exhausted before completion")
    elif any(entry.signal == "budget" for entry in ceiling.evidence):
        # The partial ceiling is kept (it reports what it verified), but
        # the meter is dry: no further family may start.
        budget_death = BudgetExhausted("budget exhausted during the ceiling ladder")
    if ceiling is not None and not active.caps.per_request_ctx:
        # The evidence class is weaker and must be stated (spec §5/§11):
        # without options.num_ctx the ladder measured the server's own
        # configured context window, not a per-probe widened one.
        dropped.append(
            "ceiling: per_request_ctx unavailable — ladder measured the "
            "server's configured context window"
        )

    if budget_death is not None:
        dropped.append("envelope: skipped, budget exhausted earlier")
    else:
        envelope = probe_envelope(active, meter, n=params.envelope_n)
        if envelope.n < params.envelope_n:
            budget_death = BudgetExhausted("budget exhausted during envelope probes")
        if envelope.n == 0:
            # Zero completed probes is no measurement at all: None, never
            # a fidelity that looks measured (spec §8).
            envelope = None
            dropped.append("envelope: budget exhausted before any probe completed")

    if budget_death is not None:
        dropped.append("codecs: skipped, budget exhausted earlier")
    else:
        codecs = probe_codecs(active, meter, n_per_cell=params.codecs_n_per_cell)
        if all(
            cell.n == 0 for grades in codecs.values() for cell in grades.values()
        ):
            codecs = None
            dropped.append("codecs: budget exhausted before any probe completed")
        else:
            # Spec §8 None rule: every UNMEASURED cell (n == 0) is named
            # in dropped — a Landing(None, 0) with an empty dropped list
            # would hide the budget death from consumers. Cells with
            # n > 0 are measurements at their honest n and stay.
            for codec, grades in codecs.items():
                for grade, cell in grades.items():
                    if cell.n == 0:
                        dropped.append(
                            f"codecs: {codec}.{grade} budget exhausted "
                            "before any probe completed"
                        )
            if any(
                cell.n < params.codecs_n_per_cell
                for grades in codecs.values()
                for cell in grades.values()
            ):
                budget_death = BudgetExhausted("budget exhausted during codec probes")

    if (
        budget_death is not None
        and geometry is None
        and ceiling is None
        and envelope is None
        and codecs is None
    ):
        raise budget_death  # nothing completed: the caller must know (exit 2)

    return Profile(
        assay_profile_version=PROFILE_VERSION,
        probe_version=__version__,
        endpoint={
            "kind": kind,
            "base_url": base_url,
            "autodetected": autodetected,
        },
        model={
            "name": info.name,
            "quant": info.quant,
            "weights_bytes": info.weights_bytes,
            "training_ctx": info.training_ctx,
        },
        geometry=geometry,
        ceiling=ceiling,
        envelope=envelope,
        codecs=codecs,
        verdicts=compute_verdicts(geometry, ceiling, envelope, codecs),
        provenance={
            "started": started,
            "finished": _utc_now(),
            "mode": mode,
            "temperature": PROBE_TEMPERATURE,
            "seeds": list(params.seeds),
            "budget": {
                "max_calls": budget.max_calls,
                "max_prompt_tokens": budget.max_prompt_tokens,
            },
            "spent": {
                "calls": meter.spent.calls,
                "prompt_tokens": meter.spent.prompt_tokens,
            },
            "calibration": _calibration_payload(calibration),
            "dropped": dropped,
        },
    )
