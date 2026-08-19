"""Central catalog and capability-aware factory for action observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .observation_runtime import CameraRole
from .observations import (
    BrewDurationObservation,
    BrewWaitTimerObservation,
    CupLayoutObservation,
    FilledCupTrayLayoutObservation,
    GaiwanLidClosureObservation,
    GaiwanToPitcherObservation,
    FrontWarmCleanSequenceObservation,
    HandAccessoryObservation,
    LidOpenSmellObservation,
    TeaCanisterToLotusObservation,
    TeaDistributionObservation,
    TeaLotusToGaiwanObservation,
    TeaWeightObservation,
    TwoHandServeTrayObservation,
    TwoHandHoldObservation,
    WarmCleanSequenceObservation,
    WaterInjectionObservation,
    WaterTemperatureObservation,
    SetupReadyObservation,
)
from .observations.sop_v2 import (
    LotusAppreciationObservation,
    RelaxedLidClosureObservation,
    ReturnAwareDecantObservation,
    ReturnAwareDistributionObservation,
    SimpleWaterInjectionObservation,
    SmellObservation,
    TeaPreparationObservation,
    TeaTransferObservation,
)


AvailabilityCheck = Callable[[set[str], bool], bool]


@dataclass(frozen=True)
class ObservationSpec:
    """Registration metadata for one independently runnable observer."""

    observation_id: str
    domain: str
    factory: Callable[[], Any]
    camera_roles: frozenset[CameraRole]
    is_available: AvailabilityCheck


def _classes_required(check: Callable[[set[str]], bool]) -> AvailabilityCheck:
    return lambda classes, accessory_configured: check(classes)


def _has_all(*names: str) -> AvailabilityCheck:
    required = set(names)
    return lambda classes, accessory_configured: required <= classes


def _has_gaiwan_parts(classes: set[str], accessory_configured: bool) -> bool:
    body_names = {"盖碗碗身", "盖碗（碗身）"}
    lid_names = {"盖碗碗盖", "盖碗（碗盖）"}
    return bool(body_names & classes) and bool(lid_names & classes)


def _has_brew_composite(classes: set[str], accessory_configured: bool) -> bool:
    return (
        GaiwanLidClosureObservation.required_classes_available(classes)
        and GaiwanToPitcherObservation.required_classes_available(classes)
    )


OBSERVATION_SPECS = (
    ObservationSpec(
        SetupReadyObservation.observation_id,
        "setup",
        SetupReadyObservation,
        frozenset(SetupReadyObservation.camera_roles),
        _classes_required(SetupReadyObservation.required_classes_available),
    ),
    ObservationSpec(
        FrontWarmCleanSequenceObservation.observation_id,
        "warm_clean",
        FrontWarmCleanSequenceObservation,
        frozenset(FrontWarmCleanSequenceObservation.camera_roles),
        _classes_required(FrontWarmCleanSequenceObservation.required_classes_available),
    ),
    ObservationSpec(
        WaterTemperatureObservation.observation_id,
        "ocr",
        WaterTemperatureObservation,
        frozenset(WaterTemperatureObservation.camera_roles),
        _has_all("水壶显示屏"),
    ),
    ObservationSpec(
        WarmCleanSequenceObservation.observation_id,
        "warm_clean",
        WarmCleanSequenceObservation,
        frozenset(WarmCleanSequenceObservation.camera_roles),
        _classes_required(WarmCleanSequenceObservation.required_classes_available),
    ),
    ObservationSpec(
        CupLayoutObservation.observation_id,
        "layout",
        CupLayoutObservation,
        frozenset(CupLayoutObservation.camera_roles),
        _has_all("品茗杯"),
    ),
    ObservationSpec(
        FilledCupTrayLayoutObservation.observation_id,
        "layout",
        FilledCupTrayLayoutObservation,
        frozenset(FilledCupTrayLayoutObservation.camera_roles),
        _has_all("品茗杯", "茶盘"),
    ),
    ObservationSpec(
        TeaCanisterToLotusObservation.observation_id,
        "tea_preparation",
        TeaPreparationObservation,
        frozenset(TeaPreparationObservation.camera_roles),
        _has_all("茶叶罐", "茶荷"),
    ),
    ObservationSpec(
        TeaWeightObservation.observation_id,
        "ocr",
        TeaWeightObservation,
        frozenset(TeaWeightObservation.camera_roles),
        _has_all("电子秤显示屏"),
    ),
    ObservationSpec(
        LotusAppreciationObservation.observation_id,
        "tea_preparation",
        LotusAppreciationObservation,
        frozenset(LotusAppreciationObservation.camera_roles),
        _has_all("茶荷"),
    ),
    ObservationSpec(
        SmellObservation.observation_id,
        "smell",
        SmellObservation,
        frozenset(SmellObservation.camera_roles),
        _has_gaiwan_parts,
    ),
    ObservationSpec(
        TeaTransferObservation.observation_id,
        "tea_transfer",
        TeaTransferObservation,
        frozenset(TeaTransferObservation.camera_roles),
        lambda classes, accessory_configured: (
            "茶荷" in classes and bool({"盖碗碗身", "盖碗（碗身）"} & classes)
        ),
    ),
    ObservationSpec(
        SimpleWaterInjectionObservation.observation_id,
        "brewing",
        SimpleWaterInjectionObservation,
        frozenset(SimpleWaterInjectionObservation.camera_roles),
        _classes_required(SimpleWaterInjectionObservation.required_classes_available),
    ),
    ObservationSpec(
        RelaxedLidClosureObservation.observation_id,
        "brewing",
        RelaxedLidClosureObservation,
        frozenset(RelaxedLidClosureObservation.camera_roles),
        _classes_required(RelaxedLidClosureObservation.required_classes_available),
    ),
    ObservationSpec(
        ReturnAwareDecantObservation.observation_id,
        "brewing",
        ReturnAwareDecantObservation,
        frozenset(ReturnAwareDecantObservation.camera_roles),
        _classes_required(ReturnAwareDecantObservation.required_classes_available),
    ),
    ObservationSpec(
        BrewWaitTimerObservation.observation_id,
        "brewing",
        BrewWaitTimerObservation,
        frozenset(BrewWaitTimerObservation.camera_roles),
        _has_brew_composite,
    ),
    ObservationSpec(
        BrewDurationObservation.observation_id,
        "brewing",
        BrewDurationObservation,
        frozenset(BrewDurationObservation.camera_roles),
        lambda classes, accessory_configured: (
            "烧水壶" in classes
            and _has_brew_composite(classes, accessory_configured)
        ),
    ),
    ObservationSpec(
        ReturnAwareDistributionObservation.observation_id,
        "serving",
        ReturnAwareDistributionObservation,
        frozenset(ReturnAwareDistributionObservation.camera_roles),
        _classes_required(ReturnAwareDistributionObservation.required_classes_available),
    ),
    ObservationSpec(
        TwoHandServeTrayObservation.observation_id,
        "serving",
        TwoHandServeTrayObservation,
        frozenset(TwoHandServeTrayObservation.camera_roles),
        _has_all("茶盘"),
    ),
    ObservationSpec(
        HandAccessoryObservation.observation_id,
        "hand_compliance",
        HandAccessoryObservation,
        frozenset(HandAccessoryObservation.camera_roles),
        lambda classes, accessory_configured: accessory_configured,
    ),
)


def observation_specs() -> tuple[ObservationSpec, ...]:
    return OBSERVATION_SPECS


def registered_observation_ids() -> frozenset[str]:
    return frozenset(spec.observation_id for spec in OBSERVATION_SPECS)


def build_default_observations() -> list[Any]:
    """Instantiate every registered observer for simulation and regression tests."""

    return [spec.factory() for spec in OBSERVATION_SPECS]


def build_available_observations(
    model_classes: Iterable[str],
    camera_role: CameraRole | str,
    accessory_configured: bool = False,
) -> list[Any]:
    """Instantiate observers supported by the model, camera and optional modules."""

    classes = {str(value) for value in model_classes}
    role = CameraRole(camera_role)
    observations: list[Any] = []
    for spec in OBSERVATION_SPECS:
        if not _role_matches(role, spec.camera_roles):
            continue
        if spec.is_available(classes, accessory_configured):
            observations.append(spec.factory())
    return observations


def _role_matches(
    selected: CameraRole, supported: frozenset[CameraRole]
) -> bool:
    if selected in {CameraRole.SINGLE, CameraRole.FRONT}:
        return bool(
            supported & {CameraRole.FRONT, CameraRole.TABLETOP, CameraRole.SIDE}
        )
    return selected in supported
