"""Backward-compatible action-observation imports.

New code should import implementations from :mod:`src.observations` and use
:mod:`src.observation_catalog` for capability-aware construction.
"""

from .observation_catalog import (
    build_available_observations,
    build_default_observations,
)
from .observations import (
    BrewWaitTimerObservation,
    CupLayoutObservation,
    FilledCupTrayLayoutObservation,
    GaiwanLidClosureObservation,
    GaiwanToPitcherObservation,
    HandAccessoryObservation,
    LidOpenSmellObservation,
    TeaCanisterToLotusObservation,
    TwoHandHoldObservation,
    WarmCleanSequenceObservation,
)
from .observations._rules import LayoutEvaluation, PourEvaluation, PourStage

__all__ = [
    "BrewWaitTimerObservation",
    "CupLayoutObservation",
    "FilledCupTrayLayoutObservation",
    "GaiwanLidClosureObservation",
    "GaiwanToPitcherObservation",
    "HandAccessoryObservation",
    "LayoutEvaluation",
    "LidOpenSmellObservation",
    "PourEvaluation",
    "PourStage",
    "TeaCanisterToLotusObservation",
    "TwoHandHoldObservation",
    "WarmCleanSequenceObservation",
    "build_available_observations",
    "build_default_observations",
]
