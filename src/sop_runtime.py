"""Runtime assembly of the configured SOP state machine."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .sop_config import build_sop_steps, load_sop_config
from .sop_state_machine import SopMode, SopStateMachine


DEFAULT_SOP_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "sop_red_tea_front_v1.yaml"
)


def build_sop_state_machine(
    *,
    config_path: str | Path = DEFAULT_SOP_CONFIG,
    mode: SopMode | str = SopMode.FREE_OBSERVATION,
    available_observation_ids: Iterable[str] | None = None,
    include_deferred: bool = False,
    include_disabled: bool = False,
    allow_empty: bool = False,
) -> SopStateMachine | None:
    """Build the state machine from YAML and the active observer ID set."""

    config = load_sop_config(config_path)
    steps = build_sop_steps(
        config,
        include_deferred=include_deferred,
        include_disabled=include_disabled,
        available_observation_ids=available_observation_ids,
        allow_empty=allow_empty,
    )
    if not steps:
        return None
    return SopStateMachine(steps, mode=mode)
