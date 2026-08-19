"""Load and validate the red-tea SOP runtime configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .sop_state_machine import SopStepConfig


DEFAULT_ACTIVE_AVAILABILITIES = frozenset(
    {"available", "experimental", "partial"}
)
DEFERRED_AVAILABILITIES = frozenset({"deferred", "deferred_optional"})
EXCLUDED_AVAILABILITIES = frozenset({"excluded_current_scope"})


class SopConfigError(ValueError):
    """Raised when an SOP YAML document is structurally invalid."""


def load_sop_config(path: str | Path) -> dict[str, Any]:
    """Read an SOP YAML file and validate all runtime references."""

    config_path = Path(path)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SopConfigError(f"cannot load SOP config {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SopConfigError("SOP config root must be a mapping")
    validate_sop_config(data)
    return data


def validate_sop_config(config: Mapping[str, Any]) -> None:
    """Validate business steps, runtime nodes and dependency order."""

    steps = _mapping_list(config.get("steps"), "steps")
    nodes = _mapping_list(config.get("runtime_nodes"), "runtime_nodes")
    if not steps:
        raise SopConfigError("steps must not be empty")
    if not nodes:
        raise SopConfigError("runtime_nodes must not be empty")

    step_ids: list[str] = []
    step_orders: list[int] = []
    for index, step in enumerate(steps):
        step_id = _required_text(step, "id", f"steps[{index}]")
        order = step.get("order")
        if not isinstance(order, int) or isinstance(order, bool):
            raise SopConfigError(f"steps[{index}].order must be an integer")
        step_ids.append(step_id)
        step_orders.append(order)
    _ensure_unique(step_ids, "business step id")
    if step_orders != sorted(step_orders) or len(step_orders) != len(set(step_orders)):
        raise SopConfigError("business step order values must be unique and ascending")

    node_ids: list[str] = []
    observation_ids: list[str] = []
    for index, node in enumerate(nodes):
        label = f"runtime_nodes[{index}]"
        node_id = _required_text(node, "node_id", label)
        observation_id = _required_text(node, "observation_id", label)
        business_step = _required_text(node, "business_step", label)
        availability = _required_text(node, "availability", label)
        runtime_enabled = node.get("runtime_enabled", True)
        if not isinstance(runtime_enabled, bool):
            raise SopConfigError(f"{label}.runtime_enabled must be a boolean")
        if business_step not in step_ids:
            raise SopConfigError(
                f"{label}.business_step references unknown step {business_step!r}"
            )
        if availability not in (
            DEFAULT_ACTIVE_AVAILABILITIES
            | DEFERRED_AVAILABILITIES
            | EXCLUDED_AVAILABILITIES
        ):
            raise SopConfigError(f"{label}.availability is unsupported: {availability}")
        prerequisites = node.get("prerequisites", [])
        if not isinstance(prerequisites, list) or not all(
            isinstance(value, str) and value for value in prerequisites
        ):
            raise SopConfigError(f"{label}.prerequisites must be a list of node IDs")
        node_ids.append(node_id)
        observation_ids.append(observation_id)
    _ensure_unique(node_ids, "runtime node id")
    _ensure_unique(observation_ids, "runtime observation id")

    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    for index, node in enumerate(nodes):
        for prerequisite in node.get("prerequisites", []):
            if prerequisite not in positions:
                raise SopConfigError(
                    f"runtime node {node_ids[index]!r} has unknown prerequisite "
                    f"{prerequisite!r}"
                )
            if positions[prerequisite] >= index:
                raise SopConfigError(
                    f"runtime node {node_ids[index]!r} prerequisite "
                    f"{prerequisite!r} must appear earlier"
                )


def build_sop_steps(
    config: Mapping[str, Any],
    *,
    include_deferred: bool = False,
    available_observation_ids: Iterable[str] | None = None,
    active_availabilities: Iterable[str] = DEFAULT_ACTIVE_AVAILABILITIES,
    allow_empty: bool = False,
    include_disabled: bool = False,
) -> list[SopStepConfig]:
    """Build state-machine steps from the filtered ``runtime_nodes`` list.

    By default only available, partial and experimental nodes are included.
    Explicit observation IDs form an additional allow-list. Missing
    prerequisites are removed when the YAML policy is ``remove_missing``.
    """

    validate_sop_config(config)
    active_statuses = set(active_availabilities)
    if include_deferred:
        active_statuses.update(DEFERRED_AVAILABILITIES)
    active_statuses.difference_update(EXCLUDED_AVAILABILITIES)
    observation_filter = (
        None
        if available_observation_ids is None
        else {str(value) for value in available_observation_ids}
    )

    selected: list[dict[str, Any]] = []
    for raw_node in config["runtime_nodes"]:
        node = deepcopy(dict(raw_node))
        if not include_disabled and not node.get("runtime_enabled", True):
            continue
        if node["availability"] not in active_statuses:
            continue
        if (
            observation_filter is not None
            and node["observation_id"] not in observation_filter
        ):
            continue
        selected.append(node)
    if not selected and not allow_empty:
        raise SopConfigError("no runtime nodes remain after capability filtering")

    selected_ids = {node["node_id"] for node in selected}
    business_steps = {
        str(step["id"]): dict(step) for step in config["steps"]
    }
    policy = config.get("runtime", {}).get(
        "missing_prerequisite_policy", "remove_missing"
    )
    steps: list[SopStepConfig] = []
    for node in selected:
        prerequisites = list(node.get("prerequisites", []))
        missing = [value for value in prerequisites if value not in selected_ids]
        if missing and policy != "remove_missing":
            raise SopConfigError(
                f"runtime node {node['node_id']!r} has filtered prerequisites: {missing}"
            )
        prerequisites = [value for value in prerequisites if value in selected_ids]
        business = business_steps[node["business_step"]]
        steps.append(
            SopStepConfig(
                step_id=node["node_id"],
                observation_id=node["observation_id"],
                name=str(node.get("name", "")),
                prerequisites=prerequisites,
                timeout_seconds=_optional_float(node.get("timeout_seconds")),
                max_retries=int(node.get("max_retries", 0)),
                skippable=bool(node.get("skippable", node.get("allow_skip", False))),
                continue_on_failure=bool(node.get("continue_on_failure", False)),
                min_confidence=float(node.get("min_confidence", 0.0)),
                business_step=str(node["business_step"]),
                business_step_name=str(business.get("name", node["business_step"])),
                business_step_order=int(business.get("order", 0)),
                action_flow=tuple(str(value) for value in node.get("action_flow", [])),
                requirements=tuple(str(value) for value in node.get("requirements", [])),
            )
        )
    return steps


def load_sop_steps(
    path: str | Path,
    **kwargs: Any,
) -> tuple[dict[str, Any], list[SopStepConfig]]:
    """Load one YAML document and return it together with filtered steps."""

    config = load_sop_config(path)
    return config, build_sop_steps(config, **kwargs)


def _mapping_list(value: Any, field_name: str) -> Sequence[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise SopConfigError(f"{field_name} must be a list of mappings")
    return value


def _required_text(value: Mapping[str, Any], key: str, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise SopConfigError(f"{label}.{key} must be a non-empty string")
    return result


def _ensure_unique(values: Sequence[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise SopConfigError(f"{label} values must be unique")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
