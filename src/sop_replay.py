"""Offline replay of observation events through an SOP state machine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .sop_config import load_sop_steps
from .sop_state_machine import SopMode, SopStateMachine, StateTransition


REPLAY_SCHEMA_VERSION = "1.0"
CONTROL_OPERATIONS = frozenset({"tick", "skip", "review", "retry"})


class SopReplayError(ValueError):
    """Raised when an event file or replay control record is invalid."""


def load_event_records(path: str | Path) -> list[dict[str, Any]]:
    """Load replay records from a JSON array/object or a JSONL file."""

    event_path = Path(path)
    try:
        text = event_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise SopReplayError(f"cannot load event file {event_path}: {exc}") from exc
    if not text.strip():
        raise SopReplayError("event file is empty")

    if event_path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SopReplayError(
                    f"invalid JSONL at line {line_number}: {exc.msg}"
                ) from exc
    else:
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SopReplayError(f"invalid JSON: {exc.msg}") from exc
        if isinstance(document, list):
            records = document
        elif isinstance(document, dict) and isinstance(document.get("events"), list):
            records = document["events"]
        else:
            records = [document]

    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SopReplayError(f"record {index} must be a JSON object")
        normalized.append(dict(record))
    if not normalized:
        raise SopReplayError("event file contains no records")
    return normalized


def replay_sop_events(
    records: Iterable[Mapping[str, Any]],
    *,
    config_path: str | Path,
    mode: SopMode | str = SopMode.FREE_OBSERVATION,
    include_deferred: bool = False,
    include_disabled: bool = False,
    available_observation_ids: Iterable[str] | None = None,
    sort_events: bool = False,
) -> dict[str, Any]:
    """Replay event/control records and return a self-contained report."""

    config, steps = load_sop_steps(
        config_path,
        include_deferred=include_deferred,
        include_disabled=include_disabled,
        available_observation_ids=available_observation_ids,
    )
    machine = SopStateMachine(steps, mode=mode)
    ordered_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise SopReplayError(f"record {index} must be a mapping")
        ordered_records.append(dict(record))
    if sort_events:
        ordered_records = _stable_timestamp_sort(ordered_records)

    record_reports: list[dict[str, Any]] = []
    accepted_records = 0
    ignored_records = 0
    review_records = 0
    for index, record in enumerate(ordered_records):
        history_start = len(machine.transition_history)
        primary = _apply_record(machine, record)
        new_transitions = machine.transition_history[history_start:]
        primary_accepted = _primary_accepted(primary)
        if primary_accepted:
            accepted_records += 1
        else:
            ignored_records += 1
        if any(
            transition.action in {"needs_review", "review_approved", "review_rejected"}
            for transition in new_transitions
        ):
            review_records += 1
        record_reports.append(
            {
                "index": index,
                "record_type": "control" if "operation" in record else "event",
                "timestamp": _record_timestamp(record),
                "accepted": primary_accepted,
                "input": record,
                "transitions": [item.to_dict() for item in new_transitions],
            }
        )

    final_machine = machine.to_dict()
    needs_review = [
        step_id
        for step_id, state in final_machine["runtime"].items()
        if state["status"] == "needs_review"
    ]
    configured_node_ids = {step.step_id for step in steps}
    runtime_metadata = {
        node["node_id"]: node for node in config.get("runtime_nodes", [])
    }
    configured_nodes = []
    for step in steps:
        node = runtime_metadata[step.step_id]
        configured_nodes.append(
            {
                **step.to_dict(),
                "business_step": node["business_step"],
                "availability": node["availability"],
            }
        )
    omitted_nodes = [
        {
            "step_id": node["node_id"],
            "observation_id": node["observation_id"],
            "availability": node["availability"],
        }
        for node in config.get("runtime_nodes", [])
        if node["node_id"] not in configured_node_ids
    ]
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "sop_id": config.get("sop_id"),
        "config_path": str(Path(config_path)),
        "mode": machine.mode.value,
        "include_deferred": include_deferred,
        "include_disabled": include_disabled,
        "sort_events": sort_events,
        "scope": {
            "current_capabilities_only": not include_deferred and not include_disabled,
            "formal_acceptance_enabled": bool(
                config.get("formal_acceptance", {}).get("enabled", False)
            ),
            "omitted_runtime_nodes": omitted_nodes,
        },
        "configured_nodes": configured_nodes,
        "configured_observation_ids": [step.observation_id for step in steps],
        "summary": {
            "input_record_count": len(ordered_records),
            "accepted_record_count": accepted_records,
            "ignored_record_count": ignored_records,
            "review_record_count": review_records,
            "transition_count": len(machine.transition_history),
            "final_status": machine.status,
            "is_complete": machine.is_complete,
            "steps_needing_review": needs_review,
        },
        "records": record_reports,
        "final_machine": final_machine,
    }


def save_replay_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SopReplayError(f"cannot save replay report {output_path}: {exc}") from exc
    return output_path


def _apply_record(
    machine: SopStateMachine, record: Mapping[str, Any]
) -> StateTransition | Sequence[StateTransition]:
    operation = record.get("operation")
    if operation is None:
        try:
            return machine.process_event(record)
        except (TypeError, ValueError) as exc:
            raise SopReplayError(f"invalid observation event: {exc}") from exc
    operation = str(operation).lower()
    if operation not in CONTROL_OPERATIONS:
        raise SopReplayError(f"unsupported replay operation: {operation}")
    timestamp = _required_timestamp(record)
    try:
        if operation == "tick":
            return machine.tick(timestamp)
        step_id = _required_text(record, "step_id")
        if operation == "skip":
            return machine.skip_step(
                step_id,
                str(record.get("reason", "offline replay skip")),
                timestamp,
                force=_strict_bool(record.get("force", False), "force"),
            )
        if operation == "review":
            return machine.resolve_review(
                step_id,
                _strict_bool(record.get("approved"), "approved"),
                timestamp,
                str(record.get("reason", "")),
            )
        return machine.retry_step(step_id, timestamp)
    except (KeyError, TypeError, ValueError) as exc:
        raise SopReplayError(f"invalid {operation} control record: {exc}") from exc


def _stable_timestamp_sort(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = list(enumerate(records))
    try:
        indexed.sort(key=lambda item: (_required_timestamp(item[1]), item[0]))
    except SopReplayError:
        raise
    return [record for _, record in indexed]


def _record_timestamp(record: Mapping[str, Any]) -> float | None:
    value = record.get("timestamp", record.get("end_time"))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SopReplayError("timestamp or end_time must be numeric") from exc


def _required_timestamp(record: Mapping[str, Any]) -> float:
    value = _record_timestamp(record)
    if value is None:
        raise SopReplayError("record must provide timestamp or end_time")
    return value


def _required_text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise SopReplayError(f"record must provide non-empty {key}")
    return value


def _strict_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise SopReplayError(f"{field_name} must be a JSON boolean")
    return value


def _primary_accepted(
    result: StateTransition | Sequence[StateTransition],
) -> bool:
    if isinstance(result, StateTransition):
        return result.accepted
    return any(transition.accepted for transition in result)
