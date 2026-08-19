"""Configurable SOP state machine for observation events.

The module deliberately does not import the observation implementation.  Events
may be mappings or arbitrary objects. Both ``event_type``/``timestamp`` and the
observation runtime's ``phase``/``end_time`` field names are accepted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCHEMA_VERSION = "1.0"


class SopMode(str, Enum):
    FREE_OBSERVATION = "free_observation"
    STRICT = "strict"


class StepStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_REVIEW = "needs_review"


TERMINAL_SUCCESS_STATUSES = {StepStatus.COMPLETED, StepStatus.SKIPPED}


@dataclass(frozen=True)
class SopStepConfig:
    """Static definition of one SOP step.

    ``prerequisites`` contains step IDs, not observation IDs.  Free-observation
    mode intentionally ignores these dependencies; strict mode enforces them.
    ``max_retries`` is the number of retries after the initial attempt.
    """

    step_id: str
    observation_id: str
    name: str = ""
    prerequisites: Sequence[str] = field(default_factory=tuple)
    timeout_seconds: Optional[float] = None
    max_retries: int = 0
    skippable: bool = False
    continue_on_failure: bool = False
    min_confidence: float = 0.0
    business_step: str = ""
    business_step_name: str = ""
    business_step_order: int = 0
    action_flow: Sequence[str] = field(default_factory=tuple)
    requirements: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prerequisites", tuple(self.prerequisites))
        object.__setattr__(self, "action_flow", tuple(self.action_flow))
        object.__setattr__(self, "requirements", tuple(self.requirements))
        if not self.step_id or not self.observation_id:
            raise ValueError("step_id and observation_id must not be empty")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

    @classmethod
    def from_value(cls, value: "SopStepConfig | Mapping[str, Any]") -> "SopStepConfig":
        if isinstance(value, cls):
            return value
        data = dict(value)
        if "allow_skip" in data and "skippable" not in data:
            data["skippable"] = data.pop("allow_skip")
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["prerequisites"] = list(self.prerequisites)
        data["action_flow"] = list(self.action_flow)
        data["requirements"] = list(self.requirements)
        return data


@dataclass
class StepRuntime:
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    last_event_type: Optional[str] = None
    confidence: float = 0.0
    review_reason: Optional[str] = None
    skip_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepRuntime":
        values = dict(data)
        values["status"] = StepStatus(values.get("status", StepStatus.PENDING.value))
        return cls(**values)


@dataclass(frozen=True)
class StateTransition:
    accepted: bool
    step_id: Optional[str]
    action: str
    previous_status: Optional[StepStatus] = None
    status: Optional[StepStatus] = None
    timestamp: Optional[float] = None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.accepted

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["previous_status"] = (
            self.previous_status.value if self.previous_status is not None else None
        )
        data["status"] = self.status.value if self.status is not None else None
        return data


def _event_value(event: Any, name: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(name, default)
    return getattr(event, name, default)


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


class SopStateMachine:
    """Consume observation events and maintain serializable SOP progress."""

    def __init__(
        self,
        steps: Iterable[SopStepConfig | Mapping[str, Any]],
        mode: SopMode | str = SopMode.FREE_OBSERVATION,
    ) -> None:
        self.mode = SopMode(mode)
        self.steps = [SopStepConfig.from_value(step) for step in steps]
        if not self.steps:
            raise ValueError("at least one SOP step is required")
        self._validate_steps()
        self._configs = {step.step_id: step for step in self.steps}
        self._observation_steps = {
            step.observation_id: step.step_id for step in self.steps
        }
        self.runtime = {step.step_id: StepRuntime() for step in self.steps}
        self.transition_history: List[StateTransition] = []
        self.last_timestamp: Optional[float] = None
        if self.mode is SopMode.STRICT:
            self._activate_next(None)

    def _validate_steps(self) -> None:
        step_ids = [step.step_id for step in self.steps]
        observation_ids = [step.observation_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("step_id values must be unique")
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation_id values must be unique")
        positions = {step_id: index for index, step_id in enumerate(step_ids)}
        for index, step in enumerate(self.steps):
            for prerequisite in step.prerequisites:
                if prerequisite not in positions:
                    raise ValueError(
                        f"unknown prerequisite {prerequisite!r} for step {step.step_id!r}"
                    )
                if positions[prerequisite] >= index:
                    raise ValueError(
                        "prerequisites must refer to an earlier step in SOP order"
                    )

    @property
    def current_step_id(self) -> Optional[str]:
        if self.mode is SopMode.FREE_OBSERVATION:
            return None
        for step in self.steps:
            if self.runtime[step.step_id].status in {
                StepStatus.ACTIVE,
                StepStatus.NEEDS_REVIEW,
            }:
                return step.step_id
        return None

    @property
    def needs_review(self) -> bool:
        return any(
            state.status is StepStatus.NEEDS_REVIEW
            for state in self.runtime.values()
        )

    @property
    def current_step_config(self) -> Optional[SopStepConfig]:
        step_id = self.current_step_id
        return self._configs.get(step_id) if step_id is not None else None

    @property
    def is_complete(self) -> bool:
        return all(
            state.status in TERMINAL_SUCCESS_STATUSES
            for state in self.runtime.values()
        )

    @property
    def status(self) -> str:
        if self.needs_review:
            return StepStatus.NEEDS_REVIEW.value
        if self.is_complete:
            return StepStatus.COMPLETED.value
        if any(state.status is StepStatus.FAILED for state in self.runtime.values()):
            return StepStatus.FAILED.value
        return "running"

    def get_step_state(self, step_id: str) -> StepRuntime:
        try:
            return self.runtime[step_id]
        except KeyError as exc:
            raise KeyError(f"unknown SOP step: {step_id}") from exc

    def reset(self) -> None:
        """Reset runtime progress while preserving the configured SOP steps."""
        self.runtime = {
            step.step_id: StepRuntime() for step in self.steps
        }
        self.transition_history.clear()
        self.last_timestamp = None
        if self.mode is SopMode.STRICT:
            self._activate_next(None)

    def process_event(self, event: Any) -> StateTransition:
        observation_id = _event_value(event, "observation_id")
        event_type = _enum_value(
            _event_value(event, "event_type", _event_value(event, "phase"))
        )
        timestamp = _event_value(
            event, "timestamp", _event_value(event, "end_time")
        )
        confidence = _event_value(event, "confidence", 1.0)
        if observation_id is None or event_type is None or timestamp is None:
            raise ValueError(
                "event must provide observation_id, event_type and timestamp"
            )
        event_type = str(event_type).lower()
        if event_type not in {"started", "completed", "failed", "uncertain"}:
            raise ValueError(f"unsupported event_type: {event_type}")
        timestamp = float(timestamp)
        confidence = float(confidence)
        self.tick(timestamp)

        step_id = self._observation_steps.get(str(observation_id))
        if step_id is None:
            return self._record(False, None, "ignored", None, None, timestamp,
                                "observation is not configured")
        state = self.runtime[step_id]
        config = self._configs[step_id]
        if self.mode is SopMode.STRICT and self.current_step_id != step_id:
            return self._record(False, step_id, "ignored", state.status,
                                state.status, timestamp, "step is not currently active")
        if state.status in TERMINAL_SUCCESS_STATUSES:
            return self._record(False, step_id, "ignored", state.status,
                                state.status, timestamp, "step is already terminal")
        if state.status is StepStatus.FAILED:
            return self._record(False, step_id, "ignored", state.status,
                                state.status, timestamp, "step has exhausted retries")

        if confidence < config.min_confidence and event_type != "failed":
            return self._set_review(
                step_id, timestamp, confidence,
                f"confidence {confidence:.3f} is below {config.min_confidence:.3f}",
            )

        state.last_event_type = event_type
        state.confidence = confidence
        if event_type == "started":
            return self._start_step(step_id, timestamp)
        if event_type == "completed":
            return self._complete_step(step_id, timestamp)
        if event_type == "uncertain":
            reason = _event_value(event, "reason") or _event_value(
                event, "uncertain_reason", "observation evidence is uncertain"
            )
            return self._set_review(step_id, timestamp, confidence, str(reason))
        reason = _event_value(event, "reason", "observation reported failure")
        return self._fail_or_retry(step_id, timestamp, str(reason), "failed")

    def tick(self, timestamp: float) -> List[StateTransition]:
        """Advance the clock and apply active-step timeouts."""

        timestamp = float(timestamp)
        if self.last_timestamp is None or timestamp > self.last_timestamp:
            self.last_timestamp = timestamp
        transitions: List[StateTransition] = []
        states = list(self.runtime.items())
        for step_id, state in states:
            config = self._configs[step_id]
            if state.status is not StepStatus.ACTIVE:
                continue
            if state.started_at is None:
                state.started_at = timestamp
                if state.attempts == 0:
                    state.attempts = 1
                continue
            if (
                config.timeout_seconds is not None
                and timestamp - state.started_at >= config.timeout_seconds
            ):
                transitions.append(
                    self._fail_or_retry(step_id, timestamp, "step timed out", "timeout")
                )
        return transitions

    def skip_step(
        self,
        step_id: str,
        reason: str,
        timestamp: float,
        *,
        force: bool = False,
    ) -> StateTransition:
        state = self.get_step_state(step_id)
        config = self._configs[step_id]
        timestamp = float(timestamp)
        if not config.skippable and not force:
            return self._record(False, step_id, "skip_rejected", state.status,
                                state.status, timestamp, "step is not skippable")
        if self.mode is SopMode.STRICT and self.current_step_id != step_id:
            return self._record(False, step_id, "skip_rejected", state.status,
                                state.status, timestamp, "step is not currently active")
        if state.status in TERMINAL_SUCCESS_STATUSES:
            return self._record(False, step_id, "skip_rejected", state.status,
                                state.status, timestamp, "step is already terminal")
        previous = state.status
        state.status = StepStatus.SKIPPED
        state.completed_at = timestamp
        state.skip_reason = reason
        transition = self._record(True, step_id, "skipped", previous,
                                  state.status, timestamp, reason)
        if self.mode is SopMode.STRICT:
            self._activate_next(timestamp)
        return transition

    def resolve_review(
        self,
        step_id: str,
        approved: bool,
        timestamp: float,
        reason: str = "",
    ) -> StateTransition:
        state = self.get_step_state(step_id)
        timestamp = float(timestamp)
        if state.status is not StepStatus.NEEDS_REVIEW:
            return self._record(False, step_id, "review_rejected", state.status,
                                state.status, timestamp, "step does not need review")
        if approved:
            return self._complete_step(step_id, timestamp, action="review_approved")
        return self._fail_or_retry(
            step_id, timestamp, reason or "manual review rejected", "review_rejected"
        )

    def retry_step(self, step_id: str, timestamp: float) -> StateTransition:
        """Manually retry a failed or review-blocked step within its retry budget."""

        state = self.get_step_state(step_id)
        timestamp = float(timestamp)
        if state.status not in {StepStatus.FAILED, StepStatus.NEEDS_REVIEW}:
            return self._record(False, step_id, "retry_rejected", state.status,
                                state.status, timestamp, "step cannot be retried")
        if not self._has_retry(step_id):
            return self._record(False, step_id, "retry_rejected", state.status,
                                state.status, timestamp, "retry budget is exhausted")
        previous = state.status
        state.status = StepStatus.ACTIVE
        state.attempts = max(state.attempts, 1) + 1
        state.started_at = timestamp
        state.completed_at = None
        state.review_reason = None
        return self._record(True, step_id, "retried", previous, state.status,
                            timestamp, "manual retry")

    def _start_step(self, step_id: str, timestamp: float) -> StateTransition:
        state = self.runtime[step_id]
        previous = state.status
        if state.status is StepStatus.NEEDS_REVIEW:
            return self._record(False, step_id, "ignored", previous, previous,
                                timestamp, "step is waiting for manual review")
        state.status = StepStatus.ACTIVE
        if state.attempts == 0:
            state.attempts = 1
        if state.started_at is None:
            state.started_at = timestamp
        return self._record(True, step_id, "started", previous, state.status,
                            timestamp)

    def _complete_step(
        self, step_id: str, timestamp: float, action: str = "completed"
    ) -> StateTransition:
        state = self.runtime[step_id]
        previous = state.status
        state.status = StepStatus.COMPLETED
        if state.started_at is None:
            state.started_at = timestamp
        if state.attempts == 0:
            state.attempts = 1
        state.completed_at = timestamp
        state.review_reason = None
        transition = self._record(True, step_id, action, previous, state.status,
                                  timestamp)
        if self.mode is SopMode.STRICT:
            self._activate_next(timestamp)
        return transition

    def _set_review(
        self, step_id: str, timestamp: float, confidence: float, reason: str
    ) -> StateTransition:
        state = self.runtime[step_id]
        previous = state.status
        state.status = StepStatus.NEEDS_REVIEW
        state.confidence = confidence
        state.review_reason = reason
        if state.started_at is None:
            state.started_at = timestamp
        if state.attempts == 0:
            state.attempts = 1
        return self._record(True, step_id, "needs_review", previous, state.status,
                            timestamp, reason)

    def _has_retry(self, step_id: str) -> bool:
        state = self.runtime[step_id]
        retries_used = max(state.attempts - 1, 0)
        return retries_used < self._configs[step_id].max_retries

    def _fail_or_retry(
        self, step_id: str, timestamp: float, reason: str, action: str
    ) -> StateTransition:
        state = self.runtime[step_id]
        previous = state.status
        if state.attempts == 0:
            state.attempts = 1
        if self._has_retry(step_id):
            state.attempts += 1
            state.status = StepStatus.ACTIVE
            state.started_at = timestamp
            state.review_reason = None
            return self._record(True, step_id, f"{action}_retry", previous,
                                state.status, timestamp, reason)
        if self._configs[step_id].continue_on_failure:
            state.status = StepStatus.SKIPPED
            state.completed_at = timestamp
            state.skip_reason = reason
            transition = self._record(
                True, step_id, f"{action}_continue", previous,
                state.status, timestamp, reason,
            )
            if self.mode is SopMode.STRICT:
                self._activate_next(timestamp)
            return transition
        state.status = StepStatus.FAILED
        state.completed_at = timestamp
        return self._record(True, step_id, action, previous, state.status,
                            timestamp, reason)

    def _prerequisites_met(self, config: SopStepConfig) -> bool:
        return all(
            self.runtime[step_id].status in TERMINAL_SUCCESS_STATUSES
            for step_id in config.prerequisites
        )

    def _activate_next(self, timestamp: Optional[float]) -> None:
        if self.mode is not SopMode.STRICT or self.current_step_id is not None:
            return
        for config in self.steps:
            state = self.runtime[config.step_id]
            if state.status is StepStatus.FAILED:
                return
            if state.status is not StepStatus.PENDING:
                continue
            if not self._prerequisites_met(config):
                return
            state.status = StepStatus.ACTIVE
            state.started_at = timestamp
            state.attempts = 1
            return

    def _record(
        self,
        accepted: bool,
        step_id: Optional[str],
        action: str,
        previous: Optional[StepStatus],
        status: Optional[StepStatus],
        timestamp: Optional[float],
        reason: str = "",
    ) -> StateTransition:
        transition = StateTransition(
            accepted, step_id, action, previous, status, timestamp, reason
        )
        self.transition_history.append(transition)
        return transition

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode.value,
            "status": self.status,
            "current_step_id": self.current_step_id,
            "last_timestamp": self.last_timestamp,
            "steps": [step.to_dict() for step in self.steps],
            "runtime": {
                step_id: state.to_dict() for step_id, state in self.runtime.items()
            },
            "transition_history": [
                transition.to_dict() for transition in self.transition_history
            ],
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json() + "\n", encoding="utf-8")
        return output

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SopStateMachine":
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported SOP state schema version")
        machine = cls(data["steps"], mode=data["mode"])
        runtime_data = data.get("runtime", {})
        machine.runtime = {
            step.step_id: StepRuntime.from_dict(runtime_data.get(step.step_id, {}))
            for step in machine.steps
        }
        machine.last_timestamp = data.get("last_timestamp")
        machine.transition_history = []
        for item in data.get("transition_history", []):
            values = dict(item)
            previous = values.get("previous_status")
            status = values.get("status")
            values["previous_status"] = StepStatus(previous) if previous else None
            values["status"] = StepStatus(status) if status else None
            machine.transition_history.append(StateTransition(**values))
        return machine

    @classmethod
    def from_json(cls, value: str) -> "SopStateMachine":
        return cls.from_dict(json.loads(value))
