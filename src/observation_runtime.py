"""Temporal observation runtime shared by action and result observers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Protocol, Set, Tuple
import time
import uuid


class CameraRole(str, Enum):
    FRONT = "front"
    TABLETOP = "tabletop"
    SIDE = "side"
    SINGLE = "single"
    UNKNOWN = "unknown"


class ObservationState(str, Enum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class EventPhase(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


@dataclass
class EvidenceFrame:
    frame_idx: int
    timestamp: float
    camera_role: str
    bboxes: List[Dict[str, Any]] = field(default_factory=list)
    track_ids: List[int] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 4),
            "camera_role": self.camera_role,
            "bboxes": self.bboxes,
            "track_ids": self.track_ids,
            "metrics": self.metrics,
        }


@dataclass
class FrameContext:
    frame_idx: int
    timestamp: float
    camera_role: CameraRole
    frame_shape: Tuple[int, int]
    detections: List[Any]
    hand_results: List[Dict[str, Any]]
    pose_results: List[Dict[str, Any]]
    model_version: str = "unknown"
    model_classes: Set[str] = field(default_factory=set)
    capabilities: Set[str] = field(default_factory=set)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationSnapshot:
    observation_id: str
    name: str
    sop_step: int
    state: ObservationState
    confidence: float = 0.0
    value: Any = None
    reason: str = ""
    started_at: Optional[float] = None
    updated_at: float = field(default_factory=time.monotonic)
    metrics: Dict[str, Any] = field(default_factory=dict)
    experimental: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "name": self.name,
            "sop_step": self.sop_step,
            "state": self.state.value,
            "confidence": round(float(self.confidence), 4),
            "value": self.value,
            "reason": self.reason,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "metrics": self.metrics,
            "experimental": self.experimental,
        }


@dataclass
class ObservationEvent:
    observation_id: str
    name: str
    sop_step: int
    phase: EventPhase
    start_time: float
    end_time: float
    confidence: float
    camera_role: str
    value: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    evidence: List[EvidenceFrame] = field(default_factory=list)
    model_version: str = "unknown"
    rule_version: str = "1.0"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "observation_id": self.observation_id,
            "name": self.name,
            "sop_step": self.sop_step,
            "phase": self.phase.value,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "confidence": round(float(self.confidence), 4),
            "camera_role": self.camera_role,
            "value": self.value,
            "metrics": self.metrics,
            "evidence": [item.to_dict() for item in self.evidence],
            "model_version": self.model_version,
            "rule_version": self.rule_version,
        }


class TemporalObservation(Protocol):
    observation_id: str
    name: str
    sop_step: int
    camera_roles: Set[CameraRole]

    def update(
        self, context: FrameContext
    ) -> Tuple[ObservationSnapshot, List[ObservationEvent]]:
        ...

    def reset(self) -> None:
        ...


class ObservationEngine:
    """Routes frame contexts and retains session-wide event history."""

    def __init__(self, observations: Iterable[TemporalObservation]):
        self.observations = {item.observation_id: item for item in observations}
        self.snapshots: Dict[str, ObservationSnapshot] = {}
        self.events: List[ObservationEvent] = []

    def process(
        self, context: FrameContext
    ) -> Tuple[Dict[str, ObservationSnapshot], List[ObservationEvent]]:
        new_events: List[ObservationEvent] = []
        # Later observers can compose events emitted earlier in the same frame.
        # The list is frame-local and is replaced on every process call.
        context.extras["session_observation_events"] = list(self.events)
        context.extras["frame_observation_events"] = []
        for observation in self.observations.values():
            role_matches = context.camera_role in observation.camera_roles
            if context.camera_role in {CameraRole.SINGLE, CameraRole.FRONT}:
                role_matches = bool(
                    observation.camera_roles
                    & {CameraRole.FRONT, CameraRole.TABLETOP, CameraRole.SIDE}
                )
            if not role_matches:
                continue
            try:
                snapshot, emitted = observation.update(context)
            except Exception as exc:
                snapshot = ObservationSnapshot(
                    observation_id=observation.observation_id,
                    name=observation.name,
                    sop_step=observation.sop_step,
                    state=ObservationState.UNCERTAIN,
                    reason=f"观测器异常: {exc}",
                    updated_at=context.timestamp,
                )
                emitted = []
            self.snapshots[observation.observation_id] = snapshot
            new_events.extend(emitted)
            context.extras["frame_observation_events"].extend(emitted)
        self.events.extend(new_events)
        return dict(self.snapshots), new_events

    def reset(self) -> None:
        self.snapshots.clear()
        self.events.clear()
        for observation in self.observations.values():
            observation.reset()

    def reset_observation(self, observation_id: str) -> bool:
        observation = self.observations.get(observation_id)
        if observation is None:
            return False
        observation.reset()
        self.snapshots.pop(observation_id, None)
        return True

    def prepare_observation(self, observation_id: str, method_name: str) -> bool:
        """Run an optional lifecycle hook on one observation."""
        observation = self.observations.get(observation_id)
        method = getattr(observation, method_name, None) if observation is not None else None
        if method is None:
            return False
        method()
        self.snapshots.pop(observation_id, None)
        return True

    def events_as_dicts(self) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self.events]
