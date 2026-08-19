"""Front-camera SOP observations built from OCR and vessel-pose interactions."""

from __future__ import annotations

from collections import deque
from math import atan2, degrees, hypot
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..observation_runtime import (
    CameraRole,
    EventPhase,
    EvidenceFrame,
    FrameContext,
    ObservationEvent,
    ObservationSnapshot,
    ObservationState,
)
from ._rules import BrewWaitTimerObservation, PourStage, WarmCleanSequenceObservation


def _name(item: Any) -> str:
    return str(getattr(item, "item_name", ""))


def _bbox(item: Any) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in getattr(item, "bbox"))


def _center(item: Any) -> tuple[float, float]:
    value = getattr(item, "centroid", None)
    if value is not None:
        return float(value[0]), float(value[1])
    x, y, w, h = _bbox(item)
    return x + w / 2, y + h / 2


def _event(
    observation: Any,
    context: FrameContext,
    phase: EventPhase,
    started_at: float,
    confidence: float,
    value: Any,
    metrics: Dict[str, Any],
    evidence: Sequence[EvidenceFrame] = (),
) -> ObservationEvent:
    return ObservationEvent(
        observation_id=observation.observation_id,
        name=observation.name,
        sop_step=observation.sop_step,
        phase=phase,
        start_time=started_at,
        end_time=context.timestamp,
        confidence=confidence,
        camera_role=context.camera_role.value,
        value=value,
        metrics=metrics,
        evidence=list(evidence)[-8:],
        model_version=context.model_version,
        rule_version=observation.rule_version,
    )


def _interaction_evidence(context: FrameContext, interaction: Dict[str, Any]) -> EvidenceFrame:
    return EvidenceFrame(
        frame_idx=context.frame_idx,
        timestamp=context.timestamp,
        camera_role=context.camera_role.value,
        track_ids=[
            int(value) for value in (
                interaction.get("source_track_id"), interaction.get("target_track_id")
            ) if value is not None
        ],
        metrics=dict(interaction),
    )


class NumericRangeObservation:
    camera_roles: Set[CameraRole] = {CameraRole.FRONT}
    rule_version = "1.0"
    kind = ""
    minimum = 0.0
    maximum = 0.0
    unit = ""

    def __init__(self):
        self._emitted_key: Optional[tuple[float, bool]] = None

    def update(self, context: FrameContext):
        measurement = context.extras.get("ocr_measurements", {}).get(self.kind)
        if not measurement:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.UNCERTAIN,
                reason="未检测到对应显示屏或OCR尚未产生读数",
                updated_at=context.timestamp,
                metrics={"range": [self.minimum, self.maximum], "unit": self.unit},
            ), []
        value = measurement.get("value")
        stable = bool(measurement.get("stable", False))
        confidence = float(measurement.get("confidence", 0.0))
        metrics = {
            **dict(measurement),
            "range": [self.minimum, self.maximum],
            "within_range": bool(value is not None and self.minimum <= float(value) <= self.maximum),
        }
        if not stable or value is None:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.UNCERTAIN,
                confidence=confidence, reason=str(measurement.get("reason", "读数不稳定")),
                updated_at=context.timestamp, metrics=metrics,
            ), []
        value = float(value)
        passed = self.minimum <= value <= self.maximum
        state = ObservationState.COMPLETED if passed else ObservationState.FAILED
        phase = EventPhase.COMPLETED if passed else EventPhase.FAILED
        key = (round(value, 3), passed)
        events = []
        if key != self._emitted_key:
            events.append(_event(
                self, context, phase, context.timestamp, confidence, value, metrics
            ))
            self._emitted_key = key
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step, state,
            confidence=confidence, value=value,
            reason=("读数在合格区间内" if passed else "稳定读数超出合格区间"),
            started_at=context.timestamp, updated_at=context.timestamp, metrics=metrics,
        ), events

    def reset(self) -> None:
        self._emitted_key = None


class WaterTemperatureObservation(NumericRangeObservation):
    observation_id = "result_water_temperature"
    name = "烧水壶温度90至95摄氏度"
    sop_step = 2
    kind = "temperature"
    minimum = 90.0
    maximum = 95.0
    unit = "celsius"


class TeaWeightObservation(NumericRangeObservation):
    observation_id = "result_tea_weight"
    name = "投茶重量3至5克"
    sop_step = 3
    kind = "weight"
    minimum = 3.0
    maximum = 5.0
    unit = "grams"


class FrontWarmCleanSequenceObservation(WarmCleanSequenceObservation):
    """Complete front-view order including final waste-water disposal."""

    observation_id = "seq_warm_clean_front"
    name = "温杯洁具完整顺序"
    camera_roles: Set[CameraRole] = {CameraRole.FRONT}
    rule_version = "2.0-front-pose"
    STAGES = WarmCleanSequenceObservation.STAGES + (
        PourStage("waste", "建水", {"公道杯"}, {"建水"}),
    )

    def __init__(self, *args, **kwargs):
        # The current front-view SOP uses three tasting cups. Requiring all
        # three prevents a partial warm-clean pass from advancing the SOP.
        kwargs.setdefault("min_cups", 3)
        super().__init__(*args, **kwargs)

    @classmethod
    def required_classes_available(cls, classes: Set[str]) -> bool:
        return super().required_classes_available(classes) and "建水" in classes


class BrewDurationObservation(BrewWaitTimerObservation):
    observation_id = "result_brew_time_8_12"
    name = "冲泡时间8至12秒"
    camera_roles: Set[CameraRole] = {CameraRole.FRONT}
    rule_version = "2.0-front-formal"

    def __init__(self):
        super().__init__(
            minimum_wait_seconds=8.0,
            maximum_wait_seconds=12.0,
            require_injection=True,
        )


class StablePourObservation:
    """Debounce one pose-derived source-to-target interaction."""

    camera_roles: Set[CameraRole] = {CameraRole.FRONT}
    rule_version = "1.0-pose-gesture"
    source_names: frozenset[str] = frozenset()
    target_names: frozenset[str] = frozenset()
    stable_seconds = 0.6
    min_samples = 3

    def __init__(self, stable_seconds: Optional[float] = None, min_samples: Optional[int] = None):
        self.stable_seconds = float(stable_seconds or self.stable_seconds)
        self.min_samples = int(min_samples or self.min_samples)
        self._history: Deque[tuple[float, float, EvidenceFrame, Dict[str, Any]]] = deque()
        self._started_at: Optional[float] = None
        self._started_emitted = False
        self._completed = False
        self._completion_metrics: Dict[str, Any] = {}
        self._completion_confidence = 0.0

    @classmethod
    def required_classes_available(cls, classes: Set[str]) -> bool:
        return bool(cls.source_names & classes) and bool(cls.target_names & classes)

    def _matches(self, context: FrameContext) -> List[Dict[str, Any]]:
        return [
            dict(row) for row in context.extras.get("pour_interactions", [])
            if str(row.get("source", "")) in self.source_names
            and str(row.get("target", "")) in self.target_names
            and float(row.get("confidence", 0.0)) >= 0.45
        ]

    def update(self, context: FrameContext):
        if not self.required_classes_available(context.model_classes):
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.UNCERTAIN,
                reason="当前正面模型缺少动作所需器具类别",
                updated_at=context.timestamp, experimental=True,
            ), []
        if self._completed:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.COMPLETED,
                confidence=self._completion_confidence, value=True,
                reason="规范倾倒动作已完成；未验证实际液体流出",
                started_at=self._started_at, updated_at=context.timestamp,
                metrics=self._completion_metrics, experimental=True,
            ), []
        matches = self._matches(context)
        if not matches:
            self._history.clear()
            self._started_at = None
            self._started_emitted = False
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.IDLE,
                reason="等待器具离桌、倾斜且出水端对准目标",
                updated_at=context.timestamp, experimental=True,
            ), []
        interaction = max(matches, key=lambda row: float(row.get("confidence", 0.0)))
        evidence = _interaction_evidence(context, interaction)
        confidence = float(interaction.get("confidence", 0.0))
        self._history.append((context.timestamp, confidence, evidence, interaction))
        cutoff = context.timestamp - self.stable_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        if self._started_at is None:
            self._started_at = context.timestamp
        emitted = []
        if not self._started_emitted:
            emitted.append(_event(
                self, context, EventPhase.STARTED, self._started_at, confidence,
                True, interaction, [evidence],
            ))
            self._started_emitted = True
        span = self._history[-1][0] - self._history[0][0] if len(self._history) > 1 else 0.0
        if len(self._history) >= self.min_samples and span >= self.stable_seconds * 0.95:
            self._completed = True
            self._completion_confidence = float(np.mean([row[1] for row in self._history]))
            self._completion_metrics = {
                **interaction,
                "duration_seconds": round(span, 3),
                "liquid_verified": False,
                "gesture_only": True,
            }
            emitted.append(_event(
                self, context, EventPhase.COMPLETED, self._started_at,
                self._completion_confidence, True, self._completion_metrics,
                [row[2] for row in self._history],
            ))
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.COMPLETED,
                confidence=self._completion_confidence, value=True,
                reason="规范倾倒动作已完成；未验证实际液体流出",
                started_at=self._started_at, updated_at=context.timestamp,
                metrics=self._completion_metrics, experimental=True,
            ), emitted
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step,
            ObservationState.ACTIVE, confidence=confidence, value="正在倾倒",
            reason="倾倒关系已成立，等待持续时间满足",
            started_at=self._started_at, updated_at=context.timestamp,
            metrics=interaction, experimental=True,
        ), emitted

    def reset(self) -> None:
        self._history.clear()
        self._started_at = None
        self._started_emitted = False
        self._completed = False
        self._completion_metrics.clear()
        self._completion_confidence = 0.0


class TeaLotusToGaiwanObservation(StablePourObservation):
    observation_id = "action_tea_lotus_to_gaiwan"
    name = "茶叶投入盖碗"
    sop_step = 4
    source_names = frozenset({"茶荷"})
    target_names = frozenset({"盖碗碗身", "盖碗（碗身）"})


class WaterInjectionObservation(StablePourObservation):
    observation_id = "action_water_injection"
    name = "烧水壶向盖碗旋转注水"
    sop_step = 5
    source_names = frozenset({"烧水壶"})
    target_names = frozenset({"盖碗碗身", "盖碗（碗身）"})
    rule_version = "1.0-pose-arc"

    def __init__(self, stable_seconds: float = 0.8, min_samples: int = 4, minimum_arc_degrees: float = 30.0):
        super().__init__(stable_seconds, min_samples)
        self.minimum_arc_degrees = float(minimum_arc_degrees)
        self._angles: Deque[tuple[float, float]] = deque()

    def _matches(self, context: FrameContext) -> List[Dict[str, Any]]:
        rows = super()._matches(context)
        accepted = []
        for row in rows:
            outlet = row.get("outlet_point")
            target_center = row.get("target_center")
            if not outlet or not target_center:
                continue
            angle = degrees(atan2(float(outlet[1]) - float(target_center[1]), float(outlet[0]) - float(target_center[0])))
            self._angles.append((context.timestamp, angle))
            cutoff = context.timestamp - max(1.5, self.stable_seconds * 2)
            while self._angles and self._angles[0][0] < cutoff:
                self._angles.popleft()
            arc = 0.0
            for (_, previous), (_, current) in zip(self._angles, list(self._angles)[1:]):
                arc += abs((current - previous + 180.0) % 360.0 - 180.0)
            row["orbit_arc_degrees"] = round(arc, 3)
            row["orbit_verified"] = arc >= self.minimum_arc_degrees
            if row["orbit_verified"]:
                accepted.append(row)
        return accepted

    def reset(self) -> None:
        super().reset()
        self._angles.clear()


class TeaDistributionObservation:
    observation_id = "action_tea_distribution"
    name = "公道杯依次分茶"
    sop_step = 6
    camera_roles: Set[CameraRole] = {CameraRole.FRONT}
    rule_version = "1.0-pose-sequence"

    def __init__(self, dwell_seconds: float = 0.6, min_samples: int = 3, minimum_cups: int = 2):
        self.dwell_seconds = float(dwell_seconds)
        self.min_samples = int(min_samples)
        self.minimum_cups = int(minimum_cups)
        self._current_id: Any = None
        self._current: Deque[tuple[float, Dict[str, Any]]] = deque()
        self._targets: list[tuple[Any, float]] = []
        self._completed = False
        self._failed = False
        self._started_at: Optional[float] = None

    @staticmethod
    def required_classes_available(classes: Set[str]) -> bool:
        return {"公道杯", "品茗杯"} <= classes

    def update(self, context: FrameContext):
        if not self.required_classes_available(context.model_classes):
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                reason="当前正面模型需要公道杯和品茗杯", updated_at=context.timestamp,
            ), []
        if self._completed or self._failed:
            metrics = self._metrics(context)
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.COMPLETED if self._completed else ObservationState.FAILED,
                confidence=0.9, value=metrics.get("direction"),
                reason="分茶顺序完成" if self._completed else "分茶目标顺序发生折返",
                started_at=self._started_at, updated_at=context.timestamp, metrics=metrics,
            ), []
        rows = [
            dict(row) for row in context.extras.get("pour_interactions", [])
            if row.get("source") == "公道杯" and row.get("target") == "品茗杯"
        ]
        if not rows:
            self._current.clear()
            self._current_id = None
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.IDLE,
                reason="等待公道杯对准品茗杯并稳定倾倒", updated_at=context.timestamp,
                metrics=self._metrics(context),
            ), []
        row = max(rows, key=lambda value: float(value.get("confidence", 0.0)))
        target_id = row.get("target_track_id")
        if target_id is None:
            target_id = tuple(round(float(v), -1) for v in row.get("target_center", (0, 0)))
        if target_id != self._current_id:
            self._current_id = target_id
            self._current.clear()
        self._current.append((context.timestamp, row))
        cutoff = context.timestamp - self.dwell_seconds
        while self._current and self._current[0][0] < cutoff:
            self._current.popleft()
        if self._started_at is None:
            self._started_at = context.timestamp
        emitted = []
        span = self._current[-1][0] - self._current[0][0] if len(self._current) > 1 else 0.0
        if len(self._current) >= self.min_samples and span >= self.dwell_seconds * 0.95:
            if not any(value[0] == target_id for value in self._targets):
                x = float(row.get("target_center", [0, 0])[0])
                self._targets.append((target_id, x))
                self._current.clear()
                if len(self._targets) >= 3:
                    deltas = np.diff([value[1] for value in self._targets])
                    if not (np.all(deltas > 0) or np.all(deltas < 0)):
                        self._failed = True
                        metrics = self._metrics(context)
                        emitted.append(_event(
                            self, context, EventPhase.FAILED, self._started_at, 0.9,
                            metrics.get("direction"), metrics,
                        ))
                        return ObservationSnapshot(
                            self.observation_id, self.name, self.sop_step, ObservationState.FAILED,
                            confidence=0.9, reason="分茶目标顺序发生折返",
                            started_at=self._started_at, updated_at=context.timestamp, metrics=metrics,
                        ), emitted
            cup_count = sum(1 for item in context.detections if _name(item) == "品茗杯")
            required = max(self.minimum_cups, cup_count)
            if len(self._targets) >= required:
                self._completed = True
                metrics = self._metrics(context)
                emitted.append(_event(
                    self, context, EventPhase.COMPLETED, self._started_at, 0.9,
                    metrics.get("direction"), metrics,
                ))
        metrics = self._metrics(context)
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step,
            ObservationState.COMPLETED if self._completed else ObservationState.ACTIVE,
            confidence=float(row.get("confidence", 0.0)), value=metrics.get("direction"),
            reason="分茶顺序完成" if self._completed else "正在记录不同品茗杯目标",
            started_at=self._started_at, updated_at=context.timestamp, metrics=metrics,
        ), emitted

    def _metrics(self, context: FrameContext) -> Dict[str, Any]:
        positions = [value[1] for value in self._targets]
        direction = None
        if len(positions) >= 2:
            direction = "从左到右" if positions[-1] > positions[0] else "从右到左"
        return {
            "target_track_ids": [value[0] for value in self._targets],
            "target_x_positions": positions,
            "target_count": len(self._targets),
            "direction": direction,
            "liquid_verified": False,
            "gesture_only": True,
        }

    def reset(self) -> None:
        self._current_id = None
        self._current.clear()
        self._targets.clear()
        self._completed = False
        self._failed = False
        self._started_at = None


class TwoHandServeTrayObservation:
    observation_id = "action_two_hand_serve_tray"
    name = "双手奉茶"
    sop_step = 6
    camera_roles: Set[CameraRole] = {CameraRole.FRONT}
    rule_version = "1.0-front-geometry"

    def __init__(self, stable_seconds: float = 0.8):
        self.stable_seconds = float(stable_seconds)
        self._rest_y: Deque[float] = deque(maxlen=20)
        self._rest_area: Deque[float] = deque(maxlen=20)
        self._active: Deque[tuple[float, float]] = deque()
        self._started_at: Optional[float] = None
        self._completed = False

    def update(self, context: FrameContext):
        trays = [item for item in context.detections if _name(item) == "茶盘"]
        hands = [row for row in context.hand_results if float(row.get("confidence", 0.0)) >= 0.55]
        if not trays or len(hands) < 2:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                reason="需要同时看清茶盘和双手", updated_at=context.timestamp,
            ), []
        tray = max(trays, key=lambda item: float(getattr(item, "confidence", 0.0)))
        x, y, w, h = _bbox(tray)
        cx, cy = _center(tray)
        hand_centers = [tuple(map(float, row.get("center", (0, 0)))) for row in hands]
        left = any(x - 0.25 * w <= px <= x + 0.25 * w and y - 0.2 * h <= py <= y + 1.2 * h for px, py in hand_centers)
        right = any(x + 0.75 * w <= px <= x + 1.25 * w and y - 0.2 * h <= py <= y + 1.2 * h for px, py in hand_centers)
        contact = left and right
        if not contact:
            self._rest_y.append(cy)
            self._rest_area.append(w * h)
        rest_y = float(np.median(self._rest_y)) if len(self._rest_y) >= 5 else None
        rest_area = float(np.median(self._rest_area)) if len(self._rest_area) >= 5 else None
        lift = rest_y is not None and rest_y - cy >= max(0.12 * h, context.frame_shape[0] * 0.015)
        forward = rest_area is not None and w * h >= rest_area * 1.08
        active = contact and lift and forward
        metrics = {
            "hands_on_opposite_sides": contact,
            "tray_lifted": lift,
            "forward_motion": forward,
            "area_growth_ratio": round(w * h / rest_area, 3) if rest_area else None,
        }
        if not active:
            self._active.clear()
            self._started_at = None
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.IDLE,
                reason="等待双手接触茶盘两侧、离桌并向前移动",
                updated_at=context.timestamp, metrics=metrics,
            ), []
        if self._started_at is None:
            self._started_at = context.timestamp
        self._active.append((context.timestamp, float(getattr(tray, "confidence", 0.0))))
        cutoff = context.timestamp - self.stable_seconds
        while self._active and self._active[0][0] < cutoff:
            self._active.popleft()
        span = self._active[-1][0] - self._active[0][0] if len(self._active) > 1 else 0.0
        events = []
        if not self._completed and span >= self.stable_seconds * 0.95:
            self._completed = True
            confidence = float(np.mean([row[1] for row in self._active]))
            events.append(_event(
                self, context, EventPhase.COMPLETED, self._started_at,
                confidence, True, metrics,
            ))
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step,
            ObservationState.COMPLETED if self._completed else ObservationState.ACTIVE,
            confidence=float(getattr(tray, "confidence", 0.0)), value=self._completed,
            reason="双手奉茶动作完成" if self._completed else "奉茶动作保持中",
            started_at=self._started_at, updated_at=context.timestamp, metrics=metrics,
        ), events

    def reset(self) -> None:
        self._rest_y.clear()
        self._rest_area.clear()
        self._active.clear()
        self._started_at = None
        self._completed = False


__all__ = [
    "BrewDurationObservation",
    "FrontWarmCleanSequenceObservation",
    "TeaDistributionObservation",
    "TeaLotusToGaiwanObservation",
    "TeaWeightObservation",
    "TwoHandServeTrayObservation",
    "WaterInjectionObservation",
    "WaterTemperatureObservation",
]
