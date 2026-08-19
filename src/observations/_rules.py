"""Rule-based temporal observations for the first action-recognition phase."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
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


def _item_name(item: Any) -> str:
    return str(getattr(item, "item_name", ""))


def _bbox(item: Any) -> Tuple[float, float, float, float]:
    x, y, w, h = getattr(item, "bbox")
    return float(x), float(y), float(w), float(h)


def _centroid(item: Any) -> Tuple[float, float]:
    center = getattr(item, "centroid", None)
    if center is not None:
        return float(center[0]), float(center[1])
    x, y, w, h = _bbox(item)
    return x + w / 2, y + h / 2


def _evidence(context: FrameContext, items: Iterable[Any], metrics: Dict[str, Any]) -> EvidenceFrame:
    boxes = []
    track_ids = []
    for item in items:
        x, y, w, h = _bbox(item)
        boxes.append({"label": _item_name(item), "bbox": [round(x), round(y), round(w), round(h)]})
        track_id = getattr(item, "track_id", None)
        if track_id is not None:
            track_ids.append(int(track_id))
    return EvidenceFrame(
        frame_idx=context.frame_idx,
        timestamp=context.timestamp,
        camera_role=context.camera_role.value,
        bboxes=boxes,
        track_ids=track_ids,
        metrics=metrics,
    )


def _event(
    observation: Any,
    context: FrameContext,
    phase: EventPhase,
    start_time: float,
    confidence: float,
    value: Any,
    metrics: Dict[str, Any],
    evidence: Sequence[EvidenceFrame],
) -> ObservationEvent:
    return ObservationEvent(
        observation_id=observation.observation_id,
        name=observation.name,
        sop_step=observation.sop_step,
        phase=phase,
        start_time=start_time,
        end_time=context.timestamp,
        confidence=confidence,
        camera_role=context.camera_role.value,
        value=value,
        metrics=metrics,
        evidence=list(evidence)[-8:],
        model_version=context.model_version,
        rule_version=observation.rule_version,
    )


@dataclass
class LayoutEvaluation:
    label: Optional[str]
    confidence: float
    metrics: Dict[str, Any]
    reason: str = ""


class CupLayoutObservation:
    observation_id = "result_cup_layout"
    name = "品茗杯布局"
    sop_step = 6
    camera_roles: Set[CameraRole] = {CameraRole.TABLETOP}
    rule_version = "1.1"

    def __init__(self, stable_seconds: float = 1.0, min_samples: int = 5, stable_ratio: float = 0.8):
        self.stable_seconds = stable_seconds
        self.min_samples = min_samples
        self.stable_ratio = stable_ratio
        self._history: Deque[Tuple[float, Optional[str], float, EvidenceFrame]] = deque()
        self._last_completed_label: Optional[str] = None
        self._candidate_label: Optional[str] = None
        self._started_at: Optional[float] = None
        self._started_event_emitted = False

    @staticmethod
    def evaluate(items: Sequence[Any]) -> LayoutEvaluation:
        cups = [item for item in items if _item_name(item) == "品茗杯"]
        if len(cups) < 2:
            return LayoutEvaluation(
                None,
                0.0,
                {"cup_count": len(cups), "classification": "无法判断"},
                "至少需要检测到2个品茗杯",
            )

        points = np.asarray([_centroid(item) for item in cups], dtype=np.float64)
        diameters = [max(1.0, (_bbox(item)[2] + _bbox(item)[3]) / 2) for item in cups]
        diameter = float(np.median(diameters))

        pin_metrics = CupLayoutObservation._pin_layout(points, diameter)
        if len(cups) == 3 and pin_metrics[0]:
            return LayoutEvaluation("品字形", pin_metrics[1], {
                "cup_count": len(cups), "median_cup_diameter": round(diameter, 3),
                "classification": "品字形", **pin_metrics[2]
            })

        line_metrics = CupLayoutObservation._line_layout(points, diameter)
        if line_metrics[0]:
            return LayoutEvaluation("一字形", line_metrics[1], {
                "cup_count": len(cups), "median_cup_diameter": round(diameter, 3),
                "classification": "一字形", **line_metrics[2]
            })
        shape_confidence = max(line_metrics[1], pin_metrics[1])
        return LayoutEvaluation(None, max(0.0, 1.0 - shape_confidence), {
            "cup_count": len(cups), "median_cup_diameter": round(diameter, 3),
            "classification": "其他布局",
            "closest_shape_confidence": round(shape_confidence, 4),
            **line_metrics[2],
        }, "当前杯位属于其他布局")

    @staticmethod
    def _line_layout(points: np.ndarray, diameter: float) -> Tuple[bool, float, Dict[str, Any]]:
        centered = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        projection = centered @ axis
        orthogonal = centered - np.outer(projection, axis)
        orthogonal_distances = np.sqrt(np.sum(orthogonal ** 2, axis=1)) / diameter
        rmse = float(np.sqrt(np.mean(orthogonal_distances ** 2)))
        max_error = float(np.max(orthogonal_distances))
        ordered = np.sort(projection)
        gaps = np.diff(ordered) / diameter
        gap_cv = float(np.std(gaps) / max(np.mean(gaps), 1e-6)) if len(gaps) else 0.0
        valid = (
            rmse <= 0.35
            and max_error <= 0.25
            and gap_cv <= 0.35
            and bool(np.all(gaps >= 0.45))
        )
        confidence = max(0.0, min(
            1.0,
            1.0 - 0.35 * rmse / 0.35 - 0.4 * max_error / 0.25 - 0.25 * gap_cv / 0.35,
        ))
        return valid, confidence, {
            "line_rmse_diameter": round(rmse, 4),
            "line_max_error_diameter": round(max_error, 4),
            "spacing_cv": round(gap_cv, 4),
            "min_gap_diameter": round(float(gaps.min()), 4) if len(gaps) else 0.0,
        }

    @staticmethod
    def _pin_layout(points: np.ndarray, diameter: float) -> Tuple[bool, float, Dict[str, Any]]:
        if len(points) != 3:
            return False, 0.0, {}
        best = None
        for lone_index in range(3):
            row_indices = [idx for idx in range(3) if idx != lone_index]
            a, b = points[row_indices]
            lone = points[lone_index]
            row_error = abs(a[1] - b[1]) / diameter
            row_mid = (a + b) / 2
            center_error = abs(lone[0] - row_mid[0]) / diameter
            row_gap = abs(lone[1] - row_mid[1]) / diameter
            pair_gap = abs(a[0] - b[0]) / diameter
            # A compact 品-shaped layout often has vertically overlapping cup
            # boxes under an oblique camera, so its center-row gap can be below
            # one full cup diameter.
            valid = row_error <= 0.5 and center_error <= 0.6 and 0.5 <= row_gap <= 3.0 and pair_gap >= 0.8
            penalty = row_error / 0.5 + center_error / 0.6 + abs(row_gap - 1.5) / 2.2
            confidence = max(0.0, min(1.0, 1.0 - penalty / 3.0))
            candidate = (valid, confidence, row_error, center_error, row_gap, pair_gap)
            if (
                best is None
                or (candidate[0] and not best[0])
                or (candidate[0] == best[0] and candidate[1] > best[1])
            ):
                best = candidate
        assert best is not None
        return best[0], best[1], {
            "pin_row_error_diameter": round(best[2], 4),
            "pin_center_error_diameter": round(best[3], 4),
            "pin_row_gap_diameter": round(best[4], 4),
            "pin_pair_gap_diameter": round(best[5], 4),
        }

    def update(self, context: FrameContext):
        evaluation = self.evaluate(context.detections)
        cups = [item for item in context.detections if _item_name(item) == "品茗杯"]
        evidence = _evidence(context, cups, evaluation.metrics)
        self._history.append((context.timestamp, evaluation.label, evaluation.confidence, evidence))
        cutoff = context.timestamp - self.stable_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        valid = [row for row in self._history if row[1] is not None]
        if evaluation.label is None:
            self._history.clear()
            self._candidate_label = None
            self._started_at = None
            self._started_event_emitted = False
            state = ObservationState.UNCERTAIN if len(cups) < 2 else ObservationState.IDLE
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, state,
                confidence=evaluation.confidence,
                value="其他布局" if len(cups) >= 2 else None,
                reason=evaluation.reason,
                updated_at=context.timestamp, metrics=evaluation.metrics,
            ), []

        if evaluation.label != self._candidate_label:
            self._candidate_label = evaluation.label
            self._started_at = context.timestamp
            self._started_event_emitted = False
        state = ObservationState.CANDIDATE
        started_events: List[ObservationEvent] = []
        if not self._started_event_emitted:
            self._started_event_emitted = True
            started_events.append(_event(
                self, context, EventPhase.STARTED, self._started_at,
                evaluation.confidence, evaluation.label, evaluation.metrics, [evidence],
            ))
        if len(self._history) >= self.min_samples and valid:
            matching = [row for row in valid if row[1] == evaluation.label]
            span = self._history[-1][0] - self._history[0][0]
            ratio = len(matching) / len(self._history)
            if span >= self.stable_seconds * 0.95 and ratio >= self.stable_ratio:
                confidence = float(np.mean([row[2] for row in matching]))
                completed_metrics = {
                    **evaluation.metrics,
                    "stable_ratio": round(ratio, 4),
                }
                completed_events = []
                if self._last_completed_label != evaluation.label:
                    completed_events.append(_event(
                        self, context, EventPhase.COMPLETED, self._started_at, confidence,
                        evaluation.label, completed_metrics,
                        [row[3] for row in matching],
                    ))
                    self._last_completed_label = evaluation.label
                return ObservationSnapshot(
                    self.observation_id, self.name, self.sop_step, ObservationState.COMPLETED,
                    confidence=confidence, value=evaluation.label,
                    reason="布局已稳定", started_at=self._started_at,
                    updated_at=context.timestamp, metrics=completed_metrics,
                ), started_events + completed_events
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step, state,
            confidence=evaluation.confidence, value=evaluation.label,
            reason="等待布局稳定", started_at=self._started_at,
            updated_at=context.timestamp, metrics=evaluation.metrics,
        ), started_events

    def reset(self) -> None:
        self._history.clear()
        self._last_completed_label = None
        self._candidate_label = None
        self._started_at = None
        self._started_event_emitted = False


class FilledCupTrayLayoutObservation:
    """Classify tea-filled cups after they have been placed on the tea tray."""

    observation_id = "result_filled_cup_tray_layout"
    name = "茶汤品茗杯茶盘布局"
    sop_step = 6
    camera_roles: Set[CameraRole] = {CameraRole.TABLETOP}
    rule_version = "1.0-experimental"

    def __init__(self, stable_seconds: float = 1.0, min_samples: int = 5, stable_ratio: float = 0.8):
        self.stable_seconds = stable_seconds
        self.min_samples = min_samples
        self.stable_ratio = stable_ratio
        self._history: Deque[Tuple[float, str, float, EvidenceFrame]] = deque()
        self._candidate_label: Optional[str] = None
        self._started_at: Optional[float] = None
        self._started_event_emitted = False
        self._last_completed_label: Optional[str] = None

    @staticmethod
    def _tray_contains(tray: Any, cup: Any) -> bool:
        tx, ty, tw, th = _bbox(tray)
        cx, cy = _centroid(cup)
        # A small expansion tolerates box edges and perspective at the tray rim.
        margin_x, margin_y = 0.06 * tw, 0.06 * th
        return tx - margin_x <= cx <= tx + tw + margin_x and ty - margin_y <= cy <= ty + th + margin_y

    @staticmethod
    def _filled_track_ids(context: FrameContext) -> Set[int]:
        filled: Set[int] = set()
        for value in context.extras.get("filled_cup_track_ids", []):
            try:
                filled.add(int(value))
            except (TypeError, ValueError):
                continue
        for row in context.extras.get("cup_liquid_states", []):
            if not bool(row.get("filled", row.get("liquid_present", False))):
                continue
            confidence = float(row.get("confidence", 1.0))
            if confidence < 0.5:
                continue
            track_id = row.get("track_id", row.get("cup_track_id"))
            if track_id is not None:
                try:
                    filled.add(int(track_id))
                except (TypeError, ValueError):
                    pass
        for event in context.extras.get("session_observation_events", []):
            if getattr(event, "observation_id", "") != "action_tea_distribution":
                continue
            phase = getattr(getattr(event, "phase", None), "value", None)
            if phase != EventPhase.COMPLETED.value:
                continue
            for value in (getattr(event, "metrics", {}) or {}).get("target_track_ids", []):
                try:
                    filled.add(int(value))
                except (TypeError, ValueError):
                    continue
        return filled

    @staticmethod
    def _is_filled(cup: Any, context: FrameContext, filled_ids: Set[int]) -> bool:
        track_id = getattr(cup, "track_id", None)
        if track_id is not None and int(track_id) in filled_ids:
            return True
        cx, cy = _centroid(cup)
        for row in context.extras.get("cup_liquid_states", []):
            if not bool(row.get("filled", row.get("liquid_present", False))):
                continue
            center = row.get("center", row.get("cup_center"))
            if center is None or len(center) < 2:
                continue
            diameter = max((_bbox(cup)[2] + _bbox(cup)[3]) / 2, 1.0)
            if hypot(cx - float(center[0]), cy - float(center[1])) <= 0.7 * diameter:
                return float(row.get("confidence", 1.0)) >= 0.5
        return False

    def _reset_candidate(self) -> None:
        self._history.clear()
        self._candidate_label = None
        self._started_at = None
        self._started_event_emitted = False

    def update(self, context: FrameContext):
        if "茶盘" not in context.model_classes or "品茗杯" not in context.model_classes:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                reason="当前模型需要支持茶盘和品茗杯", updated_at=context.timestamp,
                metrics={"required_classes": ["茶盘", "品茗杯"]}, experimental=True,
            ), []

        trays = [item for item in context.detections if _item_name(item) == "茶盘"]
        cups = [item for item in context.detections if _item_name(item) == "品茗杯"]
        filled_ids = self._filled_track_ids(context)
        tray = max(trays, key=lambda item: float(getattr(item, "confidence", 0.0)), default=None)
        if tray is None or len(cups) < 2:
            self._reset_candidate()
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                reason="需要同时检测到茶盘和至少两只品茗杯", updated_at=context.timestamp,
                metrics={"tray_count": len(trays), "cup_count": len(cups)}, experimental=True,
            ), []

        tray_cups = [cup for cup in cups if self._tray_contains(tray, cup)]
        filled_cups = [cup for cup in tray_cups if self._is_filled(cup, context, filled_ids)]
        if len(filled_cups) < 2:
            self._reset_candidate()
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                reason="缺少至少两只已确认有茶汤且位于茶盘内的品茗杯",
                updated_at=context.timestamp,
                metrics={
                    "tray_cup_count": len(tray_cups),
                    "filled_cup_count": len(filled_cups),
                    "content_evidence": bool(filled_ids or context.extras.get("cup_liquid_states")),
                    "liquid_evidence": bool(context.extras.get("cup_liquid_states")),
                },
                experimental=True,
            ), []

        evaluation = CupLayoutObservation.evaluate(filled_cups)
        label = evaluation.label or "不规范摆放"
        metrics = {
            **evaluation.metrics,
            "classification": label,
            "tray_cup_count": len(tray_cups),
            "filled_cup_count": len(filled_cups),
            "content_evidence": True,
            "liquid_verified": bool(context.extras.get("cup_liquid_states")),
            "distribution_gesture_verified": bool(filled_ids),
            "layout_valid": evaluation.label is not None,
        }
        confidence = float(evaluation.confidence)
        evidence = _evidence(context, [tray, *filled_cups], metrics)
        self._history.append((context.timestamp, label, confidence, evidence))
        cutoff = context.timestamp - self.stable_seconds - 1e-6
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if label != self._candidate_label:
            self._candidate_label = label
            self._started_at = context.timestamp
            self._started_event_emitted = False
        emitted: List[ObservationEvent] = []
        if not self._started_event_emitted:
            self._started_event_emitted = True
            emitted.append(_event(
                self, context, EventPhase.STARTED, self._started_at or context.timestamp,
                confidence, label, metrics, [evidence],
            ))
        matching = [row for row in self._history if row[1] == label]
        span = self._history[-1][0] - self._history[0][0] if len(self._history) > 1 else 0.0
        ratio = len(matching) / len(self._history) if self._history else 0.0
        if len(self._history) >= self.min_samples and span >= self.stable_seconds * 0.95 and ratio >= self.stable_ratio:
            completed_confidence = float(np.mean([row[2] for row in matching]))
            completed_metrics = {**metrics, "stable_ratio": round(ratio, 4)}
            completed_events: List[ObservationEvent] = []
            if self._last_completed_label != label:
                completed_events.append(_event(
                    self, context, EventPhase.COMPLETED, self._started_at or context.timestamp,
                    completed_confidence, label, completed_metrics, [row[3] for row in matching],
                ))
                self._last_completed_label = label
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.COMPLETED,
                confidence=completed_confidence, value=label,
                reason=("品茗杯已在茶盘内稳定形成合规布局" if evaluation.label
                        else "已检测到茶盘内品茗杯不规范摆放"),
                started_at=self._started_at, updated_at=context.timestamp,
                metrics=completed_metrics, experimental=True,
            ), emitted + completed_events
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step, ObservationState.CANDIDATE,
            confidence=confidence, value=label, reason="等待茶盘内杯位稳定",
            started_at=self._started_at, updated_at=context.timestamp,
            metrics={**metrics, "stable_ratio": round(ratio, 4)}, experimental=True,
        ), emitted

    def reset(self) -> None:
        self._reset_candidate()
        self._last_completed_label = None


@dataclass(frozen=True)
class PourStage:
    key: str
    label: str
    source_names: Set[str]
    target_names: Set[str]


@dataclass
class PourEvaluation:
    stage_index: int
    passed: Optional[bool]
    confidence: float
    source: Optional[Any]
    target: Optional[Any]
    target_key: Optional[str]
    metrics: Dict[str, Any]
    reason: str


class WarmCleanSequenceObservation:
    """Experimental warm-clean order: gaiwan -> fairness pitcher -> cups.

    Geometry is only a low-confidence fallback. A future liquid-flow module can
    publish ``extras["pour_interactions"]`` rows with ``source``, ``target``,
    ``confidence`` and an optional ``target_track_id``/``target_id``.
    """

    observation_id = "seq_warm_clean_order"
    name = "温杯洁具顺序"
    sop_step = 2
    camera_roles: Set[CameraRole] = {CameraRole.TABLETOP, CameraRole.SIDE}
    rule_version = "1.0-experimental"
    BODY_NAMES = {"盖碗", "盖碗碗身", "盖碗（碗身）"}
    STAGES = (
        PourStage("gaiwan", "盖碗", {"烧水壶"}, BODY_NAMES),
        PourStage("fairness", "公道杯", BODY_NAMES, {"公道杯"}),
        PourStage("cups", "品茗杯", {"公道杯"}, {"品茗杯"}),
    )

    def __init__(
        self,
        stable_seconds: float = 0.5,
        min_samples: int = 3,
        min_cups: int = 2,
        stable_ratio: float = 0.7,
        interaction_confidence: float = 0.45,
    ):
        self.stable_seconds = stable_seconds
        self.min_samples = min_samples
        self.min_cups = min_cups
        self.stable_ratio = stable_ratio
        self.interaction_confidence = interaction_confidence
        self._stage_index = 0
        self._candidate_key: Optional[str] = None
        self._candidate_history: Deque[Tuple[float, bool, float, EvidenceFrame]] = deque()
        self._wrong_key: Optional[str] = None
        self._wrong_history: Deque[Tuple[float, bool, float, EvidenceFrame]] = deque()
        self._motion_history: Dict[str, Deque[Tuple[float, float, float]]] = {}
        self._completed_labels: List[str] = []
        self._completed_confidences: List[float] = []
        self._cup_targets: Set[str] = set()
        self._evidence: List[EvidenceFrame] = []
        self._started_at: Optional[float] = None
        self._started_event_emitted = False
        self._completed = False

    @classmethod
    def required_classes_available(cls, classes: Set[str]) -> bool:
        return (
            "烧水壶" in classes
            and bool(cls.BODY_NAMES & classes)
            and "公道杯" in classes
            and "品茗杯" in classes
        )

    @staticmethod
    def _hand_center(hand: Dict[str, Any]) -> Tuple[float, float]:
        center = hand.get("center")
        if center is not None:
            return float(center[0]), float(center[1])
        x, y, w, h = hand.get("bbox", (0, 0, 0, 0))
        return float(x + w / 2), float(y + h / 2)

    @staticmethod
    def _bbox_gap(left: Any, right: Any) -> float:
        lx, ly, lw, lh = _bbox(left)
        rx, ry, rw, rh = _bbox(right)
        dx = max(rx - (lx + lw), lx - (rx + rw), 0.0)
        dy = max(ry - (ly + lh), ly - (ry + rh), 0.0)
        return hypot(dx, dy)

    @staticmethod
    def _target_key(stage: PourStage, target: Any) -> str:
        track_id = getattr(target, "track_id", None)
        if track_id is not None:
            return f"{stage.key}:track:{int(track_id)}"
        cx, cy = _centroid(target)
        _, _, w, h = _bbox(target)
        return (
            f"{stage.key}:position:"
            f"{round(cx / max(w * 0.75, 1.0))}:"
            f"{round(cy / max(h * 0.75, 1.0))}"
        )

    def _direct_evaluation(
        self, context: FrameContext, stage_index: int, stage: PourStage
    ) -> Optional[PourEvaluation]:
        for row in context.extras.get("pour_interactions", []):
            source_name = str(row.get("source", ""))
            target_name = str(row.get("target", ""))
            if source_name not in stage.source_names or target_name not in stage.target_names:
                continue
            confidence = float(row.get("confidence", 1.0))
            active = bool(row.get("active", True))
            target_identity = row.get("target_track_id", row.get("target_id"))
            if target_identity is None:
                center = row.get("target_center")
                target_identity = (
                    f"{round(float(center[0]))}:{round(float(center[1]))}"
                    if center is not None and len(center) >= 2
                    else target_name
                )
            metrics = {
                "stage": stage.key,
                "stage_label": stage.label,
                "signal_source": str(row.get("signal_source", "liquid_or_external")),
                "source_name": source_name,
                "target_name": target_name,
                "target_key": f"{stage.key}:direct:{target_identity}",
            }
            return PourEvaluation(
                stage_index,
                active and confidence >= self.interaction_confidence,
                confidence,
                None,
                None,
                str(metrics["target_key"]),
                metrics,
                "检测到液流/外部倒水关系" if active else "倒水关系未激活",
            )
        return None

    def _motion_ratio(
        self,
        stage: PourStage,
        context: FrameContext,
        source: Any,
        scale: float,
    ) -> float:
        key = f"{stage.key}:{_item_name(source)}"
        history = self._motion_history.setdefault(key, deque())
        cx, cy = _centroid(source)
        history.append((context.timestamp, cx, cy))
        cutoff = context.timestamp - 1.5
        while history and history[0][0] < cutoff:
            history.popleft()
        if len(history) < 2:
            return 0.0
        return max(
            hypot(cx - old_x, cy - old_y) / max(scale, 1.0)
            for _, old_x, old_y in history
        )

    def _geometry_evaluation(
        self, context: FrameContext, stage_index: int, stage: PourStage
    ) -> PourEvaluation:
        sources = [item for item in context.detections if _item_name(item) in stage.source_names]
        targets = [item for item in context.detections if _item_name(item) in stage.target_names]
        if not sources or not targets:
            return PourEvaluation(
                stage_index,
                None,
                0.0,
                None,
                None,
                None,
                {
                    "stage": stage.key,
                    "stage_label": stage.label,
                    "source_count": len(sources),
                    "target_count": len(targets),
                    "signal_source": "geometry",
                },
                f"等待看清{stage.label}及倒水源器具",
            )

        best: Optional[PourEvaluation] = None
        for source in sources:
            for target in targets:
                sx, sy, sw, sh = _bbox(source)
                tx, ty, tw, th = _bbox(target)
                source_diag = max(hypot(sw, sh), 1.0)
                target_diag = max(hypot(tw, th), 1.0)
                interaction_scale = max(target_diag, source_diag * 0.45)
                gap_ratio = self._bbox_gap(source, target) / interaction_scale
                scx, scy = _centroid(source)
                tcx, tcy = _centroid(target)
                center_ratio = hypot(scx - tcx, scy - tcy) / interaction_scale
                source_above = (
                    context.camera_role is not CameraRole.SIDE
                    or scy <= tcy + 0.3 * th
                )
                hands = [
                    hand for hand in context.hand_results
                    if float(hand.get("confidence", 0.0)) >= 0.5
                ]
                hand_distances = []
                for hand in hands:
                    hx, hy = self._hand_center(hand)
                    dx = max(sx - hx, 0.0, hx - (sx + sw))
                    dy = max(sy - hy, 0.0, hy - (sy + sh))
                    hand_distances.append(hypot(dx, dy) / source_diag)
                hand_near_source = bool(hand_distances) and min(hand_distances) <= 0.55
                motion_ratio = self._motion_ratio(
                    stage, context, source, max(source_diag, target_diag)
                )
                source_moved = motion_ratio >= 0.05
                proximity_valid = gap_ratio <= 0.9 and center_ratio <= 2.8
                passed = proximity_valid and source_above and hand_near_source and source_moved
                base_confidence = min(
                    float(getattr(source, "confidence", 0.0)),
                    float(getattr(target, "confidence", 0.0)),
                )
                confidence = min(0.75, base_confidence * (0.9 if passed else 0.45))
                metrics = {
                    "stage": stage.key,
                    "stage_label": stage.label,
                    "signal_source": "geometry",
                    "source_name": _item_name(source),
                    "target_name": _item_name(target),
                    "target_key": self._target_key(stage, target),
                    "source_target_gap_ratio": round(gap_ratio, 4),
                    "source_target_center_ratio": round(center_ratio, 4),
                    "source_above_target": source_above,
                    "hand_near_source": hand_near_source,
                    "source_motion_ratio": round(motion_ratio, 4),
                    "source_moved": source_moved,
                }
                reason = (
                    f"检测到{_item_name(source)}向{stage.label}倒水的几何关系"
                    if passed
                    else f"等待{_item_name(source)}靠近并向{stage.label}倒水"
                )
                evaluation = PourEvaluation(
                    stage_index,
                    passed,
                    confidence,
                    source,
                    target,
                    str(metrics["target_key"]),
                    metrics,
                    reason,
                )
                if best is None or evaluation.confidence > best.confidence:
                    best = evaluation
        assert best is not None
        return best

    def _evaluate_stage(
        self, context: FrameContext, stage_index: int
    ) -> PourEvaluation:
        stage = self.STAGES[stage_index]
        direct = self._direct_evaluation(context, stage_index, stage)
        return direct if direct is not None else self._geometry_evaluation(
            context, stage_index, stage
        )

    def _append_history(
        self,
        history: Deque[Tuple[float, bool, float, EvidenceFrame]],
        context: FrameContext,
        passed: bool,
        confidence: float,
        evidence: EvidenceFrame,
    ) -> None:
        history.append((context.timestamp, passed, confidence, evidence))
        cutoff = context.timestamp - self.stable_seconds
        while history and history[0][0] < cutoff:
            history.popleft()

    def _history_stable(
        self, history: Deque[Tuple[float, bool, float, EvidenceFrame]]
    ) -> bool:
        if len(history) < self.min_samples:
            return False
        span = history[-1][0] - history[0][0]
        ratio = sum(1 for row in history if row[1]) / len(history)
        return span >= self.stable_seconds * 0.95 and ratio >= self.stable_ratio

    def _progress_metrics(self, evaluation: Optional[PourEvaluation] = None) -> Dict[str, Any]:
        expected = (
            self.STAGES[self._stage_index].label
            if self._stage_index < len(self.STAGES)
            else None
        )
        metrics: Dict[str, Any] = {
            "expected_target": expected,
            "completed_targets": list(self._completed_labels),
            "cup_targets_completed": len(self._cup_targets),
            "minimum_cups": self.min_cups,
            "required_order": [stage.label for stage in self.STAGES],
        }
        if evaluation is not None:
            metrics.update(evaluation.metrics)
        return metrics

    def _reset_candidate(self) -> None:
        self._candidate_key = None
        self._candidate_history.clear()

    def _sequence_value(self) -> str:
        return "→".join(stage.label for stage in self.STAGES)

    def update(self, context: FrameContext):
        if not self.required_classes_available(context.model_classes):
            return ObservationSnapshot(
                self.observation_id,
                self.name,
                self.sop_step,
                ObservationState.UNCERTAIN,
                reason="当前模型需要同时支持烧水壶、盖碗碗身、公道杯和品茗杯",
                updated_at=context.timestamp,
                metrics={"required_capability": "warm_clean_utensils"},
                experimental=True,
            ), []

        if self._completed:
            return ObservationSnapshot(
                self.observation_id,
                self.name,
                self.sop_step,
                ObservationState.COMPLETED,
                confidence=float(np.mean(self._completed_confidences)),
                value=self._sequence_value(),
                reason="温杯洁具顺序已完成",
                started_at=self._started_at,
                updated_at=context.timestamp,
                metrics=self._progress_metrics(),
                experimental=True,
            ), []

        evaluations = [
            self._evaluate_stage(context, index) for index in range(len(self.STAGES))
        ]
        expected = evaluations[self._stage_index]
        expected_stage = self.STAGES[self._stage_index]
        evidence_items = [item for item in (expected.source, expected.target) if item is not None]
        evidence = _evidence(context, evidence_items, self._progress_metrics(expected))
        emitted: List[ObservationEvent] = []

        if expected.passed is True:
            self._wrong_key = None
            self._wrong_history.clear()
            candidate_key = f"{expected.stage_index}:{expected.target_key}"
            if candidate_key != self._candidate_key:
                self._candidate_key = candidate_key
                self._candidate_history.clear()
            self._append_history(
                self._candidate_history,
                context,
                True,
                expected.confidence,
                evidence,
            )
            if self._started_at is None:
                self._started_at = context.timestamp
            if not self._started_event_emitted:
                self._started_event_emitted = True
                emitted.append(_event(
                    self,
                    context,
                    EventPhase.STARTED,
                    self._started_at,
                    expected.confidence,
                    expected_stage.label,
                    self._progress_metrics(expected),
                    [evidence],
                ))

            if self._history_stable(self._candidate_history):
                positives = [row for row in self._candidate_history if row[1]]
                stage_confidence = float(np.mean([row[2] for row in positives]))
                self._evidence.extend(row[3] for row in positives)
                if expected_stage.key == "cups":
                    assert expected.target_key is not None
                    self._cup_targets.add(expected.target_key)
                    self._reset_candidate()
                    if len(self._cup_targets) < self.min_cups:
                        return ObservationSnapshot(
                            self.observation_id,
                            self.name,
                            self.sop_step,
                            ObservationState.ACTIVE,
                            confidence=stage_confidence,
                            value=f"已温杯 {len(self._cup_targets)}/{self.min_cups}",
                            reason="继续温洁下一只品茗杯",
                            started_at=self._started_at,
                            updated_at=context.timestamp,
                            metrics=self._progress_metrics(expected),
                            experimental=True,
                        ), emitted
                self._completed_labels.append(expected_stage.label)
                self._completed_confidences.append(stage_confidence)
                self._stage_index += 1
                self._reset_candidate()
                if self._stage_index >= len(self.STAGES):
                    self._completed = True
                    confidence = float(np.mean(self._completed_confidences))
                    metrics = self._progress_metrics(expected)
                    event = _event(
                        self,
                        context,
                        EventPhase.COMPLETED,
                        self._started_at,
                        confidence,
                        self._sequence_value(),
                        metrics,
                        self._evidence,
                    )
                    return ObservationSnapshot(
                        self.observation_id,
                        self.name,
                        self.sop_step,
                        ObservationState.COMPLETED,
                        confidence=confidence,
                        value=event.value,
                        reason=f"按顺序完成{self._sequence_value()}温洁流程",
                        started_at=self._started_at,
                        updated_at=context.timestamp,
                        metrics=metrics,
                        experimental=True,
                    ), emitted + [event]
                return ObservationSnapshot(
                    self.observation_id,
                    self.name,
                    self.sop_step,
                    ObservationState.ACTIVE,
                    confidence=stage_confidence,
                    value=f"下一目标：{self.STAGES[self._stage_index].label}",
                    reason=f"已完成{expected_stage.label}，等待下一目标",
                    started_at=self._started_at,
                    updated_at=context.timestamp,
                    metrics=self._progress_metrics(expected),
                    experimental=True,
                ), emitted

            return ObservationSnapshot(
                self.observation_id,
                self.name,
                self.sop_step,
                ObservationState.CANDIDATE,
                confidence=expected.confidence,
                value=f"当前目标：{expected_stage.label}",
                reason="等待倒水关系稳定",
                started_at=self._started_at,
                updated_at=context.timestamp,
                metrics=self._progress_metrics(expected),
                experimental=True,
            ), emitted

        if self._candidate_history:
            self._append_history(
                self._candidate_history,
                context,
                False,
                expected.confidence,
                evidence,
            )
            if not any(row[1] for row in self._candidate_history):
                self._reset_candidate()

        wrong = max(
            (
                row for row in evaluations
                if row.stage_index != self._stage_index and row.passed is True
            ),
            key=lambda row: row.confidence,
            default=None,
        )
        if wrong is not None:
            wrong_stage = self.STAGES[wrong.stage_index]
            wrong_key = f"{wrong.stage_index}:{wrong.target_key}"
            wrong_items = [item for item in (wrong.source, wrong.target) if item is not None]
            wrong_evidence = _evidence(
                context, wrong_items, self._progress_metrics(wrong)
            )
            if wrong_key != self._wrong_key:
                self._wrong_key = wrong_key
                self._wrong_history.clear()
            self._append_history(
                self._wrong_history,
                context,
                True,
                wrong.confidence,
                wrong_evidence,
            )
            if self._history_stable(self._wrong_history):
                return ObservationSnapshot(
                    self.observation_id,
                    self.name,
                    self.sop_step,
                    ObservationState.UNCERTAIN,
                    confidence=wrong.confidence,
                    value=f"疑似错序：{wrong_stage.label}",
                    reason=f"当前应处理{expected_stage.label}，但检测到{wrong_stage.label}倒水关系",
                    started_at=self._started_at,
                    updated_at=context.timestamp,
                    metrics={
                        **self._progress_metrics(wrong),
                        "out_of_order": True,
                        "observed_target": wrong_stage.label,
                    },
                    experimental=True,
                ), []
        else:
            self._wrong_key = None
            self._wrong_history.clear()

        state = ObservationState.ACTIVE if self._completed_labels else ObservationState.IDLE
        return ObservationSnapshot(
            self.observation_id,
            self.name,
            self.sop_step,
            state,
            confidence=expected.confidence,
            value=f"等待{expected_stage.label}",
            reason=expected.reason,
            started_at=self._started_at,
            updated_at=context.timestamp,
            metrics=self._progress_metrics(expected),
            experimental=True,
        ), []

    def reset(self) -> None:
        self._stage_index = 0
        self._candidate_key = None
        self._candidate_history.clear()
        self._wrong_key = None
        self._wrong_history.clear()
        self._motion_history.clear()
        self._completed_labels.clear()
        self._completed_confidences.clear()
        self._cup_targets.clear()
        self._evidence.clear()
        self._started_at = None
        self._started_event_emitted = False
        self._completed = False


class TeaCanisterToLotusObservation:
    """Observe one hand moving from the tea canister to the tea lotus.

    This proves the transfer gesture, not the presence or weight of tea leaves.
    A future leaf/utensil model can publish a stronger direct interaction through
    ``extras["tea_transfer_interactions"]`` without changing the event contract.
    """

    observation_id = "action_tea_canister_to_lotus"
    name = "从茶叶罐取茶至茶荷"
    sop_step = 3
    camera_roles: Set[CameraRole] = {CameraRole.SIDE}
    rule_version = "1.0-experimental"

    def __init__(
        self,
        source_dwell_seconds: float = 0.4,
        target_dwell_seconds: float = 0.5,
        transfer_timeout_seconds: float = 6.0,
        min_samples: int = 3,
        stable_ratio: float = 0.7,
    ):
        self.source_dwell_seconds = source_dwell_seconds
        self.target_dwell_seconds = target_dwell_seconds
        self.transfer_timeout_seconds = transfer_timeout_seconds
        self.min_samples = min_samples
        self.stable_ratio = stable_ratio
        self._stage = "waiting_source"
        self._source_history: Deque[Tuple[float, bool, float, EvidenceFrame]] = deque()
        self._target_history: Deque[Tuple[float, bool, float, EvidenceFrame]] = deque()
        self._direct_history: Deque[Tuple[float, bool, float, EvidenceFrame]] = deque()
        self._active_hand_center: Optional[Tuple[float, float]] = None
        self._started_at: Optional[float] = None
        self._left_source = False
        self._completed = False
        self._started_event_emitted = False
        self._evidence: List[EvidenceFrame] = []
        self._completion_confidence = 0.0
        self._completion_metrics: Dict[str, Any] = {}

    @staticmethod
    def required_classes_available(classes: Set[str]) -> bool:
        return "茶叶罐" in classes and "茶荷" in classes

    @staticmethod
    def _hand_center(hand: Dict[str, Any]) -> Tuple[float, float]:
        center = hand.get("center")
        if center is not None:
            return float(center[0]), float(center[1])
        x, y, w, h = hand.get("bbox", (0, 0, 0, 0))
        return float(x + w / 2), float(y + h / 2)

    @staticmethod
    def _hand_item_gap_ratio(hand: Dict[str, Any], item: Any) -> float:
        hx, hy, hw, hh = [float(value) for value in hand.get("bbox", (0, 0, 0, 0))]
        x, y, w, h = _bbox(item)
        dx = max(x - (hx + hw), hx - (x + w), 0.0)
        dy = max(y - (hy + hh), hy - (y + h), 0.0)
        return hypot(dx, dy) / max(hypot(w, h), 1.0)

    def _append_history(
        self,
        history: Deque[Tuple[float, bool, float, EvidenceFrame]],
        context: FrameContext,
        passed: bool,
        confidence: float,
        evidence: EvidenceFrame,
        window_seconds: float,
    ) -> None:
        history.append((context.timestamp, passed, confidence, evidence))
        cutoff = context.timestamp - window_seconds
        while history and history[0][0] < cutoff:
            history.popleft()

    def _history_stable(
        self,
        history: Deque[Tuple[float, bool, float, EvidenceFrame]],
        window_seconds: float,
    ) -> bool:
        if len(history) < self.min_samples:
            return False
        span = history[-1][0] - history[0][0]
        ratio = sum(1 for row in history if row[1]) / len(history)
        return span >= window_seconds * 0.95 and ratio >= self.stable_ratio

    def _select_hand(
        self, hands: Sequence[Dict[str, Any]], reference: Optional[Tuple[float, float]]
    ) -> Optional[Dict[str, Any]]:
        valid = [hand for hand in hands if float(hand.get("confidence", 0.0)) >= 0.55]
        if not valid:
            return None
        if reference is None:
            return max(valid, key=lambda hand: float(hand.get("confidence", 0.0)))
        return min(
            valid,
            key=lambda hand: hypot(
                self._hand_center(hand)[0] - reference[0],
                self._hand_center(hand)[1] - reference[1],
            ),
        )

    def _direct_signal(self, context: FrameContext) -> Optional[Tuple[bool, float, str]]:
        matches = []
        for row in context.extras.get("tea_transfer_interactions", []):
            if str(row.get("source", "")) != "茶叶罐" or str(row.get("target", "")) != "茶荷":
                continue
            confidence = float(row.get("confidence", 0.0))
            active = bool(row.get("active", True)) and confidence >= 0.5
            matches.append((active, confidence, str(row.get("signal_source", "external"))))
        return max(matches, key=lambda value: (value[0], value[1]), default=None)

    def _completed_snapshot(
        self, context: FrameContext, confidence: float, metrics: Dict[str, Any]
    ) -> ObservationSnapshot:
        return ObservationSnapshot(
            self.observation_id,
            self.name,
            self.sop_step,
            ObservationState.COMPLETED,
            confidence=confidence,
            value=True,
            reason="已检测到从茶叶罐取茶并转移至茶荷的动作",
            started_at=self._started_at,
            updated_at=context.timestamp,
            metrics=metrics,
            experimental=True,
        )

    def update(self, context: FrameContext):
        if not self.required_classes_available(context.model_classes):
            return ObservationSnapshot(
                self.observation_id,
                self.name,
                self.sop_step,
                ObservationState.UNCERTAIN,
                reason="当前模型需要同时支持茶叶罐和茶荷",
                updated_at=context.timestamp,
                metrics={"required_classes": ["茶叶罐", "茶荷"]},
                experimental=True,
            ), []

        if self._completed:
            return self._completed_snapshot(
                context,
                self._completion_confidence,
                self._completion_metrics,
            ), []

        canisters = [item for item in context.detections if _item_name(item) == "茶叶罐"]
        lotuses = [item for item in context.detections if _item_name(item) == "茶荷"]
        if not canisters or not lotuses:
            return ObservationSnapshot(
                self.observation_id,
                self.name,
                self.sop_step,
                ObservationState.UNCERTAIN,
                reason="需要同时看清茶叶罐和茶荷",
                updated_at=context.timestamp,
                metrics={
                    "canister_count": len(canisters),
                    "lotus_count": len(lotuses),
                    "stage": self._stage,
                },
                experimental=True,
            ), []

        canister = max(canisters, key=lambda item: float(getattr(item, "confidence", 0.0)))
        lotus = max(lotuses, key=lambda item: float(getattr(item, "confidence", 0.0)))
        source_target_distance = hypot(
            _centroid(canister)[0] - _centroid(lotus)[0],
            _centroid(canister)[1] - _centroid(lotus)[1],
        )
        source_target_scale = max(hypot(*_bbox(canister)[2:]), hypot(*_bbox(lotus)[2:]), 1.0)
        separation_ratio = source_target_distance / source_target_scale
        base_metrics = {
            "stage": self._stage,
            "source_target_separation": round(separation_ratio, 4),
            "content_verified": False,
        }
        items = [canister, lotus]

        direct = self._direct_signal(context)
        if direct is not None:
            active, confidence, signal_source = direct
            metrics = {**base_metrics, "signal_source": signal_source, "content_verified": True}
            evidence = _evidence(context, items, metrics)
            self._append_history(
                self._direct_history,
                context,
                active,
                confidence,
                evidence,
                self.target_dwell_seconds,
            )
            emitted: List[ObservationEvent] = []
            if active and self._started_at is None:
                self._started_at = context.timestamp
            if active and not self._started_event_emitted:
                self._started_event_emitted = True
                emitted.append(_event(
                    self,
                    context,
                    EventPhase.STARTED,
                    self._started_at or context.timestamp,
                    confidence,
                    True,
                    metrics,
                    [evidence],
                ))
            if self._history_stable(self._direct_history, self.target_dwell_seconds):
                positives = [row for row in self._direct_history if row[1]]
                event_confidence = float(np.mean([row[2] for row in positives]))
                self._completed = True
                self._target_history = deque(positives)
                self._completion_confidence = event_confidence
                self._completion_metrics = {**metrics, "stage": "completed"}
                event = _event(
                    self,
                    context,
                    EventPhase.COMPLETED,
                    self._started_at or context.timestamp,
                    event_confidence,
                    True,
                    metrics,
                    [row[3] for row in positives],
                )
                return self._completed_snapshot(
                    context, event_confidence, self._completion_metrics
                ), emitted + [event]
            return ObservationSnapshot(
                self.observation_id,
                self.name,
                self.sop_step,
                ObservationState.CANDIDATE if active else ObservationState.IDLE,
                confidence=confidence,
                value="检测到茶叶转移证据" if active else None,
                reason="等待茶叶转移证据稳定",
                started_at=self._started_at,
                updated_at=context.timestamp,
                metrics=metrics,
                experimental=True,
            ), emitted

        hands = [hand for hand in context.hand_results if float(hand.get("confidence", 0.0)) >= 0.55]
        if not hands:
            return ObservationSnapshot(
                self.observation_id,
                self.name,
                self.sop_step,
                ObservationState.UNCERTAIN,
                reason="需要看清执行取茶动作的手",
                started_at=self._started_at,
                updated_at=context.timestamp,
                metrics={**base_metrics, "hand_count": 0},
                experimental=True,
            ), []

        if separation_ratio < 0.65:
            return ObservationSnapshot(
                self.observation_id,
                self.name,
                self.sop_step,
                ObservationState.UNCERTAIN,
                reason="茶叶罐与茶荷距离过近，无法确认手部转移轨迹",
                updated_at=context.timestamp,
                metrics=base_metrics,
                experimental=True,
            ), []

        emitted: List[ObservationEvent] = []
        if self._stage == "waiting_source":
            hand = min(hands, key=lambda value: self._hand_item_gap_ratio(value, canister))
            source_gap = self._hand_item_gap_ratio(hand, canister)
            source_contact = source_gap <= 0.12
            confidence = min(
                float(hand.get("confidence", 0.0)),
                float(getattr(canister, "confidence", 0.0)),
                float(getattr(lotus, "confidence", 0.0)),
            )
            metrics = {**base_metrics, "source_hand_gap": round(source_gap, 4)}
            evidence = _evidence(context, items, metrics)
            self._append_history(
                self._source_history,
                context,
                source_contact,
                confidence,
                evidence,
                self.source_dwell_seconds,
            )
            if self._history_stable(self._source_history, self.source_dwell_seconds):
                positives = [row for row in self._source_history if row[1]]
                self._stage = "moving_to_lotus"
                self._started_at = positives[0][0]
                self._active_hand_center = self._hand_center(hand)
                self._evidence.extend(row[3] for row in positives)
                self._started_event_emitted = True
                emitted.append(_event(
                    self,
                    context,
                    EventPhase.STARTED,
                    self._started_at,
                    float(np.mean([row[2] for row in positives])),
                    "已从茶叶罐取茶",
                    {**metrics, "stage": self._stage},
                    [row[3] for row in positives],
                ))
                return ObservationSnapshot(
                    self.observation_id,
                    self.name,
                    self.sop_step,
                    ObservationState.ACTIVE,
                    confidence=confidence,
                    value="已接触茶叶罐，等待移至茶荷",
                    reason="继续将取出的茶叶移动到茶荷",
                    started_at=self._started_at,
                    updated_at=context.timestamp,
                    metrics={**metrics, "stage": self._stage},
                    experimental=True,
                ), emitted
            state = ObservationState.CANDIDATE if source_contact else ObservationState.IDLE
            return ObservationSnapshot(
                self.observation_id,
                self.name,
                self.sop_step,
                state,
                confidence=confidence,
                value="正在茶叶罐取茶" if source_contact else None,
                reason="等待手在茶叶罐处完成取茶动作",
                updated_at=context.timestamp,
                metrics=metrics,
                experimental=True,
            ), []

        assert self._started_at is not None
        if context.timestamp - self._started_at > self.transfer_timeout_seconds:
            self._stage = "waiting_source"
            self._source_history.clear()
            self._target_history.clear()
            self._active_hand_center = None
            self._started_at = None
            self._left_source = False
            self._started_event_emitted = False
            return ObservationSnapshot(
                self.observation_id,
                self.name,
                self.sop_step,
                ObservationState.UNCERTAIN,
                reason="取茶后未在规定时间内移动到茶荷，请重新操作",
                updated_at=context.timestamp,
                metrics={**base_metrics, "transfer_timeout": True},
                experimental=True,
            ), []

        hand = self._select_hand(hands, self._active_hand_center)
        assert hand is not None
        self._active_hand_center = self._hand_center(hand)
        source_gap = self._hand_item_gap_ratio(hand, canister)
        target_gap = self._hand_item_gap_ratio(hand, lotus)
        self._left_source = self._left_source or source_gap >= 0.2
        target_contact = self._left_source and target_gap <= 0.12
        confidence = min(
            float(hand.get("confidence", 0.0)),
            float(getattr(canister, "confidence", 0.0)),
            float(getattr(lotus, "confidence", 0.0)),
        ) * 0.85
        metrics = {
            **base_metrics,
            "stage": self._stage,
            "source_hand_gap": round(source_gap, 4),
            "target_hand_gap": round(target_gap, 4),
            "left_source": self._left_source,
        }
        evidence = _evidence(context, items, metrics)
        self._append_history(
            self._target_history,
            context,
            target_contact,
            confidence,
            evidence,
            self.target_dwell_seconds,
        )
        if self._history_stable(self._target_history, self.target_dwell_seconds):
            positives = [row for row in self._target_history if row[1]]
            event_confidence = float(np.mean([row[2] for row in positives]))
            self._completed = True
            self._completion_confidence = event_confidence
            self._completion_metrics = {**metrics, "stage": "completed"}
            self._evidence.extend(row[3] for row in positives)
            event = _event(
                self,
                context,
                EventPhase.COMPLETED,
                self._started_at,
                event_confidence,
                True,
                metrics,
                self._evidence,
            )
            return self._completed_snapshot(
                context, event_confidence, self._completion_metrics
            ), [event]
        return ObservationSnapshot(
            self.observation_id,
            self.name,
            self.sop_step,
            ObservationState.CANDIDATE if target_contact else ObservationState.ACTIVE,
            confidence=confidence,
            value="正在放入茶荷" if target_contact else "正在转移茶叶",
            reason="等待手在茶荷处稳定，确认转移动作完成",
            started_at=self._started_at,
            updated_at=context.timestamp,
            metrics=metrics,
            experimental=True,
        ), []

    def reset(self) -> None:
        self._stage = "waiting_source"
        self._source_history.clear()
        self._target_history.clear()
        self._direct_history.clear()
        self._active_hand_center = None
        self._started_at = None
        self._left_source = False
        self._completed = False
        self._started_event_emitted = False
        self._evidence.clear()
        self._completion_confidence = 0.0
        self._completion_metrics.clear()


class TwoHandHoldObservation:
    rule_version = "1.2-experimental"
    camera_roles: Set[CameraRole] = {CameraRole.SIDE}

    def __init__(
        self,
        target_class: str = "茶荷",
        stable_seconds: Optional[float] = None,
        stable_ratio: float = 0.7,
        release_seconds: float = 0.5,
        min_lift_height_ratio: float = 0.6,
    ):
        self.target_class = target_class
        suffix = {"茶荷": "lotus", "茶盘": "tray"}.get(target_class, target_class)
        self.observation_id = f"action_hold_{suffix}"
        self.name = "双手托举茶荷赏茶" if target_class == "茶荷" else f"双手托举{target_class}"
        self.sop_step = 4 if target_class == "茶荷" else 6
        self.stable_seconds = (
            float(stable_seconds)
            if stable_seconds is not None
            else (5.0 if target_class == "茶荷" else 0.8)
        )
        self.stable_ratio = stable_ratio
        self.release_seconds = release_seconds
        self.min_lift_height_ratio = min_lift_height_ratio
        self._history: Deque[Tuple[float, bool, float, EvidenceFrame]] = deque()
        self._rest_y_samples: Deque[float] = deque(maxlen=20)
        self._resting_center_y: Optional[float] = None
        self._started_at: Optional[float] = None
        self._completed = False
        self._release_started: Optional[float] = None
        self._started_event_emitted = False

    @staticmethod
    def _palm_center(hand: Dict[str, Any]) -> Tuple[float, float]:
        landmarks = hand.get("landmarks")
        if landmarks is not None and len(landmarks) >= 18:
            palm = np.asarray(landmarks)[[0, 5, 9, 13, 17], :2]
            return float(palm[:, 0].mean()), float(palm[:, 1].mean())
        center = hand.get("center", (0.0, 0.0))
        return float(center[0]), float(center[1])

    def _update_resting_baseline(
        self,
        target: Any,
        hands: Sequence[Dict[str, Any]],
    ) -> None:
        x, y, w, h = _bbox(target)
        cx, cy = _centroid(target)
        diagonal = max(hypot(w, h), 1.0)
        hand_near_target = False
        for hand_item in hands:
            px, py = self._palm_center(hand_item)
            dx = max(x - px, 0, px - (x + w))
            dy = max(y - py, 0, py - (y + h))
            if hypot(dx, dy) / diagonal <= 0.5:
                hand_near_target = True
                break
        if hand_near_target:
            return
        self._rest_y_samples.append(cy)
        if len(self._rest_y_samples) >= 5:
            self._resting_center_y = float(np.median(self._rest_y_samples))

    def _evaluate(self, context: FrameContext) -> Tuple[Optional[bool], float, Dict[str, Any], List[Any], str]:
        targets = [item for item in context.detections if _item_name(item) == self.target_class]
        hands = [hand for hand in context.hand_results if float(hand.get("confidence", 0)) >= 0.6]
        if not targets:
            return None, 0.0, {"hand_count": len(hands)}, [], f"未检测到{self.target_class}"
        target = max(targets, key=lambda item: float(getattr(item, "confidence", 0)))
        self._update_resting_baseline(target, hands)
        if len(hands) < 2:
            return None, 0.0, {
                "hand_count": len(hands),
                "resting_center_y": self._resting_center_y,
            }, [target], "需要同时清晰检测到两只手"
        single_camera = context.camera_role is CameraRole.SINGLE
        if not context.pose_results and not single_camera:
            return None, 0.0, {"hand_count": len(hands)}, [target], "未检测到正侧面人体姿态"

        x, y, w, h = _bbox(target)
        cx, cy = _centroid(target)
        diagonal = max(hypot(w, h), 1.0)
        palms = sorted([self._palm_center(hand) for hand in hands[:2]], key=lambda point: point[0])
        distances = []
        for px, py in palms:
            dx = max(x - px, 0, px - (x + w))
            dy = max(y - py, 0, py - (y + h))
            distances.append(hypot(dx, dy) / diagonal)
        palm_array = np.asarray(palms, dtype=np.float64)
        target_center = np.asarray((cx, cy), dtype=np.float64)
        vectors = palm_array - target_center
        vector_norms = np.linalg.norm(vectors, axis=1)
        opposition_cosine = float(
            np.dot(vectors[0], vectors[1])
            / max(vector_norms[0] * vector_norms[1], 1e-6)
        )
        palm_separation = float(np.linalg.norm(palm_array[1] - palm_array[0])) / diagonal
        segment = palm_array[1] - palm_array[0]
        segment_length_sq = float(np.dot(segment, segment))
        segment_t = float(
            np.clip(np.dot(target_center - palm_array[0], segment) / max(segment_length_sq, 1e-6), 0, 1)
        )
        nearest_on_segment = palm_array[0] + segment_t * segment
        target_segment_distance = float(
            np.linalg.norm(target_center - nearest_on_segment) / diagonal
        )
        opposite_sides = (
            opposition_cosine <= -0.15
            and palm_separation >= 0.35
            and target_segment_distance <= 0.4
        )
        straddles_target = palms[0][0] <= cx - 0.1 * w and palms[1][0] >= cx + 0.1 * w
        hand_contact_distances = []
        for hand_item in hands[:2]:
            hx, hy, hw, hh = hand_item.get("bbox", (0, 0, 0, 0))
            gap_x = max(x - (hx + hw), hx - (x + w), 0)
            gap_y = max(y - (hy + hh), hy - (y + h), 0)
            hand_contact_distances.append(hypot(gap_x, gap_y) / diagonal)
        hands_contact_target = max(hand_contact_distances) <= 0.08

        lift_height_ratio = None
        baseline_lift_pixels = None
        required_target_lift_pixels = None
        lift_valid = not single_camera
        if single_camera and self._resting_center_y is not None:
            baseline_lift_pixels = self._resting_center_y - cy
            lift_height_ratio = baseline_lift_pixels / max(h, 1.0)
            required_target_lift_pixels = max(
                self.min_lift_height_ratio * h,
                context.frame_shape[0] * 0.035,
            )
            lift_valid = baseline_lift_pixels >= required_target_lift_pixels

        chest_valid = single_camera
        chest_verified = False
        shoulder_width = 0.0
        landmarks = np.asarray(
            context.pose_results[0].get("landmarks", [])
            if context.pose_results else []
        )
        if not single_camera and len(landmarks) >= 15:
            shoulder_y = float(np.mean(landmarks[[11, 12], 1]))
            elbow_y = float(np.mean(landmarks[[13, 14], 1]))
            shoulder_width = float(np.linalg.norm(landmarks[11, :2] - landmarks[12, :2]))
            margin = max(shoulder_width * 0.2, 1.0)
            low, high = sorted((shoulder_y - margin, elbow_y + margin))
            shoulder_mid_x = float(np.mean(landmarks[[11, 12], 0]))
            chest_valid = low <= cy <= high and abs(cx - shoulder_mid_x) <= max(shoulder_width, w)
            chest_verified = True

        passed = (
            opposite_sides
            and straddles_target
            and hands_contact_target
            and chest_valid
            and lift_valid
        )
        confidence = float(np.mean([float(hand.get("confidence", 0)) for hand in hands[:2]]))
        confidence *= float(getattr(target, "confidence", 0))
        if single_camera:
            confidence *= 0.9
        confidence *= 1.0 if passed else 0.5
        metrics = {
            "hand_count": len(hands),
            "opposite_sides": opposite_sides,
            "opposition_cosine": round(opposition_cosine, 4),
            "palm_separation_target_diagonal": round(palm_separation, 4),
            "target_segment_distance": round(target_segment_distance, 4),
            "max_hand_distance_target": round(max(distances), 4),
            "straddles_target": straddles_target,
            "hands_contact_target": hands_contact_target,
            "max_hand_bbox_gap_target": round(max(hand_contact_distances), 4),
            "chest_region": chest_valid,
            "chest_region_verified": chest_verified,
            "geometry_mode": "single_camera_relational" if single_camera else "side_pose",
            "shoulder_width": round(shoulder_width, 3),
            "resting_center_y": (
                round(self._resting_center_y, 3)
                if self._resting_center_y is not None else None
            ),
            "lift_height_ratio": (
                round(lift_height_ratio, 4) if lift_height_ratio is not None else None
            ),
            "baseline_lift_pixels": (
                round(baseline_lift_pixels, 2) if baseline_lift_pixels is not None else None
            ),
            "required_target_lift_pixels": (
                round(required_target_lift_pixels, 2)
                if required_target_lift_pixels is not None else None
            ),
            "lift_verified": lift_valid,
        }
        if passed:
            reason = (
                "单摄像头已确认双手与茶荷托举关系（未验证齐胸高度）"
                if single_camera else "已确认双手托举且茶荷位于齐胸区域"
            )
        elif not opposite_sides:
            reason = "两只手需要分处茶荷两侧并托住茶荷"
        elif not straddles_target or not hands_contact_target:
            reason = "两只手需要分别与茶荷两侧接触"
        elif single_camera and self._resting_center_y is None:
            reason = "请先让茶荷在桌面静置约0.5秒以建立位置基线"
        elif single_camera and not lift_valid:
            reason = "双手已接触茶荷，但茶荷仍处于桌面位置"
        else:
            reason = "茶荷未处于肩部到肘部之间的齐胸区域"
        return passed, confidence, metrics, [target], reason

    def update(self, context: FrameContext):
        passed, confidence, metrics, targets, reason = self._evaluate(context)
        evidence = _evidence(context, targets, metrics)
        if self._completed:
            if passed is True:
                self._release_started = None
                return ObservationSnapshot(
                    self.observation_id, self.name, self.sop_step, ObservationState.COMPLETED,
                    confidence=confidence, value=True, reason=reason,
                    started_at=self._started_at, updated_at=context.timestamp,
                    metrics=metrics, experimental=True,
                ), []
            if self._release_started is None:
                self._release_started = context.timestamp
            release_elapsed = context.timestamp - self._release_started
            if release_elapsed < self.release_seconds:
                return ObservationSnapshot(
                    self.observation_id, self.name, self.sop_step, ObservationState.COMPLETED,
                    confidence=confidence, value=True,
                    reason="托举关系短暂丢失，等待确认是否已放下",
                    started_at=self._started_at, updated_at=context.timestamp,
                    metrics={**metrics, "release_elapsed": round(release_elapsed, 4)},
                    experimental=True,
                ), []
            self._completed = False
            self._release_started = None
            self._history.clear()
            self._started_at = None
            self._started_event_emitted = False

        if passed is None:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                confidence=confidence, reason=reason, updated_at=context.timestamp,
                metrics=metrics, experimental=True,
            ), []
        self._history.append((context.timestamp, passed, confidence, evidence))
        cutoff = context.timestamp - self.stable_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if not passed:
            if not any(row[1] for row in self._history):
                self._started_at = None
                self._started_event_emitted = False
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.IDLE,
                confidence=confidence, reason=reason, updated_at=context.timestamp,
                metrics=metrics, experimental=True,
            ), []

        if self._started_at is None:
            self._started_at = context.timestamp
        started_events: List[ObservationEvent] = []
        if not self._started_event_emitted:
            self._started_event_emitted = True
            started_events.append(_event(
                self, context, EventPhase.STARTED, self._started_at,
                confidence, True, metrics, [evidence],
            ))
        positives = [row for row in self._history if row[1]]
        span = self._history[-1][0] - self._history[0][0] if len(self._history) > 1 else 0.0
        ratio = len(positives) / len(self._history)
        if len(self._history) >= 4 and span >= self.stable_seconds * 0.95 and ratio >= self.stable_ratio:
            event_confidence = float(np.mean([row[2] for row in positives]))
            self._completed = True
            self._release_started = None
            event = _event(
                self, context, EventPhase.COMPLETED, self._started_at, event_confidence,
                True, {**metrics, "stable_ratio": round(ratio, 4)}, [row[3] for row in positives],
            )
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.COMPLETED,
                confidence=event_confidence, value=True,
                reason=(
                    "检测到双手托举茶荷赏茶持续至少5秒"
                    if self.target_class == "茶荷" and self.stable_seconds >= 5.0
                    else "检测到稳定双手托举"
                ),
                started_at=self._started_at, updated_at=context.timestamp,
                metrics=event.metrics, experimental=True,
            ), started_events + [event]
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step, ObservationState.CANDIDATE,
            confidence=confidence, value=True, reason="等待托举动作稳定",
            started_at=self._started_at, updated_at=context.timestamp,
            metrics={**metrics, "stable_ratio": round(ratio, 4)}, experimental=True,
        ), started_events

    def reset(self) -> None:
        self._history.clear()
        self._rest_y_samples.clear()
        self._resting_center_y = None
        self._started_at = None
        self._completed = False
        self._release_started = None
        self._started_event_emitted = False


class LidOpenSmellObservation:
    observation_id = "action_open_lid_smell"
    name = "打开盖碗闻香"
    sop_step = 4
    camera_roles: Set[CameraRole] = {CameraRole.SIDE}
    rule_version = "1.0-experimental"
    BODY_NAMES = {"盖碗碗身", "盖碗（碗身）"}
    LID_NAMES = {"盖碗碗盖", "盖碗（碗盖）"}

    def __init__(self, smell_seconds: float = 0.5, sequence_timeout: float = 8.0):
        self.smell_seconds = smell_seconds
        self.sequence_timeout = sequence_timeout
        self._stage = "waiting_closed"
        self._started_at: Optional[float] = None
        self._near_started: Optional[float] = None
        self._evidence: List[EvidenceFrame] = []
        self._completed = False
        self._started_event_emitted = False

    @staticmethod
    def _iou(a: Any, b: Any) -> float:
        ax, ay, aw, ah = _bbox(a)
        bx, by, bw, bh = _bbox(b)
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return inter / max(aw * ah + bw * bh - inter, 1.0)

    def update(self, context: FrameContext):
        has_parts = bool(self.BODY_NAMES & context.model_classes) and bool(self.LID_NAMES & context.model_classes)
        if "gaiwan_parts" not in context.capabilities and not has_parts:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                reason="当前模型不能区分盖碗碗身和碗盖，闻香观测未启用",
                updated_at=context.timestamp, experimental=True,
                metrics={"required_capability": "gaiwan_parts"},
            ), []

        bodies = [item for item in context.detections if _item_name(item) in self.BODY_NAMES]
        lids = [item for item in context.detections if _item_name(item) in self.LID_NAMES]
        if not bodies or not lids or not context.pose_results:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                reason="需要同时看清碗身、碗盖和鼻部关键点", updated_at=context.timestamp,
                experimental=True, metrics={"body_count": len(bodies), "lid_count": len(lids)},
            ), []

        body = max(bodies, key=lambda item: float(getattr(item, "confidence", 0)))
        lid = max(lids, key=lambda item: float(getattr(item, "confidence", 0)))
        body_center, lid_center = _centroid(body), _centroid(lid)
        body_width = max(_bbox(body)[2], 1.0)
        center_distance = hypot(body_center[0] - lid_center[0], body_center[1] - lid_center[1]) / body_width
        iou = self._iou(body, lid)
        is_closed = iou >= 0.25 and center_distance <= 0.35
        is_open = iou < 0.25 or center_distance > 0.35

        pose = np.asarray(context.pose_results[0].get("landmarks", []))
        near_nose = False
        nose_distance = None
        shoulder_width = None
        if len(pose) >= 13:
            shoulder_width = float(np.linalg.norm(pose[11, :2] - pose[12, :2]))
            nose_distance = hypot(body_center[0] - pose[0, 0], body_center[1] - pose[0, 1])
            near_nose = shoulder_width > 1 and nose_distance < 0.6 * shoulder_width
        metrics = {
            "stage": self._stage,
            "lid_body_iou": round(iou, 4),
            "lid_body_distance_width": round(center_distance, 4),
            "near_nose": near_nose,
            "nose_distance_shoulder": round(nose_distance / shoulder_width, 4)
            if nose_distance is not None and shoulder_width else None,
        }
        evidence = _evidence(context, [body, lid], metrics)

        if self._completed:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.COMPLETED,
                confidence=1.0, value=True, reason="闻香事件已完成", started_at=self._started_at,
                updated_at=context.timestamp, metrics=metrics, experimental=True,
            ), []
        if self._started_at is not None and context.timestamp - self._started_at > self.sequence_timeout:
            self._stage = "waiting_closed"
            self._started_at = None
            self._near_started = None
            self._evidence.clear()
            self._started_event_emitted = False
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                reason="开盖后未在时限内完成靠近鼻部并重新盖合", updated_at=context.timestamp,
                metrics=metrics, experimental=True,
            ), []

        if self._stage == "waiting_closed":
            if is_closed:
                self._stage = "closed_seen"
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.IDLE,
                confidence=0.5 if is_closed else 0.0, reason="等待开盖闻香",
                updated_at=context.timestamp, metrics=metrics, experimental=True,
            ), []
        emitted: List[ObservationEvent] = []
        if self._stage == "closed_seen" and is_open:
            self._stage = "open"
            self._started_at = context.timestamp
            self._evidence = [evidence]
            if not self._started_event_emitted:
                self._started_event_emitted = True
                emitted.append(_event(
                    self, context, EventPhase.STARTED, self._started_at,
                    float(min(getattr(body, "confidence", 0), getattr(lid, "confidence", 0))),
                    "open", metrics, [evidence],
                ))
        elif self._stage == "open":
            self._evidence.append(evidence)
            if is_open and near_nose:
                if self._near_started is None:
                    self._near_started = context.timestamp
                if context.timestamp - self._near_started >= self.smell_seconds:
                    self._stage = "smelled"
            elif not near_nose:
                self._near_started = None
        elif self._stage == "smelled":
            self._evidence.append(evidence)
            if is_closed:
                self._completed = True
                confidence = float(min(getattr(body, "confidence", 0), getattr(lid, "confidence", 0)))
                event = _event(
                    self, context, EventPhase.COMPLETED,
                    context.timestamp if self._started_at is None else self._started_at,
                    confidence, True, metrics, self._evidence,
                )
                return ObservationSnapshot(
                    self.observation_id, self.name, self.sop_step, ObservationState.COMPLETED,
                    confidence=confidence, value=True, reason="完成开盖、靠近鼻部和重新盖合",
                    started_at=self._started_at, updated_at=context.timestamp,
                    metrics=metrics, experimental=True,
                ), emitted + [event]

        state = ObservationState.ACTIVE if self._stage in {"open", "smelled"} else ObservationState.CANDIDATE
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step, state,
            confidence=float(min(getattr(body, "confidence", 0), getattr(lid, "confidence", 0))),
            value=self._stage, reason="闻香动作序列进行中", started_at=self._started_at,
            updated_at=context.timestamp, metrics={**metrics, "stage": self._stage}, experimental=True,
        ), emitted

    def reset(self) -> None:
        self._stage = "waiting_closed"
        self._started_at = None
        self._near_started = None
        self._evidence.clear()
        self._completed = False
        self._started_event_emitted = False


class GaiwanLidClosureObservation:
    """Detect an open/separated lid becoming stably closed on the gaiwan."""

    observation_id = "action_gaiwan_lid_close_brew"
    name = "冲泡后盖上碗盖"
    sop_step = 5
    camera_roles: Set[CameraRole] = {CameraRole.SIDE}
    rule_version = "1.0-experimental"
    BODY_NAMES = LidOpenSmellObservation.BODY_NAMES
    LID_NAMES = LidOpenSmellObservation.LID_NAMES

    def __init__(
        self,
        open_seconds: float = 0.4,
        close_seconds: float = 0.5,
        min_samples: int = 3,
        stable_ratio: float = 0.7,
    ):
        self.open_seconds = open_seconds
        self.close_seconds = close_seconds
        self.min_samples = min_samples
        self.stable_ratio = stable_ratio
        self._stage = "waiting_open"
        self._history: Deque[Tuple[float, bool, float, EvidenceFrame]] = deque()
        self._started_at: Optional[float] = None
        self._evidence: List[EvidenceFrame] = []
        self._completed = False
        self._completion_confidence = 0.0
        self._completion_metrics: Dict[str, Any] = {}

    @classmethod
    def required_classes_available(cls, classes: Set[str]) -> bool:
        return bool(cls.BODY_NAMES & classes) and bool(cls.LID_NAMES & classes)

    def _append(
        self,
        context: FrameContext,
        passed: bool,
        confidence: float,
        evidence: EvidenceFrame,
        seconds: float,
    ) -> None:
        self._history.append((context.timestamp, passed, confidence, evidence))
        cutoff = context.timestamp - seconds - 1e-6
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _stable(self, seconds: float) -> bool:
        if len(self._history) < self.min_samples:
            return False
        span = self._history[-1][0] - self._history[0][0]
        ratio = sum(1 for row in self._history if row[1]) / len(self._history)
        return span >= seconds * 0.95 and ratio >= self.stable_ratio

    def update(self, context: FrameContext):
        if not self.required_classes_available(context.model_classes):
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                reason="当前模型需要区分盖碗碗身和碗盖", updated_at=context.timestamp,
                metrics={"required_capability": "gaiwan_parts"}, experimental=True,
            ), []
        if self._completed:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.COMPLETED,
                confidence=self._completion_confidence, value=True,
                reason="已检测到碗盖从打开状态稳定盖合", started_at=self._started_at,
                updated_at=context.timestamp, metrics=self._completion_metrics,
                experimental=True,
            ), []

        bodies = [item for item in context.detections if _item_name(item) in self.BODY_NAMES]
        lids = [item for item in context.detections if _item_name(item) in self.LID_NAMES]
        if not bodies or not lids:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                reason="需要同时看清碗身和碗盖", updated_at=context.timestamp,
                metrics={"body_count": len(bodies), "lid_count": len(lids), "stage": self._stage},
                experimental=True,
            ), []

        body = max(bodies, key=lambda item: float(getattr(item, "confidence", 0.0)))
        lid = max(lids, key=lambda item: float(getattr(item, "confidence", 0.0)))
        body_center, lid_center = _centroid(body), _centroid(lid)
        body_width = max(_bbox(body)[2], 1.0)
        distance_ratio = hypot(
            body_center[0] - lid_center[0], body_center[1] - lid_center[1]
        ) / body_width
        iou = LidOpenSmellObservation._iou(body, lid)
        is_closed = iou >= 0.25 and distance_ratio <= 0.35
        is_open = iou < 0.25 or distance_ratio > 0.35
        confidence = min(
            float(getattr(body, "confidence", 0.0)),
            float(getattr(lid, "confidence", 0.0)),
        )
        metrics = {
            "stage": self._stage,
            "lid_body_iou": round(iou, 4),
            "lid_body_distance_width": round(distance_ratio, 4),
            "is_open": is_open,
            "is_closed": is_closed,
        }
        evidence = _evidence(context, [body, lid], metrics)

        if self._stage == "waiting_open":
            self._append(context, is_open, confidence, evidence, self.open_seconds)
            if self._stable(self.open_seconds):
                positives = [row for row in self._history if row[1]]
                self._stage = "waiting_close"
                self._started_at = positives[0][0]
                self._evidence.extend(row[3] for row in positives)
                self._history.clear()
                event = _event(
                    self, context, EventPhase.STARTED, self._started_at,
                    float(np.mean([row[2] for row in positives])), "lid_open_seen",
                    {**metrics, "stage": self._stage}, [row[3] for row in positives],
                )
                return ObservationSnapshot(
                    self.observation_id, self.name, self.sop_step, ObservationState.ACTIVE,
                    confidence=confidence, value="已打开，等待盖合",
                    reason="已确认碗盖分离，等待盖上碗盖", started_at=self._started_at,
                    updated_at=context.timestamp, metrics={**metrics, "stage": self._stage},
                    experimental=True,
                ), [event]
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.CANDIDATE if is_open else ObservationState.IDLE,
                confidence=confidence, reason="需先观察到碗盖打开，再判断盖合",
                updated_at=context.timestamp, metrics=metrics, experimental=True,
            ), []

        self._append(context, is_closed, confidence, evidence, self.close_seconds)
        if self._stable(self.close_seconds):
            positives = [row for row in self._history if row[1]]
            self._evidence.extend(row[3] for row in positives)
            self._completed = True
            self._completion_confidence = float(np.mean([row[2] for row in positives]))
            self._completion_metrics = {**metrics, "stage": "completed"}
            event = _event(
                self, context, EventPhase.COMPLETED, self._started_at or context.timestamp,
                self._completion_confidence, True, self._completion_metrics, self._evidence,
            )
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.COMPLETED,
                confidence=self._completion_confidence, value=True,
                reason="已检测到碗盖从打开状态稳定盖合", started_at=self._started_at,
                updated_at=context.timestamp, metrics=self._completion_metrics,
                experimental=True,
            ), [event]
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step,
            ObservationState.CANDIDATE if is_closed else ObservationState.ACTIVE,
            confidence=confidence, value="正在盖合" if is_closed else "等待盖合",
            reason="等待碗盖盖合关系稳定", started_at=self._started_at,
            updated_at=context.timestamp, metrics=metrics, experimental=True,
        ), []

    def reset(self) -> None:
        self._stage = "waiting_open"
        self._history.clear()
        self._started_at = None
        self._evidence.clear()
        self._completed = False
        self._completion_confidence = 0.0
        self._completion_metrics.clear()


class GaiwanToPitcherObservation:
    """Experimental geometry for decanting the gaiwan into the fairness pitcher."""

    observation_id = "action_gaiwan_to_pitcher"
    name = "盖碗向公道杯出汤"
    sop_step = 5
    camera_roles: Set[CameraRole] = {CameraRole.SIDE}
    rule_version = "1.0-experimental"
    BODY_NAMES = LidOpenSmellObservation.BODY_NAMES

    def __init__(self, stable_seconds: float = 0.5, min_samples: int = 3):
        self.stable_seconds = stable_seconds
        self.min_samples = min_samples
        self._history: Deque[Tuple[float, bool, float, EvidenceFrame]] = deque()
        self._motion_history: Deque[Tuple[float, float, float]] = deque()
        self._rest_y_samples: Deque[float] = deque(maxlen=20)
        self._resting_y: Optional[float] = None
        self._started_at: Optional[float] = None
        self._started_event_emitted = False
        self._completed = False
        self._completion_confidence = 0.0
        self._completion_metrics: Dict[str, Any] = {}

    @classmethod
    def required_classes_available(cls, classes: Set[str]) -> bool:
        return bool(cls.BODY_NAMES & classes) and "公道杯" in classes

    @staticmethod
    def _bbox_gap(a: Any, b: Any) -> float:
        ax, ay, aw, ah = _bbox(a)
        bx, by, bw, bh = _bbox(b)
        dx = max(bx - (ax + aw), ax - (bx + bw), 0.0)
        dy = max(by - (ay + ah), ay - (by + bh), 0.0)
        return hypot(dx, dy)

    @staticmethod
    def _hand_gap(hand: Dict[str, Any], item: Any) -> float:
        hx, hy, hw, hh = [float(value) for value in hand.get("bbox", (0, 0, 0, 0))]
        x, y, w, h = _bbox(item)
        dx = max(x - (hx + hw), hx - (x + w), 0.0)
        dy = max(y - (hy + hh), hy - (y + h), 0.0)
        return hypot(dx, dy) / max(hypot(w, h), 1.0)

    def _direct_signal(self, context: FrameContext) -> Optional[Tuple[bool, float, str, bool]]:
        matches = []
        rows = list(context.extras.get("brew_decant_interactions", []))
        rows.extend(context.extras.get("pour_interactions", []))
        for row in rows:
            if str(row.get("source", "")) not in self.BODY_NAMES or str(row.get("target", "")) != "公道杯":
                continue
            confidence = float(row.get("confidence", 0.0))
            matches.append((
                bool(row.get("active", True)) and confidence >= 0.5,
                confidence,
                str(row.get("signal_source", "external")),
                bool(row.get("liquid_verified", False)),
            ))
        return max(matches, key=lambda row: (row[0], row[1]), default=None)

    def update(self, context: FrameContext):
        if not self.required_classes_available(context.model_classes):
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                reason="当前模型需要支持盖碗碗身和公道杯", updated_at=context.timestamp,
                experimental=True,
            ), []
        if self._completed:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.COMPLETED,
                confidence=self._completion_confidence, value=True,
                reason="已检测到盖碗向公道杯出汤关系", started_at=self._started_at,
                updated_at=context.timestamp, metrics=self._completion_metrics,
                experimental=True,
            ), []

        bodies = [item for item in context.detections if _item_name(item) in self.BODY_NAMES]
        pitchers = [item for item in context.detections if _item_name(item) == "公道杯"]
        if not bodies or not pitchers:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                reason="需要同时看清盖碗碗身和公道杯", updated_at=context.timestamp,
                metrics={"body_count": len(bodies), "pitcher_count": len(pitchers)},
                experimental=True,
            ), []
        body = max(bodies, key=lambda item: float(getattr(item, "confidence", 0.0)))
        pitcher = max(pitchers, key=lambda item: float(getattr(item, "confidence", 0.0)))
        bx, by, bw, bh = _bbox(body)
        px, py, pw, ph = _bbox(pitcher)
        bcx, bcy = _centroid(body)
        pcx, pcy = _centroid(pitcher)
        scale = max(hypot(bw, bh), hypot(pw, ph), 1.0)
        hands = [hand for hand in context.hand_results if float(hand.get("confidence", 0.0)) >= 0.55]
        hand_gap = min((self._hand_gap(hand, body) for hand in hands), default=float("inf"))
        hand_contact = hand_gap <= 0.12

        if not hand_contact:
            self._rest_y_samples.append(bcy)
            if len(self._rest_y_samples) >= 5:
                self._resting_y = float(np.median(self._rest_y_samples))
        self._motion_history.append((context.timestamp, bcx, bcy))
        cutoff = context.timestamp - 1.5
        while self._motion_history and self._motion_history[0][0] < cutoff:
            self._motion_history.popleft()
        motion_ratio = max(
            (hypot(bcx - old_x, bcy - old_y) / scale for _, old_x, old_y in self._motion_history),
            default=0.0,
        )
        lift_pixels = self._resting_y - bcy if self._resting_y is not None else None
        lift_required = max(0.35 * bh, context.frame_shape[0] * 0.02)
        lift_valid = lift_pixels is not None and lift_pixels >= lift_required
        proximity = self._bbox_gap(body, pitcher) / scale <= 0.55
        center_ratio = hypot(bcx - pcx, bcy - pcy) / scale
        source_above = bcy <= pcy + 0.35 * ph
        geometry_passed = (
            hand_contact and lift_valid and proximity and center_ratio <= 1.8
            and source_above and motion_ratio >= 0.05
        )

        direct = self._direct_signal(context)
        passed = geometry_passed
        signal_source = "geometry"
        liquid_verified = False
        direct_confidence = None
        if direct is not None:
            passed, direct_confidence, signal_source, liquid_verified = direct
        confidence = (
            direct_confidence if direct_confidence is not None else
            min(float(getattr(body, "confidence", 0.0)), float(getattr(pitcher, "confidence", 0.0)))
            * (0.85 if passed else 0.45)
        )
        metrics = {
            "signal_source": signal_source,
            "liquid_verified": liquid_verified,
            "hand_contact": hand_contact,
            "hand_gap": None if hand_gap == float("inf") else round(hand_gap, 4),
            "resting_y": round(self._resting_y, 3) if self._resting_y is not None else None,
            "lift_pixels": round(lift_pixels, 3) if lift_pixels is not None else None,
            "lift_required": round(lift_required, 3),
            "lift_verified": lift_valid,
            "source_target_gap_ratio": round(self._bbox_gap(body, pitcher) / scale, 4),
            "source_target_center_ratio": round(center_ratio, 4),
            "source_above_target": source_above,
            "source_motion_ratio": round(motion_ratio, 4),
        }
        evidence = _evidence(context, [body, pitcher], metrics)
        self._history.append((context.timestamp, passed, float(confidence), evidence))
        history_cutoff = context.timestamp - self.stable_seconds
        while self._history and self._history[0][0] < history_cutoff:
            self._history.popleft()

        emitted: List[ObservationEvent] = []
        if passed and self._started_at is None:
            self._started_at = context.timestamp
        if passed and not self._started_event_emitted:
            self._started_event_emitted = True
            emitted.append(_event(
                self, context, EventPhase.STARTED, self._started_at or context.timestamp,
                float(confidence), True, metrics, [evidence],
            ))
        positives = [row for row in self._history if row[1]]
        span = self._history[-1][0] - self._history[0][0] if len(self._history) > 1 else 0.0
        stable = (
            len(self._history) >= self.min_samples
            and span >= self.stable_seconds * 0.95
            and len(positives) / len(self._history) >= 0.7
        )
        if stable:
            self._completed = True
            self._completion_confidence = float(np.mean([row[2] for row in positives]))
            self._completion_metrics = {**metrics, "stage": "completed"}
            event = _event(
                self, context, EventPhase.COMPLETED, self._started_at or context.timestamp,
                self._completion_confidence, True, self._completion_metrics,
                [row[3] for row in positives],
            )
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.COMPLETED,
                confidence=self._completion_confidence, value=True,
                reason=("已由液流证据确认出汤" if liquid_verified else "已检测到盖碗向公道杯出汤几何关系"),
                started_at=self._started_at, updated_at=context.timestamp,
                metrics=self._completion_metrics, experimental=True,
            ), emitted + [event]
        if not passed and not positives:
            self._started_at = None
            self._started_event_emitted = False
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step,
            ObservationState.CANDIDATE if passed else ObservationState.IDLE,
            confidence=float(confidence), value="正在出汤" if passed else None,
            reason=("等待出汤关系稳定" if passed else "等待盖碗离桌并靠近公道杯出汤"),
            updated_at=context.timestamp, metrics=metrics, experimental=True,
        ), emitted

    def reset(self) -> None:
        self._history.clear()
        self._motion_history.clear()
        self._rest_y_samples.clear()
        self._resting_y = None
        self._started_at = None
        self._started_event_emitted = False
        self._completed = False
        self._completion_confidence = 0.0
        self._completion_metrics.clear()


class BrewWaitTimerObservation:
    """Compose lid-close and decant events into the partial brew timing result."""

    observation_id = "result_brew_wait_decant_partial"
    name = "冲泡等待与出汤（不含注水）"
    sop_step = 5
    camera_roles: Set[CameraRole] = {CameraRole.SIDE}
    rule_version = "1.0-partial"

    def __init__(
        self,
        minimum_wait_seconds: float = 10.0,
        require_injection: bool = False,
        maximum_wait_seconds: Optional[float] = None,
    ):
        self.minimum_wait_seconds = minimum_wait_seconds
        self.maximum_wait_seconds = maximum_wait_seconds
        self.require_injection = require_injection
        self._lid_closed_at: Optional[float] = None
        self._injection_verified = False
        self._lid_confidence = 0.0
        self._lid_evidence: List[EvidenceFrame] = []
        self._final_state: Optional[ObservationState] = None
        self._final_confidence = 0.0
        self._final_reason = ""
        self._final_metrics: Dict[str, Any] = {}

    @staticmethod
    def _phase(event: Any) -> str:
        phase = getattr(event, "phase", None)
        return str(getattr(phase, "value", phase))

    def update(self, context: FrameContext):
        if self._final_state is not None:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, self._final_state,
                confidence=self._final_confidence,
                value=self._final_metrics.get("wait_seconds"), reason=self._final_reason,
                started_at=self._lid_closed_at, updated_at=context.timestamp,
                metrics=self._final_metrics, experimental=True,
            ), []

        events = list(context.extras.get("frame_observation_events", []))
        injection_event = next((
            event for event in events
            if getattr(event, "observation_id", "") == "action_water_injection"
            and self._phase(event) == EventPhase.COMPLETED.value
        ), None)
        if injection_event is not None:
            self._injection_verified = True
        lid_event = next((
            event for event in events
            if getattr(event, "observation_id", "") == GaiwanLidClosureObservation.observation_id
            and self._phase(event) == EventPhase.COMPLETED.value
        ), None)
        if lid_event is not None and (not self.require_injection or self._injection_verified):
            self._lid_closed_at = float(getattr(lid_event, "end_time", context.timestamp))
            self._lid_confidence = float(getattr(lid_event, "confidence", 0.0))
            self._lid_evidence = list(getattr(lid_event, "evidence", []))

        decant_event = next((
            event for event in events
            if getattr(event, "observation_id", "") == GaiwanToPitcherObservation.observation_id
            and self._phase(event) in {
                EventPhase.STARTED.value,
                EventPhase.COMPLETED.value,
            }
        ), None)
        base_metrics = {
            "minimum_wait_seconds": self.minimum_wait_seconds,
            "maximum_wait_seconds": self.maximum_wait_seconds,
            "partial_mode": not self.require_injection,
            "injection_verified": self._injection_verified,
        }
        if self.require_injection and not self._injection_verified:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                reason="正式冲泡计时需要先确认注水完成", updated_at=context.timestamp,
                metrics=base_metrics, experimental=True,
            ), []
        if decant_event is not None and self._lid_closed_at is None:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                confidence=float(getattr(decant_event, "confidence", 0.0)),
                reason="检测到出汤，但此前没有可靠的碗盖盖合计时起点",
                updated_at=context.timestamp,
                metrics={**base_metrics, "missing_lid_close": True}, experimental=True,
            ), []
        if self._lid_closed_at is None:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.IDLE,
                reason="等待冲泡后的碗盖盖合事件", updated_at=context.timestamp,
                metrics=base_metrics, experimental=True,
            ), []

        elapsed = max(0.0, context.timestamp - self._lid_closed_at)
        if decant_event is None:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.ACTIVE,
                confidence=self._lid_confidence, value=round(elapsed, 2),
                reason=(
                    f"冲泡计时中：{elapsed:.1f}秒，合格范围{self.minimum_wait_seconds:g}至{self.maximum_wait_seconds:g}秒"
                    if self.maximum_wait_seconds is not None
                    else f"冲泡计时中：{elapsed:.1f}秒，至少等待{self.minimum_wait_seconds:g}秒"
                ),
                started_at=self._lid_closed_at, updated_at=context.timestamp,
                metrics={**base_metrics, "wait_seconds": round(elapsed, 4)}, experimental=True,
            ), []

        decant_started_at = float(getattr(decant_event, "start_time", context.timestamp))
        wait_seconds = max(0.0, decant_started_at - self._lid_closed_at)
        decant_confidence = float(getattr(decant_event, "confidence", 0.0))
        confidence = float(np.mean([self._lid_confidence, decant_confidence]))
        decant_metrics = dict(getattr(decant_event, "metrics", {}) or {})
        metrics = {
            **base_metrics,
            "wait_seconds": round(wait_seconds, 4),
            "liquid_verified": bool(decant_metrics.get("liquid_verified", False)),
        }
        evidence = self._lid_evidence + list(getattr(decant_event, "evidence", []))
        within_maximum = (
            self.maximum_wait_seconds is None
            or wait_seconds <= self.maximum_wait_seconds
        )
        if wait_seconds >= self.minimum_wait_seconds and within_maximum:
            phase = EventPhase.COMPLETED
            self._final_state = ObservationState.COMPLETED
            reason = (
                f"盖合后等待{wait_seconds:.1f}秒再出汤，处于合格区间"
                if self.maximum_wait_seconds is not None
                else f"盖合后等待{wait_seconds:.1f}秒再出汤，达到最低要求"
            )
        else:
            high_confidence = self._lid_confidence >= 0.7 and decant_confidence >= 0.7
            phase = EventPhase.FAILED if high_confidence else EventPhase.UNCERTAIN
            self._final_state = ObservationState.FAILED if high_confidence else ObservationState.UNCERTAIN
            reason = (
                f"盖合后等待{wait_seconds:.1f}秒才出汤，超过{self.maximum_wait_seconds:g}秒"
                if self.maximum_wait_seconds is not None and wait_seconds > self.maximum_wait_seconds
                else f"盖合后仅等待{wait_seconds:.1f}秒即出汤，未达到{self.minimum_wait_seconds:g}秒"
            )
        self._final_confidence = confidence
        self._final_reason = reason
        self._final_metrics = metrics
        event = _event(
            self, context, phase, self._lid_closed_at, confidence,
            round(wait_seconds, 2), metrics, evidence,
        )
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step, self._final_state,
            confidence=confidence, value=round(wait_seconds, 2), reason=reason,
            started_at=self._lid_closed_at, updated_at=context.timestamp,
            metrics=metrics, experimental=True,
        ), [event]

    def reset(self) -> None:
        self._lid_closed_at = None
        self._injection_verified = False
        self._lid_confidence = 0.0
        self._lid_evidence.clear()
        self._final_state = None
        self._final_confidence = 0.0
        self._final_reason = ""
        self._final_metrics.clear()


class HandAccessoryObservation:
    observation_id = "result_hand_accessory"
    name = "手部及手腕饰品"
    sop_step = 0
    camera_roles: Set[CameraRole] = {CameraRole.SIDE}
    rule_version = "1.0"

    def __init__(self, detection_threshold: float = 0.5):
        self.detection_threshold = detection_threshold
        self._windows: Deque[Tuple[float, bool, List[Dict[str, Any]]]] = deque(maxlen=5)
        self._clear_started: Optional[float] = None
        self._completed = False

    def update(self, context: FrameContext):
        configured = bool(context.extras.get("accessory_detector_configured", False))
        if not configured:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.UNCERTAIN,
                reason="未配置手部饰品模型，不能根据颜色代替识别",
                updated_at=context.timestamp, metrics={"required_capability": "hand_accessory_detector"},
            ), []
        detections = [
            item for item in context.extras.get("accessory_detections", [])
            if float(item.get("confidence", 0)) >= self.detection_threshold
        ]
        both_hands_visible = len([h for h in context.hand_results if float(h.get("confidence", 0)) >= 0.6]) >= 2
        self._windows.append((context.timestamp, bool(detections), detections))
        positive_count = sum(1 for _, positive, _ in self._windows if positive)
        metrics = {
            "window_size": len(self._windows),
            "positive_windows": positive_count,
            "both_hands_visible": both_hands_visible,
            "detections": detections,
        }
        if self._completed:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.COMPLETED,
                confidence=1.0, reason="饰品观测已完成", updated_at=context.timestamp,
                metrics=metrics,
            ), []
        if len(self._windows) == 5 and positive_count >= 3:
            self._completed = True
            labels = sorted({str(item.get("class_name", "饰品")) for _, _, rows in self._windows for item in rows})
            evidence = EvidenceFrame(
                context.frame_idx, context.timestamp, context.camera_role.value, metrics=metrics
            )
            event = _event(
                self, context, EventPhase.COMPLETED, self._windows[0][0],
                positive_count / 5, {"present": True, "classes": labels}, metrics, [evidence],
            )
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, ObservationState.COMPLETED,
                confidence=positive_count / 5, value=event.value, reason="连续窗口检出手部饰品",
                started_at=self._windows[0][0], updated_at=context.timestamp, metrics=metrics,
            ), [event]
        if both_hands_visible and not detections:
            if self._clear_started is None:
                self._clear_started = context.timestamp
            if context.timestamp - self._clear_started >= 2.0:
                self._completed = True
                evidence = EvidenceFrame(
                    context.frame_idx, context.timestamp, context.camera_role.value, metrics=metrics
                )
                event = _event(
                    self, context, EventPhase.COMPLETED, self._clear_started, 0.9,
                    {"present": False, "classes": []}, metrics, [evidence],
                )
                return ObservationSnapshot(
                    self.observation_id, self.name, self.sop_step, ObservationState.COMPLETED,
                    confidence=0.9, value=event.value, reason="双手清晰可见2秒且未检出饰品",
                    started_at=self._clear_started, updated_at=context.timestamp, metrics=metrics,
                ), [event]
        else:
            self._clear_started = None
        state = ObservationState.CANDIDATE if both_hands_visible else ObservationState.UNCERTAIN
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step, state,
            confidence=positive_count / max(len(self._windows), 1),
            reason="等待饰品窗口稳定" if both_hands_visible else "需要同时看清两只手",
            started_at=self._clear_started, updated_at=context.timestamp, metrics=metrics,
        ), []

    def reset(self) -> None:
        self._windows.clear()
        self._clear_started = None
        self._completed = False
