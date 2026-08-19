"""Simplified front-camera rules for the six-step red-tea SOP prototype."""

from __future__ import annotations

from collections import deque
from math import atan2, degrees, hypot
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

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


BODY_NAMES = {"盖碗碗身", "盖碗（碗身）"}
LID_NAMES = {"盖碗碗盖", "盖碗（碗盖）"}


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


def _diag(item: Any) -> float:
    _, _, w, h = _bbox(item)
    return max(hypot(w, h), 1.0)


def _bbox_gap(a: Any, b: Any) -> float:
    ax, ay, aw, ah = _bbox(a)
    bx, by, bw, bh = _bbox(b)
    dx = max(bx - (ax + aw), ax - (bx + bw), 0.0)
    dy = max(by - (ay + ah), ay - (by + bh), 0.0)
    return hypot(dx, dy)


def _hand_item_gap(hand: Dict[str, Any], item: Any) -> float:
    hx, hy, hw, hh = [float(value) for value in hand.get("bbox", (0, 0, 0, 0))]
    x, y, w, h = _bbox(item)
    dx = max(x - (hx + hw), hx - (x + w), 0.0)
    dy = max(y - (hy + hh), hy - (y + h), 0.0)
    return hypot(dx, dy) / _diag(item)


def _intersects(rect: Sequence[float], zone: Sequence[float]) -> bool:
    x, y, w, h = [float(value) for value in rect]
    zx, zy, zw, zh = [float(value) for value in zone]
    return min(x + w, zx + zw) > max(x, zx) and min(y + h, zy + zh) > max(y, zy)


def _evidence(
    context: FrameContext,
    items: Sequence[Any] = (),
    metrics: Optional[Dict[str, Any]] = None,
) -> EvidenceFrame:
    return EvidenceFrame(
        frame_idx=context.frame_idx,
        timestamp=context.timestamp,
        camera_role=context.camera_role.value,
        track_ids=[
            int(track_id) for track_id in (
                getattr(item, "track_id", None) for item in items
            ) if track_id is not None
        ],
        metrics=dict(metrics or {}),
    )


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
        confidence=float(confidence),
        camera_role=context.camera_role.value,
        value=value,
        metrics=dict(metrics),
        evidence=list(evidence)[-8:],
        model_version=context.model_version,
        rule_version=observation.rule_version,
    )


class SetupReadyObservation:
    observation_id = "action_setup_ready"
    name = "茶具齐全且人员端坐"
    sop_step = 1
    camera_roles: Set[CameraRole] = {CameraRole.FRONT}
    rule_version = "2.0-six-step"
    required_items = {
        "盖碗碗身", "盖碗碗盖", "公道杯", "品茗杯", "茶荷", "茶巾",
        "茶夹", "茶拨", "烧水壶", "建水", "茶叶罐",
    }

    def __init__(self, stable_seconds: float = 0.8, recent_seconds: float = 2.5):
        self.stable_seconds = float(stable_seconds)
        self.recent_seconds = float(recent_seconds)
        self._last_seen: Dict[str, Tuple[float, float]] = {}
        self._history: Deque[Tuple[float, bool, float, EvidenceFrame]] = deque()
        self._started_at: Optional[float] = None
        self._started_emitted = False
        self._completed = False

    @staticmethod
    def required_classes_available(classes: Set[str]) -> bool:
        return SetupReadyObservation.required_items <= classes

    def _seated(self, context: FrameContext) -> tuple[bool, float, Dict[str, Any]]:
        if not context.pose_results:
            return False, 0.0, {"pose_visible": False}
        pose = context.pose_results[0]
        landmarks = np.asarray(pose.get("landmarks", []), dtype=float)
        visibility = np.asarray(pose.get("visibility", []), dtype=float)
        if len(landmarks) < 25:
            return False, 0.0, {"pose_visible": False}
        indices = [11, 12, 23, 24]
        visible = len(visibility) < 25 or float(visibility[indices].min()) >= 0.35
        shoulders = landmarks[[11, 12], :2]
        hips = landmarks[[23, 24], :2]
        shoulder_mid = shoulders.mean(axis=0)
        hip_mid = hips.mean(axis=0)
        shoulder_width = max(float(np.linalg.norm(shoulders[0] - shoulders[1])), 1.0)
        torso = hip_mid - shoulder_mid
        torso_angle = degrees(atan2(abs(float(torso[0])), max(abs(float(torso[1])), 1.0)))
        shoulder_level = abs(float(shoulders[0, 1] - shoulders[1, 1])) / shoulder_width
        upright = (
            visible
            and float(torso[1]) > shoulder_width * 0.35
            and torso_angle <= 28.0
            and shoulder_level <= 0.35
        )
        confidence = max(0.0, 1.0 - torso_angle / 45.0) if upright else 0.0
        return upright, confidence, {
            "pose_visible": visible,
            "torso_angle_degrees": round(torso_angle, 2),
            "shoulder_level_ratio": round(shoulder_level, 3),
        }

    def update(self, context: FrameContext):
        for item in context.detections:
            item_name = _name(item)
            if item_name in self.required_items:
                confidence = float(getattr(item, "confidence", 0.0))
                old = self._last_seen.get(item_name, (0.0, 0.0))[1]
                self._last_seen[item_name] = (context.timestamp, max(old, confidence))
        recent = {
            name for name, (timestamp, _) in self._last_seen.items()
            if context.timestamp - timestamp <= self.recent_seconds
        }
        cup_count = sum(1 for item in context.detections if _name(item) == "品茗杯")
        missing = sorted(self.required_items - recent)
        if cup_count < 3 and "品茗杯" not in missing:
            missing.append(f"品茗杯还需{3 - cup_count}只")
        seated, pose_confidence, pose_metrics = self._seated(context)
        complete_items = not missing
        passed = complete_items and seated
        item_confidences = [
            confidence for name, (_, confidence) in self._last_seen.items()
            if name in self.required_items
        ]
        confidence = min(item_confidences, default=0.0)
        if seated:
            confidence = min(confidence, max(pose_confidence, 0.45))
        metrics = {
            "missing_items": missing,
            "required_item_count": len(self.required_items),
            "cup_count": cup_count,
            "seated_upright": seated,
            **pose_metrics,
        }
        evidence = _evidence(context, context.detections, metrics)
        if self._completed:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.COMPLETED, confidence=confidence, value=True,
                reason="茶具齐全且人员已在桌前端坐",
                started_at=self._started_at, updated_at=context.timestamp,
                metrics=metrics,
            ), []
        self._history.append((context.timestamp, passed, confidence, evidence))
        # Keep a small amount of sampling slack so a valid continuous hold is
        # not discarded when the camera pipeline produces uneven frame gaps.
        cutoff = context.timestamp - self.stable_seconds - 0.1
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        events: List[ObservationEvent] = []
        if passed and self._started_at is None:
            self._started_at = context.timestamp
        if passed and not self._started_emitted:
            self._started_emitted = True
            events.append(_event(
                self, context, EventPhase.STARTED,
                self._started_at or context.timestamp, confidence, True,
                metrics, [evidence],
            ))
        positives = [row for row in self._history if row[1]]
        span = self._history[-1][0] - self._history[0][0] if len(self._history) > 1 else 0.0
        if len(positives) >= 3 and span >= self.stable_seconds * 0.9:
            self._completed = True
            event_confidence = float(np.mean([row[2] for row in positives]))
            events.append(_event(
                self, context, EventPhase.COMPLETED,
                self._started_at or context.timestamp, event_confidence, True,
                metrics, [row[3] for row in positives],
            ))
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.COMPLETED, confidence=event_confidence, value=True,
                reason="茶具齐全且人员已在桌前端坐",
                started_at=self._started_at, updated_at=context.timestamp,
                metrics=metrics,
            ), events
        reason = (
            "缺少：" + "、".join(missing[:5])
            if missing else "茶具已齐，等待人员保持端坐"
        )
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step,
            ObservationState.CANDIDATE if complete_items or seated else ObservationState.IDLE,
            confidence=confidence, reason=reason,
            updated_at=context.timestamp, metrics=metrics,
        ), events

    def reset(self) -> None:
        self._last_seen.clear()
        self._history.clear()
        self._started_at = None
        self._started_emitted = False
        self._completed = False


class TeaPreparationObservation:
    observation_id = "action_tea_canister_to_lotus"
    name = "从茶叶罐取茶到茶荷并归位"
    sop_step = 3
    camera_roles: Set[CameraRole] = {CameraRole.FRONT}
    rule_version = "2.0-object-state-sequence"

    def __init__(self, stable_seconds: float = 0.35):
        self.stable_seconds = float(stable_seconds)
        self._stage = "hold_canister"
        self._stage_started: Optional[float] = None
        self._started_at: Optional[float] = None
        self._started_emitted = False
        self._completed = False
        self._return_started: Optional[float] = None

    @staticmethod
    def required_classes_available(classes: Set[str]) -> bool:
        return {"茶叶罐", "茶荷", "茶拨"} <= classes

    def _advance(self, stage: str, timestamp: float) -> None:
        self._stage = stage
        self._stage_started = timestamp

    def update(self, context: FrameContext):
        groups = {
            name: [item for item in context.detections if _name(item) == name]
            for name in ("茶叶罐", "茶荷", "茶拨")
        }
        if any(not values for values in groups.values()):
            missing = [name for name, values in groups.items() if not values]
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.UNCERTAIN,
                reason="需要看清：" + "、".join(missing),
                updated_at=context.timestamp,
                metrics={"stage": self._stage, "missing": missing}, experimental=True,
            ), []
        canister = max(groups["茶叶罐"], key=lambda item: float(getattr(item, "confidence", 0)))
        lotus = max(groups["茶荷"], key=lambda item: float(getattr(item, "confidence", 0)))
        tea_pick = max(groups["茶拨"], key=lambda item: float(getattr(item, "confidence", 0)))
        hands = [hand for hand in context.hand_results if float(hand.get("confidence", 0)) >= 0.45]
        hand_canister = min((_hand_item_gap(hand, canister) for hand in hands), default=99.0)
        hand_pick = min((_hand_item_gap(hand, tea_pick) for hand in hands), default=99.0)
        canister_lotus = _bbox_gap(canister, lotus) / max(_diag(canister), _diag(lotus))
        pick_canister = _bbox_gap(tea_pick, canister) / max(_diag(tea_pick), _diag(canister))
        pick_lotus = _bbox_gap(tea_pick, lotus) / max(_diag(tea_pick), _diag(lotus))
        metrics = {
            "stage": self._stage,
            "hand_canister_gap": round(hand_canister, 3),
            "hand_pick_gap": round(hand_pick, 3),
            "canister_lotus_gap": round(canister_lotus, 3),
            "pick_canister_gap": round(pick_canister, 3),
            "pick_lotus_gap": round(pick_lotus, 3),
            "weight_verified": False,
        }
        confidence = min(float(getattr(item, "confidence", 0.0)) for item in (canister, lotus, tea_pick))
        evidence = _evidence(context, [canister, lotus, tea_pick], metrics)
        events: List[ObservationEvent] = []
        if self._completed:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.COMPLETED, confidence=confidence, value=True,
                reason="茶叶罐和茶拨已放回并远离茶荷",
                started_at=self._started_at, updated_at=context.timestamp,
                metrics=metrics, experimental=True,
            ), []

        condition = False
        next_stage = self._stage
        reason = ""
        if self._stage == "hold_canister":
            condition = hand_canister <= 0.35
            next_stage = "canister_near_lotus"
            reason = "等待手拿住茶叶罐"
        elif self._stage == "canister_near_lotus":
            condition = canister_lotus <= 0.75
            next_stage = "tea_pick_action"
            reason = "茶叶罐已拿起，等待靠近茶荷"
        elif self._stage == "tea_pick_action":
            condition = hand_pick <= 0.45 and min(pick_canister, pick_lotus) <= 0.75
            next_stage = "return_items"
            reason = "等待手持茶拨靠近茶叶罐或茶荷完成取茶"
        else:
            # The long, narrow tea pick can still have a modest normalized box
            # gap after it has visibly been returned.  Requiring 0.75 made the
            # return state unnecessarily difficult to reach in front view.
            returned = canister_lotus >= 0.55 and pick_lotus >= 0.55
            if returned:
                if self._return_started is None:
                    self._return_started = context.timestamp
                condition = context.timestamp - self._return_started >= 0.45
            else:
                self._return_started = None
            next_stage = "completed"
            reason = "等待茶叶罐和茶拨均远离茶荷"

        if condition:
            if self._stage_started is None:
                self._stage_started = context.timestamp
            if self._stage == "hold_canister" and self._started_at is None:
                self._started_at = context.timestamp
            if self._stage == "hold_canister" and not self._started_emitted:
                self._started_emitted = True
                events.append(_event(
                    self, context, EventPhase.STARTED,
                    self._started_at or context.timestamp, confidence,
                    "开始取茶", metrics, [evidence],
                ))
            if (
                next_stage == "completed"
                or context.timestamp - self._stage_started >= self.stable_seconds
            ):
                if next_stage == "completed":
                    self._completed = True
                    events.append(_event(
                        self, context, EventPhase.COMPLETED,
                        self._started_at or context.timestamp, confidence,
                        True, {**metrics, "stage": "completed"}, [evidence],
                    ))
                    return ObservationSnapshot(
                        self.observation_id, self.name, self.sop_step,
                        ObservationState.COMPLETED, confidence=confidence, value=True,
                        reason="茶叶罐和茶拨已放回并远离茶荷",
                        started_at=self._started_at, updated_at=context.timestamp,
                        metrics={**metrics, "stage": "completed"}, experimental=True,
                    ), events
                self._advance(next_stage, context.timestamp)
                reason = {
                    "canister_near_lotus": "已拿住茶叶罐，继续靠近茶荷",
                    "tea_pick_action": "茶叶罐已靠近茶荷，使用茶拨取茶",
                    "return_items": "取茶动作已确认，请放回茶叶罐和茶拨",
                }[next_stage]
        else:
            self._stage_started = None
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step,
            ObservationState.ACTIVE if self._started_at is not None else ObservationState.IDLE,
            confidence=confidence, value=self._stage, reason=reason,
            started_at=self._started_at, updated_at=context.timestamp,
            metrics={**metrics, "stage": self._stage}, experimental=True,
        ), events

    def reset(self) -> None:
        self._stage = "hold_canister"
        self._stage_started = None
        self._started_at = None
        self._started_emitted = False
        self._completed = False
        self._return_started = None


class LotusAppreciationObservation:
    observation_id = "action_hold_lotus"
    name = "双手托举茶荷从左向右赏茶"
    sop_step = 4
    camera_roles: Set[CameraRole] = {CameraRole.FRONT}
    rule_version = "2.0-two-hand-direction"

    def __init__(self):
        self._rest_centers: Deque[Tuple[float, float]] = deque(maxlen=15)
        self._rest_center: Optional[Tuple[float, float]] = None
        self._stage = "waiting_grip"
        self._started_at: Optional[float] = None
        self._stage_started: Optional[float] = None
        self._start_x: Optional[float] = None
        self._max_x: Optional[float] = None
        self._completed = False

    @staticmethod
    def _two_sides(hands: Sequence[Dict[str, Any]], lotus: Any) -> bool:
        x, y, w, h = _bbox(lotus)
        left_zone = (x - 0.45 * w, y - 0.55 * h, 1.0 * w, 2.1 * h)
        right_zone = (x + 0.45 * w, y - 0.55 * h, 1.0 * w, 2.1 * h)
        valid = [hand for hand in hands if float(hand.get("confidence", 0)) >= 0.45]
        for left_index, left in enumerate(valid):
            if not _intersects(left.get("bbox", (0, 0, 0, 0)), left_zone):
                continue
            for right_index, right in enumerate(valid):
                if left_index == right_index:
                    continue
                if _intersects(right.get("bbox", (0, 0, 0, 0)), right_zone):
                    return True
        return False

    def update(self, context: FrameContext):
        lotuses = [item for item in context.detections if _name(item) == "茶荷"]
        if not lotuses:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.UNCERTAIN, reason="未检测到茶荷",
                updated_at=context.timestamp, experimental=True,
            ), []
        lotus = max(lotuses, key=lambda item: float(getattr(item, "confidence", 0)))
        hands = list(context.hand_results)
        two_sides = self._two_sides(hands, lotus)
        cx, cy = _center(lotus)
        _, _, w, h = _bbox(lotus)
        if not two_sides and self._stage == "waiting_grip":
            self._rest_centers.append((cx, cy))
            if len(self._rest_centers) >= 5:
                values = np.asarray(self._rest_centers)
                self._rest_center = (float(np.median(values[:, 0])), float(np.median(values[:, 1])))
        movement = (
            hypot(cx - self._rest_center[0], cy - self._rest_center[1])
            if self._rest_center is not None else 0.0
        )
        lift_threshold = max(hypot(w, h) * 0.12, context.frame_shape[0] * 0.012)
        lifted = self._rest_center is not None and movement >= lift_threshold
        metrics = {
            "stage": self._stage,
            "two_hands_on_sides": two_sides,
            "rest_center": self._rest_center,
            "movement_from_table": round(movement, 2),
            "lift_threshold": round(lift_threshold, 2),
            "lifted_from_table": lifted,
            "start_x": self._start_x,
            "max_x": self._max_x,
            "current_x": round(cx, 2),
            "chest_height_required": False,
        }
        confidence = float(getattr(lotus, "confidence", 0.0)) * (0.9 if two_sides else 0.5)
        evidence = _evidence(context, [lotus], metrics)
        events: List[ObservationEvent] = []
        if self._completed:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.COMPLETED, confidence=confidence, value=True,
                reason="茶荷已从左向右展示并开始向左回移",
                started_at=self._started_at, updated_at=context.timestamp,
                metrics=metrics, experimental=True,
            ), []
        if self._stage == "waiting_grip":
            if not two_sides:
                reason = "等待两只手分别接触茶荷左右两端"
            elif self._rest_center is None:
                reason = "请先让茶荷在桌面静置片刻，再双手拿起"
            elif not lifted:
                reason = "双手已在茶荷两端，等待茶荷离开桌面位置"
            else:
                self._stage = "moving_right"
                self._started_at = context.timestamp
                self._start_x = cx
                self._max_x = cx
                events.append(_event(
                    self, context, EventPhase.STARTED, self._started_at,
                    confidence, "开始赏茶", metrics, [evidence],
                ))
                reason = "已拿起茶荷，请从左向右依次展示"
        elif self._stage == "moving_right":
            self._max_x = max(self._max_x or cx, cx)
            right_distance = (self._max_x or cx) - (self._start_x or cx)
            if right_distance >= max(0.45 * w, 35.0):
                self._stage = "waiting_left_return"
                self._stage_started = context.timestamp
                reason = "已完成向右展示，开始向左移动即可结束赏茶"
            else:
                reason = "保持双手托住茶荷并继续向右展示"
        else:
            left_return = (self._max_x or cx) - cx
            if two_sides and left_return >= max(0.25 * w, 20.0):
                if self._stage_started is None:
                    self._stage_started = context.timestamp
                if context.timestamp - self._stage_started >= 0.2:
                    self._completed = True
                    events.append(_event(
                        self, context, EventPhase.COMPLETED,
                        self._started_at or context.timestamp, confidence,
                        True, {**metrics, "left_return": round(left_return, 2)}, [evidence],
                    ))
                    return ObservationSnapshot(
                        self.observation_id, self.name, self.sop_step,
                        ObservationState.COMPLETED, confidence=confidence, value=True,
                        reason="茶荷已从左向右展示并开始向左回移",
                        started_at=self._started_at, updated_at=context.timestamp,
                        metrics=metrics, experimental=True,
                    ), events
            else:
                self._stage_started = None
            reason = "等待茶荷开始向左回移，回移后进入投茶"
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step,
            ObservationState.ACTIVE if self._started_at is not None else ObservationState.IDLE,
            confidence=confidence, value=self._stage, reason=reason,
            started_at=self._started_at, updated_at=context.timestamp,
            metrics={**metrics, "stage": self._stage}, experimental=True,
        ), events

    def reset(self) -> None:
        self._rest_centers.clear()
        self._rest_center = None
        self._stage = "waiting_grip"
        self._started_at = None
        self._stage_started = None
        self._start_x = None
        self._max_x = None
        self._completed = False


class TeaTransferObservation:
    observation_id = "action_tea_lotus_to_gaiwan"
    name = "茶荷配合茶拨向盖碗投茶并归位"
    sop_step = 4
    camera_roles: Set[CameraRole] = {CameraRole.FRONT}
    rule_version = "2.0-pose-and-return"

    def __init__(self):
        self._stage = "waiting_pour"
        self._active_since: Optional[float] = None
        self._return_since: Optional[float] = None
        self._started_at: Optional[float] = None
        self._completed = False

    @staticmethod
    def required_classes_available(classes: Set[str]) -> bool:
        return "茶荷" in classes and "茶拨" in classes and bool(BODY_NAMES & classes)

    def update(self, context: FrameContext):
        lotuses = [item for item in context.detections if _name(item) == "茶荷"]
        picks = [item for item in context.detections if _name(item) == "茶拨"]
        bodies = [item for item in context.detections if _name(item) in BODY_NAMES]
        if not lotuses or not picks or not bodies:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.UNCERTAIN,
                reason="需要同时看清茶荷、茶拨和盖碗碗身",
                updated_at=context.timestamp, experimental=True,
            ), []
        lotus = max(lotuses, key=lambda item: float(getattr(item, "confidence", 0)))
        tea_pick = max(picks, key=lambda item: float(getattr(item, "confidence", 0)))
        body = max(bodies, key=lambda item: float(getattr(item, "confidence", 0)))
        interactions = [
            row for row in context.extras.get("pour_interactions", [])
            if row.get("source") == "茶荷" and row.get("target") in BODY_NAMES
        ]
        interaction = max(interactions, key=lambda row: float(row.get("confidence", 0)), default=None)
        pick_lotus = _bbox_gap(tea_pick, lotus) / max(_diag(tea_pick), _diag(lotus))
        lotus_body = _bbox_gap(lotus, body) / max(_diag(lotus), _diag(body))
        pick_body = _bbox_gap(tea_pick, body) / max(_diag(tea_pick), _diag(body))
        pouring = interaction is not None and pick_lotus <= 0.75
        returned = lotus_body >= 0.8 and pick_body >= 0.8
        metrics = {
            "stage": self._stage,
            "pose_interaction": interaction is not None,
            "pick_lotus_gap": round(pick_lotus, 3),
            "lotus_gaiwan_gap": round(lotus_body, 3),
            "pick_gaiwan_gap": round(pick_body, 3),
            "tea_leaf_drop_verified": False,
        }
        if interaction:
            metrics.update({
                "tilt_delta_degrees": interaction.get("tilt_delta_degrees"),
                "outlet_point": interaction.get("outlet_point"),
            })
        confidence = min(float(getattr(item, "confidence", 0)) for item in (lotus, tea_pick, body))
        if interaction:
            confidence = min(confidence, float(interaction.get("confidence", 0)))
        evidence = _evidence(context, [lotus, tea_pick, body], metrics)
        events: List[ObservationEvent] = []
        if self._completed:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.COMPLETED, confidence=confidence, value=True,
                reason="茶荷和茶拨已远离盖碗并归位",
                started_at=self._started_at, updated_at=context.timestamp,
                metrics=metrics, experimental=True,
            ), []
        if self._stage == "waiting_pour":
            if pouring:
                if self._active_since is None:
                    self._active_since = context.timestamp
                    self._started_at = context.timestamp
                    events.append(_event(
                        self, context, EventPhase.STARTED, self._started_at,
                        confidence, "开始投茶", metrics, [evidence],
                    ))
                if context.timestamp - self._active_since >= 0.45:
                    self._stage = "waiting_return"
                    self._return_since = None
                    reason = "投茶姿态已确认，请将茶荷和茶拨放回"
                else:
                    reason = "茶荷已倾斜且茶拨靠近，等待投茶姿态稳定"
            else:
                self._active_since = None
                reason = (
                    "等待茶荷倾斜、出茶端对准盖碗且茶拨靠近茶荷"
                    if interaction is None else "茶荷已对准盖碗，等待茶拨靠近茶荷"
                )
        else:
            if returned:
                if self._return_since is None:
                    self._return_since = context.timestamp
                if context.timestamp - self._return_since >= 0.45:
                    self._completed = True
                    events.append(_event(
                        self, context, EventPhase.COMPLETED,
                        self._started_at or context.timestamp, confidence, True,
                        {**metrics, "stage": "completed"}, [evidence],
                    ))
                    return ObservationSnapshot(
                        self.observation_id, self.name, self.sop_step,
                        ObservationState.COMPLETED, confidence=confidence, value=True,
                        reason="茶荷和茶拨已远离盖碗并归位",
                        started_at=self._started_at, updated_at=context.timestamp,
                        metrics=metrics, experimental=True,
                    ), events
            else:
                self._return_since = None
            reason = "等待茶荷和茶拨均远离盖碗"
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step,
            ObservationState.ACTIVE if self._started_at is not None else ObservationState.IDLE,
            confidence=confidence, value=self._stage, reason=reason,
            started_at=self._started_at, updated_at=context.timestamp,
            metrics={**metrics, "stage": self._stage}, experimental=True,
        ), events

    def reset(self) -> None:
        self._stage = "waiting_pour"
        self._active_since = None
        self._return_since = None
        self._started_at = None
        self._completed = False


class SmellObservation:
    observation_id = "action_open_lid_smell"
    name = "开盖靠近鼻部闻香"
    sop_step = 4
    camera_roles: Set[CameraRole] = {CameraRole.FRONT}
    rule_version = "2.0-occlusion-tolerant"

    def __init__(self):
        self._stage = "waiting_open"
        self._started_at: Optional[float] = None
        self._near_since: Optional[float] = None
        self._away_since: Optional[float] = None
        self._completed = False
        self._pickup_seen = False
        self._open_signal = "none"

    @staticmethod
    def _iou(a: Any, b: Any) -> float:
        ax, ay, aw, ah = _bbox(a)
        bx, by, bw, bh = _bbox(b)
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return inter / max(aw * ah + bw * bh - inter, 1.0)

    def update(self, context: FrameContext):
        if not context.pose_results:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.UNCERTAIN,
                reason="未检测到人体鼻部；鼻部由MediaPipe自动提供，无需人工标注",
                updated_at=context.timestamp, experimental=True,
            ), []
        pose = np.asarray(context.pose_results[0].get("landmarks", []), dtype=float)
        if len(pose) < 13:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.UNCERTAIN, reason="人体关键点不完整",
                updated_at=context.timestamp, experimental=True,
            ), []
        nose = pose[0, :2]
        shoulder_width = max(float(np.linalg.norm(pose[11, :2] - pose[12, :2])), 1.0)
        bodies = [item for item in context.detections if _name(item) in BODY_NAMES]
        lids = [item for item in context.detections if _name(item) in LID_NAMES]
        body = max(bodies, key=lambda item: float(getattr(item, "confidence", 0)), default=None)
        lid = max(lids, key=lambda item: float(getattr(item, "confidence", 0)), default=None)
        is_open = False
        if body is not None and lid is not None:
            distance = hypot(*np.subtract(_center(body), _center(lid))) / max(_bbox(body)[2], 1.0)
            is_open = self._iou(body, lid) < 0.18 or distance > 0.4
        body_point = _center(body) if body is not None else None
        held_candidates: List[Tuple[float, float]] = []
        if lid is not None:
            held_candidates.append(_center(lid))
        held_candidates.extend(
            tuple(map(float, hand.get("center", (0, 0))))
            for hand in context.hand_results
            if float(hand.get("confidence", 0)) >= 0.45
        )
        candidates = list(held_candidates)
        if body_point is not None:
            candidates.append(body_point)
        distance_ratio = min(
            (hypot(point[0] - nose[0], point[1] - nose[1]) / shoulder_width for point in candidates),
            default=99.0,
        )
        held_distance_ratio = min(
            (hypot(point[0] - nose[0], point[1] - nose[1]) / shoulder_width for point in held_candidates),
            default=99.0,
        )
        body_distance_ratio = (
            hypot(body_point[0] - nose[0], body_point[1] - nose[1]) / shoulder_width
            if body_point is not None else 99.0
        )
        near_nose = distance_ratio <= 0.85
        # A gaiwan body left on the table can sit inside the hysteresis band in
        # front view. It must not prevent completion after the held lid/hand
        # has clearly moved away from the face.
        away_from_nose = held_distance_ratio >= 1.0 and body_distance_ratio >= 0.90
        hand_near_gaiwan = False
        if body is not None:
            hand_near_gaiwan = any(
                _hand_item_gap(hand, body) <= 0.45
                for hand in context.hand_results
                if float(hand.get("confidence", 0)) >= 0.45
            )
        if self._stage == "waiting_open" and hand_near_gaiwan:
            self._pickup_seen = True
        inferred_open = (
            self._stage == "waiting_open"
            and self._pickup_seen
            and body is not None
            and near_nose
        )
        open_detected = is_open or inferred_open
        if is_open:
            self._open_signal = "lid_body_separation"
        elif inferred_open:
            self._open_signal = "hand_from_gaiwan_to_nose"
        metrics = {
            "stage": self._stage,
            "body_visible": body is not None,
            "lid_visible": lid is not None,
            "open_evidence": open_detected,
            "open_signal": self._open_signal,
            "hand_pickup_seen": self._pickup_seen,
            "hand_near_gaiwan": hand_near_gaiwan,
            "near_nose": near_nose,
            "nose_distance_shoulder": round(distance_ratio, 3),
            "held_nose_distance_shoulder": round(held_distance_ratio, 3),
            "body_nose_distance_shoulder": round(body_distance_ratio, 3),
            "nose_source": "mediapipe_pose_index_0",
        }
        visible_items = [item for item in (body, lid) if item is not None]
        confidence = min(
            [float(getattr(item, "confidence", 0)) for item in visible_items] + [0.8]
        )
        evidence = _evidence(context, visible_items, metrics)
        events: List[ObservationEvent] = []
        if self._completed:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.COMPLETED, confidence=confidence, value=True,
                reason="已完成开盖、靠近鼻部闻香并离开",
                started_at=self._started_at, updated_at=context.timestamp,
                metrics=metrics, experimental=True,
            ), []
        if self._stage == "waiting_open":
            if open_detected:
                self._stage = "waiting_near_nose"
                self._started_at = context.timestamp
                if near_nose:
                    self._near_since = context.timestamp
                events.append(_event(
                    self, context, EventPhase.STARTED, self._started_at,
                    confidence, "已开盖", metrics, [evidence],
                ))
                reason = "已确认开盖，请将盖碗或持盖手靠近鼻部"
            else:
                reason = "等待开盖，或手从盖碗位置移动到鼻部"
        elif self._stage == "waiting_near_nose":
            if near_nose:
                if self._near_since is None:
                    self._near_since = context.timestamp
                if context.timestamp - self._near_since >= 0.5:
                    self._stage = "waiting_away"
                    reason = "闻香保持已确认，请将盖碗或手移离鼻部"
                else:
                    reason = "已靠近鼻部，保持闻香约0.5秒"
            else:
                self._near_since = None
                reason = "允许碗身和碗盖被遮挡，等待持盖手靠近鼻部"
        else:
            if away_from_nose:
                if self._away_since is None:
                    self._away_since = context.timestamp
                if context.timestamp - self._away_since >= 0.3:
                    self._completed = True
                    events.append(_event(
                        self, context, EventPhase.COMPLETED,
                        self._started_at or context.timestamp, confidence, True,
                        {**metrics, "stage": "completed"}, [evidence],
                    ))
                    return ObservationSnapshot(
                        self.observation_id, self.name, self.sop_step,
                        ObservationState.COMPLETED, confidence=confidence, value=True,
                        reason="已完成开盖、靠近鼻部闻香并离开",
                        started_at=self._started_at, updated_at=context.timestamp,
                        metrics=metrics, experimental=True,
                    ), events
            else:
                self._away_since = None
            reason = "闻香已确认，等待盖碗或持盖手离开鼻部"
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step,
            ObservationState.ACTIVE if self._started_at is not None else ObservationState.IDLE,
            confidence=confidence, value=self._stage, reason=reason,
            started_at=self._started_at, updated_at=context.timestamp,
            metrics={**metrics, "stage": self._stage}, experimental=True,
        ), events

    def reset(self) -> None:
        self._stage = "waiting_open"
        self._started_at = None
        self._near_since = None
        self._away_since = None
        self._completed = False
        self._pickup_seen = False
        self._open_signal = "none"


class SimpleWaterInjectionObservation:
    observation_id = "action_water_injection"
    name = "烧水壶向盖碗注水"
    sop_step = 5
    camera_roles: Set[CameraRole] = {CameraRole.FRONT}
    rule_version = "2.0-relaxed-pose"

    def __init__(self):
        self._active_since: Optional[float] = None
        self._started_at: Optional[float] = None
        self._started_emitted = False
        self._completed = False

    @staticmethod
    def required_classes_available(classes: Set[str]) -> bool:
        return "烧水壶" in classes and bool(BODY_NAMES & classes)

    def update(self, context: FrameContext):
        rows = [
            row for row in context.extras.get("pour_interactions", [])
            if row.get("source") == "烧水壶" and row.get("target") in BODY_NAMES
        ]
        interaction = max(rows, key=lambda row: float(row.get("confidence", 0)), default=None)
        active = interaction is not None and float(interaction.get("confidence", 0)) >= 0.35
        metrics = dict(interaction or {})
        metrics.update({
            "orbit_required": False,
            "lift_baseline_required": False,
            "liquid_verified": False,
        })
        confidence = float(interaction.get("confidence", 0)) if interaction else 0.0
        evidence = _evidence(context, metrics=metrics)
        events: List[ObservationEvent] = []
        if self._completed:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.COMPLETED, confidence=confidence, value=True,
                reason="已确认壶嘴靠近盖碗并保持倾斜注水姿态",
                started_at=self._started_at, updated_at=context.timestamp,
                metrics=metrics, experimental=True,
            ), []
        if active:
            if self._active_since is None:
                self._active_since = context.timestamp
                self._started_at = context.timestamp
            if not self._started_emitted:
                self._started_emitted = True
                events.append(_event(
                    self, context, EventPhase.STARTED,
                    self._started_at, confidence, "开始注水", metrics, [evidence],
                ))
            elapsed = context.timestamp - self._active_since
            if elapsed >= 0.55:
                self._completed = True
                events.append(_event(
                    self, context, EventPhase.COMPLETED,
                    self._started_at, confidence, True, metrics, [evidence],
                ))
                state = ObservationState.COMPLETED
                reason = "已确认壶嘴靠近盖碗并保持倾斜注水姿态"
            else:
                state = ObservationState.ACTIVE
                reason = f"注水姿态已成立，继续保持{max(0.0, 0.55 - elapsed):.1f}秒"
        else:
            self._active_since = None
            self._started_at = None
            self._started_emitted = False
            state = ObservationState.IDLE
            reason = "等待手握烧水壶、壶嘴靠近盖碗并形成倾斜"
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step, state,
            confidence=confidence, value=active, reason=reason,
            started_at=self._started_at, updated_at=context.timestamp,
            metrics=metrics, experimental=True,
        ), events

    def reset(self) -> None:
        self._active_since = None
        self._started_at = None
        self._started_emitted = False
        self._completed = False


class RelaxedLidClosureObservation:
    observation_id = "action_gaiwan_lid_close_brew"
    name = "注水后盖合碗盖并开始计时"
    sop_step = 5
    camera_roles: Set[CameraRole] = {CameraRole.FRONT}
    rule_version = "2.0-direct-close"

    def __init__(self):
        self._closed_since: Optional[float] = None
        self._started_at: Optional[float] = None
        self._started_emitted = False
        self._completed = False
        self._body_history: Deque[Tuple[float, float, float]] = deque()
        self._hand_near_seen = False
        self._hand_away_since: Optional[float] = None

    @staticmethod
    def required_classes_available(classes: Set[str]) -> bool:
        return bool(BODY_NAMES & classes) and bool(LID_NAMES & classes)

    def update(self, context: FrameContext):
        bodies = [item for item in context.detections if _name(item) in BODY_NAMES]
        lids = [item for item in context.detections if _name(item) in LID_NAMES]
        if not bodies:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.UNCERTAIN,
                reason="等待检测到盖碗碗身",
                updated_at=context.timestamp,
                metrics={"body_count": len(bodies), "lid_count": len(lids)},
                experimental=True,
            ), []
        body = max(bodies, key=lambda item: float(getattr(item, "confidence", 0)))
        bx, by = _center(body)
        self._body_history.append((context.timestamp, bx, by))
        while self._body_history and self._body_history[0][0] < context.timestamp - 0.8:
            self._body_history.popleft()
        body_span = (
            self._body_history[-1][0] - self._body_history[0][0]
            if len(self._body_history) > 1 else 0.0
        )
        body_motion = max(
            (hypot(bx - old_x, by - old_y) for _, old_x, old_y in self._body_history),
            default=0.0,
        ) / _diag(body)
        body_stable = body_span >= 0.4 and body_motion <= 0.12
        hand_gap = min(
            (_hand_item_gap(hand, body) for hand in context.hand_results),
            default=99.0,
        )
        if hand_gap <= 0.45:
            self._hand_near_seen = True
            self._hand_away_since = None
        elif self._hand_near_seen and hand_gap >= 0.60 and self._hand_away_since is None:
            self._hand_away_since = context.timestamp
        hand_released = (
            self._hand_away_since is not None
            and context.timestamp - self._hand_away_since >= 0.3
        )
        pairs = []
        for candidate_body in bodies:
            candidate_bx, candidate_by = _center(candidate_body)
            _, _, bw, bh = _bbox(candidate_body)
            for lid in lids:
                lx, ly = _center(lid)
                ratio = hypot(candidate_bx - lx, candidate_by - ly) / max(bw, 1.0)
                vertical = abs(candidate_by - ly) / max(bh, 1.0)
                pairs.append((ratio + 0.25 * vertical, ratio, vertical, candidate_body, lid))
        if pairs:
            _, distance_ratio, vertical_ratio, body, lid = min(pairs, key=lambda row: row[0])
            direct_closed = distance_ratio <= 0.60 and vertical_ratio <= 0.60
        else:
            lid = None
            distance_ratio = 99.0
            vertical_ratio = 99.0
            direct_closed = False
        inferred_closed = not lids and self._hand_near_seen and hand_released and body_stable
        closed = direct_closed or inferred_closed
        confidence = float(getattr(body, "confidence", 0))
        if lid is not None:
            confidence = min(confidence, float(getattr(lid, "confidence", 0)))
        elif inferred_closed:
            confidence *= 0.65
        metrics = {
            "body_count": len(bodies),
            "lid_count": len(lids),
            "lid_body_distance_width": round(distance_ratio, 3),
            "lid_body_vertical_height": round(vertical_ratio, 3),
            "closed_relation": closed,
            "closure_signal": "body_lid_overlap" if direct_closed else (
                "stable_body_after_hand_release" if inferred_closed else "none"
            ),
            "body_stable": body_stable,
            "body_motion_ratio": round(body_motion, 3),
            "hand_near_seen": self._hand_near_seen,
            "hand_released": hand_released,
            "open_prerequisite_required": False,
        }
        evidence = _evidence(context, [item for item in (body, lid) if item is not None], metrics)
        events: List[ObservationEvent] = []
        if self._completed:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.COMPLETED, confidence=confidence, value=True,
                reason="碗身和碗盖已稳定重合，冲泡计时已开始",
                started_at=self._started_at, updated_at=context.timestamp,
                metrics=metrics, experimental=True,
            ), []
        if closed:
            if self._closed_since is None:
                self._closed_since = context.timestamp
                self._started_at = context.timestamp
            if not self._started_emitted:
                self._started_emitted = True
                events.append(_event(
                    self, context, EventPhase.STARTED,
                    self._started_at, confidence, "正在确认盖合", metrics, [evidence],
                ))
            elapsed = context.timestamp - self._closed_since
            if elapsed >= 0.4:
                self._completed = True
                events.append(_event(
                    self, context, EventPhase.COMPLETED,
                    self._started_at, confidence, True, metrics, [evidence],
                ))
                state = ObservationState.COMPLETED
                reason = "碗身和碗盖已稳定重合，冲泡计时已开始"
            else:
                state = ObservationState.CANDIDATE
                reason = "已检测到盖合关系，等待稳定约0.4秒"
        else:
            self._closed_since = None
            self._started_at = None
            self._started_emitted = False
            state = ObservationState.IDLE
            reason = (
                "碗盖暂时漏检：等待合盖手离开且碗身保持稳定"
                if not lids else "等待碗盖中心与碗身中心靠近并稳定"
            )
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step, state,
            confidence=confidence, value=closed, reason=reason,
            started_at=self._started_at, updated_at=context.timestamp,
            metrics=metrics, experimental=True,
        ), events

    def reset(self) -> None:
        self._closed_since = None
        self._started_at = None
        self._started_emitted = False
        self._completed = False
        self._body_history.clear()
        self._hand_near_seen = False
        self._hand_away_since = None


class ReturnAwareDecantObservation:
    observation_id = "action_gaiwan_to_pitcher"
    name = "盖碗向公道杯出汤并放回"
    sop_step = 5
    camera_roles: Set[CameraRole] = {CameraRole.FRONT}
    rule_version = "2.0-grip-pour-return"

    def __init__(self):
        self._stage = "waiting_grip"
        self._origin: Optional[Tuple[float, float]] = None
        self._scale = 1.0
        self._started_at: Optional[float] = None
        self._pour_since: Optional[float] = None
        self._return_since: Optional[float] = None
        self._completed = False
        self._wait_for_release = False

    @classmethod
    def required_classes_available(cls, classes: Set[str]) -> bool:
        return bool(BODY_NAMES & classes) and "公道杯" in classes

    def update(self, context: FrameContext):
        bodies = [item for item in context.detections if _name(item) in BODY_NAMES]
        pitchers = [item for item in context.detections if _name(item) == "公道杯"]
        if not bodies or not pitchers:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.UNCERTAIN,
                reason="需要同时看清盖碗碗身和公道杯",
                updated_at=context.timestamp, experimental=True,
            ), []
        body = max(bodies, key=lambda item: float(getattr(item, "confidence", 0)))
        pitcher = max(pitchers, key=lambda item: float(getattr(item, "confidence", 0)))
        center = _center(body)
        if self._origin is None:
            self._origin = center
            self._scale = max(_diag(body), _diag(pitcher))
        hand_gap = min((_hand_item_gap(hand, body) for hand in context.hand_results), default=99.0)
        hand_grip = hand_gap <= 0.35
        direct = next((
            row for row in context.extras.get("pour_interactions", [])
            if row.get("source") in BODY_NAMES and row.get("target") == "公道杯"
        ), None)
        bx, by = center
        px, py = _center(pitcher)
        gap_ratio = _bbox_gap(body, pitcher) / self._scale
        moved = hypot(bx - self._origin[0], by - self._origin[1]) / self._scale
        fallback_pour = hand_grip and gap_ratio <= 0.65 and by <= py + 0.4 * _bbox(pitcher)[3] and moved >= 0.15
        pouring = direct is not None or fallback_pour
        returned = hypot(bx - self._origin[0], by - self._origin[1]) / self._scale <= 0.35
        confidence = min(float(getattr(body, "confidence", 0)), float(getattr(pitcher, "confidence", 0)))
        if direct:
            confidence = min(confidence, float(direct.get("confidence", 0)))
        metrics = {
            "stage": self._stage,
            "hand_grip": hand_grip,
            "hand_gap": round(hand_gap, 3),
            "pose_pour": direct is not None,
            "source_target_gap": round(gap_ratio, 3),
            "moved_from_origin": round(moved, 3),
            "returned_to_origin": returned,
            "liquid_verified": False,
        }
        evidence = _evidence(context, [body, pitcher], metrics)
        events: List[ObservationEvent] = []
        if self._completed:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.COMPLETED, confidence=confidence, value=True,
                reason="出汤完成且盖碗已放回原位",
                started_at=self._started_at, updated_at=context.timestamp,
                metrics=metrics, experimental=True,
            ), []
        if self._stage == "waiting_grip":
            if self._wait_for_release and not hand_grip:
                self._wait_for_release = False
                reason = "盖合操作的手已离开，重新握住盖碗时结束冲泡计时"
            elif self._wait_for_release:
                reason = "请先将盖合碗盖的手移开，再重新握住盖碗开始出汤"
            elif hand_grip:
                self._stage = "waiting_pour"
                self._started_at = context.timestamp
                events.append(_event(
                    self, context, EventPhase.STARTED, self._started_at,
                    confidence, "手已握住盖碗", metrics, [evidence],
                ))
                reason = "已握住盖碗，请移至公道杯上方并倾斜"
            else:
                reason = "等待手握住盖碗；握住时结束冲泡计时"
        elif self._stage == "waiting_pour":
            if pouring:
                if self._pour_since is None:
                    self._pour_since = context.timestamp
                if context.timestamp - self._pour_since >= 0.45:
                    self._stage = "waiting_return"
                    reason = "出汤姿态已确认，请将盖碗放回原位"
                else:
                    reason = "盖碗已位于公道杯上方，保持倾斜出汤"
            else:
                self._pour_since = None
                reason = "等待盖碗移至公道杯上方并形成倾斜"
        else:
            if returned and not pouring:
                if self._return_since is None:
                    self._return_since = context.timestamp
                if context.timestamp - self._return_since >= 0.45:
                    self._completed = True
                    events.append(_event(
                        self, context, EventPhase.COMPLETED,
                        self._started_at or context.timestamp, confidence, True,
                        {**metrics, "stage": "completed"}, [evidence],
                    ))
                    return ObservationSnapshot(
                        self.observation_id, self.name, self.sop_step,
                        ObservationState.COMPLETED, confidence=confidence, value=True,
                        reason="出汤完成且盖碗已放回原位",
                        started_at=self._started_at, updated_at=context.timestamp,
                        metrics=metrics, experimental=True,
                    ), events
            else:
                self._return_since = None
            reason = "等待盖碗放回开始位置附近"
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step,
            ObservationState.ACTIVE if self._started_at is not None else ObservationState.IDLE,
            confidence=confidence, value=self._stage, reason=reason,
            started_at=self._started_at, updated_at=context.timestamp,
            metrics={**metrics, "stage": self._stage}, experimental=True,
        ), events

    def reset(self) -> None:
        self._stage = "waiting_grip"
        self._origin = None
        self._scale = 1.0
        self._started_at = None
        self._pour_since = None
        self._return_since = None
        self._completed = False
        self._wait_for_release = False

    def arm_for_brew(self) -> None:
        """Reset before timing and reject the hand left over from lid closing."""
        self.reset()
        self._wait_for_release = True


class ReturnAwareDistributionObservation:
    observation_id = "action_tea_distribution"
    name = "公道杯依次分茶并放回"
    sop_step = 6
    camera_roles: Set[CameraRole] = {CameraRole.FRONT}
    rule_version = "2.0-grip-sequence-return"

    def __init__(self):
        self._stage = "waiting_grip"
        self._origin: Optional[Tuple[float, float]] = None
        self._scale = 1.0
        self._started_at: Optional[float] = None
        self._current_target: Any = None
        self._target_since: Optional[float] = None
        self._targets: List[Tuple[Any, float]] = []
        self._return_since: Optional[float] = None
        self._completed = False
        self._failed = False

    @staticmethod
    def required_classes_available(classes: Set[str]) -> bool:
        return {"公道杯", "品茗杯"} <= classes

    def update(self, context: FrameContext):
        pitchers = [item for item in context.detections if _name(item) == "公道杯"]
        cups = [item for item in context.detections if _name(item) == "品茗杯"]
        if not pitchers or len(cups) < 3:
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step,
                ObservationState.UNCERTAIN,
                reason=f"需要看清公道杯和三个品茗杯，当前杯数{len(cups)}",
                updated_at=context.timestamp, experimental=True,
            ), []
        pitcher = max(pitchers, key=lambda item: float(getattr(item, "confidence", 0)))
        center = _center(pitcher)
        if self._origin is None:
            self._origin = center
            self._scale = _diag(pitcher)
        hand_gap = min((_hand_item_gap(hand, pitcher) for hand in context.hand_results), default=99.0)
        hand_grip = hand_gap <= 0.35
        moved = hypot(
            center[0] - self._origin[0], center[1] - self._origin[1]
        ) / self._scale
        interactions = [
            row for row in context.extras.get("pour_interactions", [])
            if row.get("source") == "公道杯" and row.get("target") == "品茗杯"
        ]
        interaction = max(interactions, key=lambda row: float(row.get("confidence", 0)), default=None)
        if interaction is None and hand_grip and moved >= 0.10:
            nearest_cup = min(cups, key=lambda cup: _bbox_gap(pitcher, cup))
            cup_gap = _bbox_gap(pitcher, nearest_cup) / max(
                self._scale, _diag(nearest_cup)
            )
            if cup_gap <= 0.85:
                cup_center = _center(nearest_cup)
                interaction = {
                    "source": "公道杯",
                    "target": "品茗杯",
                    "confidence": min(
                        float(getattr(pitcher, "confidence", 0)),
                        float(getattr(nearest_cup, "confidence", 0)),
                    ) * 0.65,
                    "target_track_id": getattr(nearest_cup, "track_id", None),
                    "target_center": list(cup_center),
                    "signal_source": "front_geometry_fallback",
                    "liquid_verified": False,
                }
        target_id = None
        target_x = None
        if interaction is not None:
            target_id = interaction.get("target_track_id")
            target_center = interaction.get("target_center", [0, 0])
            target_x = float(target_center[0])
            if target_id is None:
                target_id = round(target_x / 40.0)
        returned = hypot(center[0] - self._origin[0], center[1] - self._origin[1]) / self._scale <= 0.4
        positions = [row[1] for row in self._targets]
        direction = None
        if len(positions) >= 2:
            direction = "从左到右" if positions[-1] > positions[0] else "从右到左"
        metrics = {
            "stage": self._stage,
            "hand_grip": hand_grip,
            "moved_from_origin": round(moved, 3),
            "signal_source": interaction.get("signal_source") if interaction else None,
            "target_count": len(self._targets),
            "target_ids": [row[0] for row in self._targets],
            "target_x_positions": positions,
            "direction": direction,
            "returned_to_origin": returned,
            "liquid_verified": False,
        }
        confidence = float(getattr(pitcher, "confidence", 0))
        evidence = _evidence(context, [pitcher, *cups], metrics)
        events: List[ObservationEvent] = []
        if self._completed or self._failed:
            state = ObservationState.COMPLETED if self._completed else ObservationState.FAILED
            return ObservationSnapshot(
                self.observation_id, self.name, self.sop_step, state,
                confidence=confidence, value=direction,
                reason="分茶完成且公道杯已放回" if self._completed else "分茶顺序发生折返",
                started_at=self._started_at, updated_at=context.timestamp,
                metrics=metrics, experimental=True,
            ), []
        if self._stage == "waiting_grip":
            if hand_grip:
                self._stage = "pouring_cups"
                self._started_at = context.timestamp
                events.append(_event(
                    self, context, EventPhase.STARTED, self._started_at,
                    confidence, "手已握住公道杯", metrics, [evidence],
                ))
                reason = "已握住公道杯，请按一个方向依次分茶"
            else:
                reason = "等待手握住公道杯"
        elif self._stage == "pouring_cups":
            if interaction is not None and target_id is not None and not any(row[0] == target_id for row in self._targets):
                if target_id != self._current_target:
                    self._current_target = target_id
                    self._target_since = context.timestamp
                    reason = "已对准当前品茗杯，保持约0.4秒"
                elif self._target_since is not None and context.timestamp - self._target_since >= 0.4:
                    self._targets.append((target_id, float(target_x)))
                    self._current_target = None
                    self._target_since = None
                    if len(self._targets) >= 3:
                        deltas = np.diff([row[1] for row in self._targets])
                        if not (np.all(deltas > -12.0) or np.all(deltas < 12.0)):
                            self._failed = True
                            events.append(_event(
                                self, context, EventPhase.FAILED,
                                self._started_at or context.timestamp, confidence,
                                "分茶顺序折返", metrics, [evidence],
                            ))
                            reason = "三个杯子的分茶方向发生折返"
                        else:
                            self._stage = "waiting_return"
                            reason = "三个品茗杯已完成分茶，请放回公道杯"
                    else:
                        reason = f"已完成{len(self._targets)}/3只品茗杯，继续同方向分茶"
                else:
                    reason = "当前品茗杯已对准，保持约0.4秒"
            else:
                self._current_target = None
                self._target_since = None
                reason = f"已完成{len(self._targets)}/3只，等待对准下一只品茗杯"
        else:
            if returned and interaction is None:
                if self._return_since is None:
                    self._return_since = context.timestamp
                if context.timestamp - self._return_since >= 0.45:
                    self._completed = True
                    events.append(_event(
                        self, context, EventPhase.COMPLETED,
                        self._started_at or context.timestamp, confidence,
                        direction, {**metrics, "stage": "completed"}, [evidence],
                    ))
                    return ObservationSnapshot(
                        self.observation_id, self.name, self.sop_step,
                        ObservationState.COMPLETED, confidence=confidence,
                        value=direction, reason="分茶完成且公道杯已放回原位",
                        started_at=self._started_at, updated_at=context.timestamp,
                        metrics=metrics, experimental=True,
                    ), events
            else:
                self._return_since = None
            reason = "等待公道杯放回开始位置附近"
        return ObservationSnapshot(
            self.observation_id, self.name, self.sop_step,
            ObservationState.ACTIVE if self._started_at is not None else ObservationState.IDLE,
            confidence=confidence, value=direction or self._stage, reason=reason,
            started_at=self._started_at, updated_at=context.timestamp,
            metrics={**metrics, "stage": self._stage}, experimental=True,
        ), events

    def reset(self) -> None:
        self._stage = "waiting_grip"
        self._origin = None
        self._scale = 1.0
        self._started_at = None
        self._current_target = None
        self._target_since = None
        self._targets.clear()
        self._return_since = None
        self._completed = False
        self._failed = False
