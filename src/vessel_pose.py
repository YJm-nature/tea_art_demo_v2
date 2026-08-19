"""YOLOv8 pose adapter and common front-view pouring geometry."""

from __future__ import annotations

from collections import defaultdict, deque
from math import atan2, degrees, hypot
from pathlib import Path
from statistics import median
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence

import numpy as np
import yaml


SOURCE_TARGETS = {
    "烧水壶": ("盖碗碗身",),
    "盖碗碗身": ("公道杯", "建水"),
    "公道杯": ("品茗杯", "建水"),
    "茶荷": ("盖碗碗身",),
}


POSE_CLASS_ALIASES = {
    "kettle": "烧水壶",
    "pitcher": "公道杯",
    "gaiwan_body": "盖碗碗身",
    "tea_lotus": "茶荷",
    "盖碗（碗身）": "盖碗碗身",
}

def _center(item: Any) -> tuple[float, float]:
    center = getattr(item, "centroid", None)
    if center is not None:
        return float(center[0]), float(center[1])
    x, y, w, h = getattr(item, "bbox")
    return float(x + w / 2), float(y + h / 2)


def _point_in_expanded_bbox(point: Sequence[float], bbox: Sequence[float], ratio: float) -> bool:
    x, y, w, h = [float(value) for value in bbox]
    px, py = float(point[0]), float(point[1])
    return x - w * ratio <= px <= x + w * (1 + ratio) and y - h * ratio <= py <= y + h * (1 + ratio)


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    """IoU for the project's xywh boxes."""
    lx, ly, lw, lh = [float(value) for value in left]
    rx, ry, rw, rh = [float(value) for value in right]
    lx2, ly2 = lx + max(0.0, lw), ly + max(0.0, lh)
    rx2, ry2 = rx + max(0.0, rw), ry + max(0.0, rh)
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, lw) * max(0.0, lh) + max(0.0, rw) * max(0.0, rh) - intersection
    return intersection / union if union > 0.0 else 0.0


class YoloV8PoseDetector:
    """Run a separately trained YOLOv8n-pose model in the same process."""

    def __init__(
        self,
        model_path: Optional[str | Path],
        conf: float = 0.45,
        imgsz: int = 640,
        require_detector_match: bool = True,
    ):
        self.model_path = Path(model_path).resolve() if model_path else None
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.require_detector_match = bool(require_detector_match)
        self.model = None
        self.configured = False
        if self.model_path is not None:
            if not self.model_path.exists():
                raise FileNotFoundError(self.model_path)
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self.configured = True

    def detect(
        self,
        frame: np.ndarray,
        reference_detections: Optional[Iterable[Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.model is None:
            return []
        predictions = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=0, verbose=False)
        rows: List[Dict[str, Any]] = []
        for result in predictions:
            if result.boxes is None or result.keypoints is None:
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            box_conf = result.boxes.conf.detach().cpu().numpy()
            xy = result.keypoints.xy.detach().cpu().numpy()
            kp_conf = result.keypoints.conf
            kp_conf_array = kp_conf.detach().cpu().numpy() if kp_conf is not None else np.ones(xy.shape[:2])
            names = result.names
            for index, class_id in enumerate(classes):
                x1, y1, x2, y2 = boxes[index]
                raw_class_name = str(names[int(class_id)])
                rows.append({
                    "class_name": POSE_CLASS_ALIASES.get(
                        raw_class_name, raw_class_name
                    ),
                    "raw_class_name": raw_class_name,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "confidence": float(box_conf[index]),
                    "keypoints": xy[index].astype(float).tolist(),
                    "keypoint_confidences": kp_conf_array[index].astype(float).tolist(),
                })
        if not self.require_detector_match or reference_detections is None:
            return rows

        # A pose-only model can hallucinate vessel keypoints on visually similar
        # white cups. Keep a pose instance only when the main detector agrees on
        # the same fine-grained class and overlapping image region.
        references = list(reference_detections)
        filtered: List[Dict[str, Any]] = []
        for row in rows:
            candidates = [
                item for item in references
                if str(getattr(item, "item_name", "")) == row["class_name"]
                and float(getattr(item, "confidence", 0.0)) >= 0.20
            ]
            if not candidates:
                continue
            pose_box = row["bbox"]
            best = max(
                candidates,
                key=lambda item: _bbox_iou(pose_box, getattr(item, "bbox", (0, 0, 0, 0))),
            )
            detector_box = getattr(best, "bbox", (0, 0, 0, 0))
            iou = _bbox_iou(pose_box, detector_box)
            pose_center = (pose_box[0] + pose_box[2] / 2.0, pose_box[1] + pose_box[3] / 2.0)
            if iou < 0.05 and not _point_in_expanded_bbox(pose_center, detector_box, 0.35):
                continue
            row["detector_match_iou"] = round(iou, 4)
            row["detector_match_confidence"] = float(getattr(best, "confidence", 0.0))
            filtered.append(row)
        return filtered

class PourInteractionAnalyzer:
    """Convert vessel keypoints into conservative source-to-target signals."""

    def __init__(self, config_path: Optional[str | Path] = None):
        config_file = Path(config_path) if config_path else Path(__file__).resolve().parents[1] / "config" / "vessel_keypoints_v1.yaml"
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        self.min_keypoint_confidence = float(config.get("minimum_keypoint_confidence", 0.5))
        self.min_tilt_degrees = float(config.get("minimum_tilt_change_degrees", 15.0))
        self.target_expansion = float(config.get("target_expansion_ratio", 0.5))
        self._baseline_angles: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=20))
        self._baseline_y: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=20))

    def update(
        self,
        detections: Iterable[Any],
        hands: Iterable[Dict[str, Any]],
        pose_instances: Iterable[Dict[str, Any]],
        timestamp: float,
    ) -> List[Dict[str, Any]]:
        detection_rows = list(detections)
        hand_rows = [
            row for row in hands
            if float(row.get("confidence", 1.0)) >= 0.4
        ]
        interactions: List[Dict[str, Any]] = []
        for pose in pose_instances:
            source_name = str(pose.get("class_name", ""))
            if source_name not in SOURCE_TARGETS:
                continue
            keypoints = np.asarray(pose.get("keypoints", []), dtype=float)
            confidences = np.asarray(pose.get("keypoint_confidences", []), dtype=float)
            if keypoints.shape != (3, 2) or len(confidences) != 3 or float(confidences.min()) < self.min_keypoint_confidence:
                continue
            bbox = pose.get("bbox", [0, 0, 0, 0])
            endpoint_a, center, endpoint_b = keypoints
            targets = [
                item for item in detection_rows
                if str(getattr(item, "item_name", "")) in SOURCE_TARGETS[source_name]
            ]
            if source_name == "茶荷" and targets:
                def endpoint_target_distance(point: np.ndarray) -> float:
                    return min(
                        hypot(_center(item)[0] - point[0], _center(item)[1] - point[1])
                        for item in targets
                    )
                if endpoint_target_distance(endpoint_b) < endpoint_target_distance(endpoint_a):
                    outlet, rear = endpoint_b, endpoint_a
                else:
                    outlet, rear = endpoint_a, endpoint_b
            else:
                outlet, rear = endpoint_a, endpoint_b
            angle = degrees(atan2(outlet[1] - rear[1], outlet[0] - rear[0]))
            hand_gaps = []
            bx, by, bw, bh = [float(value) for value in bbox]
            source_diag = max(hypot(bw, bh), 1.0)
            for hand in hand_rows:
                hx, hy, hw, hh = [
                    float(value) for value in hand.get("bbox", (0, 0, 0, 0))
                ]
                dx = max(bx - (hx + hw), hx - (bx + bw), 0.0)
                dy = max(by - (hy + hh), hy - (by + bh), 0.0)
                hand_gaps.append(hypot(dx, dy) / source_diag)
            hand_centers = [
                tuple(map(float, hand.get("center", ())))
                for hand in hand_rows
                if len(hand.get("center", ())) >= 2
            ]
            hand_contact = (
                (bool(hand_gaps) and min(hand_gaps) <= 0.60)
                or any(
                    _point_in_expanded_bbox(point, bbox, 0.45)
                    for point in hand_centers
                )
            )
            explicit_delta = pose.get("tilt_delta_degrees")
            if explicit_delta is None and self._baseline_angles[source_name]:
                baseline_angle = float(median(self._baseline_angles[source_name]))
                delta = abs((angle - baseline_angle + 180.0) % 360.0 - 180.0)
            else:
                delta = float(explicit_delta or 0.0)
            source_height = max(1.0, float(bbox[3]))
            explicit_lifted = pose.get("lifted")
            if explicit_lifted is None and self._baseline_y[source_name]:
                lifted = float(median(self._baseline_y[source_name])) - float(center[1]) >= source_height * 0.15
            else:
                lifted = bool(explicit_lifted)

            aligned = [item for item in targets if _point_in_expanded_bbox(outlet, getattr(item, "bbox"), self.target_expansion)]
            if not hand_contact and not aligned:
                self._baseline_angles[source_name].append(angle)
                self._baseline_y[source_name].append(float(center[1]))
                continue
            endpoint_slope = abs(float(outlet[1] - rear[1])) / max(
                abs(float(outlet[0] - rear[0])), 1.0
            )
            pose_tilt_valid = (
                delta >= min(self.min_tilt_degrees, 8.0)
                or endpoint_slope >= 0.12
            )
            # For the prototype, alignment of the learned outlet with the target
            # plus a nearby hand is sufficient. A resting-height baseline is
            # useful evidence but no longer mandatory because many clips begin
            # after the vessel has already been picked up.
            lift_not_explicitly_rejected = explicit_lifted is not False
            if not (
                hand_contact and pose_tilt_valid and aligned
                and lift_not_explicitly_rejected
            ):
                continue
            target = min(aligned, key=lambda item: hypot(_center(item)[0] - outlet[0], _center(item)[1] - outlet[1]))
            confidence = min(
                float(pose.get("confidence", 0.0)),
                float(confidences.min()),
                float(getattr(target, "confidence", 1.0)),
            )
            interactions.append({
                "source": source_name,
                "target": str(getattr(target, "item_name", "")),
                "source_track_id": pose.get("track_id"),
                "target_track_id": getattr(target, "track_id", None),
                "confidence": confidence,
                "signal_source": "yolov8_pose_geometry",
                "timestamp": float(timestamp),
                "outlet_point": [float(outlet[0]), float(outlet[1])],
                "endpoint_selection": (
                    "nearest_target_endpoint" if source_name == "茶荷"
                    else "fixed_first_keypoint"
                ),
                "source_center": [float(center[0]), float(center[1])],
                "target_center": [float(value) for value in _center(target)],
                "tilt_delta_degrees": round(delta, 3),
                "endpoint_slope": round(endpoint_slope, 3),
                "pose_tilt_valid": pose_tilt_valid,
                "lifted": bool(lifted),
                "lift_baseline_required": False,
                "hand_contact": True,
                "minimum_hand_gap": round(min(hand_gaps), 3) if hand_gaps else None,
                "liquid_verified": False,
            })
        return interactions

    def reset(self) -> None:
        self._baseline_angles.clear()
        self._baseline_y.clear()
