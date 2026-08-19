"""
目标跟踪模块 — Ultralytics ByteTrack 封装。
"""

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .tea_detector import DetectedItem


class ByteTrackAdapter:
    """将 YOLO track/detect 输出转换为项目统一的 DetectedItem。"""

    def __init__(
        self,
        yolo_model: object,
        class_names: List[str],
        tracker_config: str = "bytetrack.yaml",
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 832,
        active_class_ids: Optional[List[int]] = None,
    ):
        self.yolo_model = yolo_model
        self.class_names = class_names
        self.tracker_config = tracker_config
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.active_class_ids = (
            sorted({int(value) for value in active_class_ids})
            if active_class_ids is not None else None
        )

    def track(self, frame: np.ndarray) -> List[DetectedItem]:
        """执行 ByteTrack 检测 + 关联，返回带 track_id 的检测结果。"""
        track_args = dict(
            conf=self.conf,
            iou=self.iou,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
            imgsz=self.imgsz,
        )
        if self.active_class_ids is not None:
            track_args["classes"] = self.active_class_ids
        results = self.yolo_model.track(frame, **track_args)
        return self._results_to_items(results, source="track")

    def detect(self, frame: np.ndarray) -> List[DetectedItem]:
        """执行普通 YOLO 检测，返回无 track_id 的检测结果。"""
        predict_args = dict(
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            imgsz=self.imgsz,
        )
        if self.active_class_ids is not None:
            predict_args["classes"] = self.active_class_ids
        results = self.yolo_model(frame, **predict_args)
        return self._results_to_items(results, source="yolo")

    def reset(self):
        """尽量重置 Ultralytics 内部 tracker 状态。"""
        if hasattr(self.yolo_model, "predictor") and self.yolo_model.predictor is not None:
            if hasattr(self.yolo_model.predictor, "trackers"):
                self.yolo_model.predictor.trackers = []

    def set_thresholds(self, conf: Optional[float] = None, iou: Optional[float] = None):
        if conf is not None:
            self.conf = conf
        if iou is not None:
            self.iou = iou

    def _results_to_items(self, results, source: str) -> List[DetectedItem]:
        items: List[DetectedItem] = []
        if not results:
            return items

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return items

        frame_h, frame_w = result.orig_shape if hasattr(result, "orig_shape") else (0, 0)
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            if not 0 <= cls_id < len(self.class_names):
                continue
            if self.active_class_ids is not None and cls_id not in self.active_class_ids:
                continue

            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy[:4])
            bw, bh = x2 - x1, y2 - y1
            area = bw * bh

            if frame_h and frame_w:
                if area < 200 or area > frame_w * frame_h * 0.7:
                    continue
                at_edge = (x1 <= 2 or y1 <= 2 or x2 >= frame_w - 2 or y2 >= frame_h - 2)
                if at_edge and area > 50000:
                    continue

            track_id = None
            if getattr(boxes, "id", None) is not None and i < len(boxes.id):
                track_id = int(boxes.id[i])

            items.append(DetectedItem(
                bbox=(x1, y1, bw, bh),
                contour_area=area,
                centroid=((x1 + x2) / 2, (y1 + y2) / 2),
                aspect_ratio=bw / max(bh, 1),
                confidence=conf,
                item_name=self.class_names[cls_id],
                class_id=cls_id,
                track_id=track_id,
                source=source,
            ))

        items.sort(key=lambda it: it.contour_area, reverse=True)
        return items


class TrajectoryStore:
    """维护 track_id 到中心点轨迹的映射。"""

    def __init__(self, max_trail: int = 30, stale_after: int = 90):
        self.max_trail = max_trail
        self.stale_after = stale_after
        self._trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max_trail))
        self._last_seen: Dict[int, int] = {}

    def update(self, items: Iterable[DetectedItem], frame_idx: int = 0):
        """用当前帧检测结果更新轨迹。"""
        active_ids = set()
        for item in items:
            if item.track_id is None:
                continue
            cx, cy = item.centroid
            tid = int(item.track_id)
            self._trails[tid].append((int(cx), int(cy)))
            self._last_seen[tid] = frame_idx
            active_ids.add(tid)

        if frame_idx > 0 and self.stale_after > 0:
            stale_ids = [
                tid for tid, last_seen in self._last_seen.items()
                if frame_idx - last_seen > self.stale_after
            ]
            for tid in stale_ids:
                self._trails.pop(tid, None)
                self._last_seen.pop(tid, None)

    def clear(self):
        self._trails.clear()
        self._last_seen.clear()

    def get_trails(self) -> Dict[int, List[Tuple[int, int]]]:
        return {tid: list(trail) for tid, trail in self._trails.items()}

    def __len__(self) -> int:
        return len(self._trails)
