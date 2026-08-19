"""Two-stage hand/wrist accessory detector using MediaPipe ROIs and YOLO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


ACCESSORY_MODEL_CLASSES = ["ring", "bracelet_or_bangle", "watch"]
ACCESSORY_DISPLAY_NAMES = ["戒指", "手链/手镯", "手表"]


class HandAccessoryDetector:
    """Runs a dedicated nano detector only inside expanded hand/wrist crops."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        imgsz: int = 512,
        conf: float = 0.35,
        detect_every_n_frames: int = 5,
        device: str = "0",
    ):
        self.model_path = Path(model_path).resolve() if model_path else None
        self.imgsz = imgsz
        self.conf = conf
        self.detect_every_n_frames = max(1, detect_every_n_frames)
        self.device = device
        self._model = None
        self._loaded = False
        self._last_results: List[Dict[str, Any]] = []
        self.load_error = ""

    @property
    def configured(self) -> bool:
        return self.model_path is not None and self.model_path.is_file()

    def _ensure_loaded(self) -> None:
        if self._loaded or not self.configured:
            return
        self._loaded = True
        try:
            from ultralytics import YOLO

            model = YOLO(str(self.model_path))
            names = model.names
            names_list = [str(names[index]) for index in sorted(names)] if isinstance(names, dict) else list(names)
            if names_list != ACCESSORY_MODEL_CLASSES:
                self.load_error = f"饰品模型类别必须为 {ACCESSORY_MODEL_CLASSES}，实际为 {names_list}"
                return
            self._model = model
        except Exception as exc:
            self.load_error = str(exc)

    @staticmethod
    def _expanded_roi(
        hand: Dict[str, Any], frame_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        frame_h, frame_w = frame_shape
        x, y, w, h = hand.get("bbox", (0, 0, 0, 0))
        margin_x = max(int(w * 0.65), 24)
        margin_y = max(int(h * 0.65), 24)
        x1 = max(0, int(x) - margin_x)
        y1 = max(0, int(y) - margin_y)
        x2 = min(frame_w, int(x + w) + margin_x)
        y2 = min(frame_h, int(y + h) + margin_y)
        return x1, y1, x2, y2

    def detect(
        self,
        frame: np.ndarray,
        hands: List[Dict[str, Any]],
        frame_idx: int,
    ) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        if self._model is None:
            return []
        if frame_idx % self.detect_every_n_frames != 0:
            return list(self._last_results)

        crops = []
        rois = []
        for hand_index, hand in enumerate(hands):
            if float(hand.get("confidence", 0)) < 0.6:
                continue
            x1, y1, x2, y2 = self._expanded_roi(hand, frame.shape[:2])
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            crops.append(frame[y1:y2, x1:x2])
            rois.append((hand_index, x1, y1, x2, y2))
        if not crops:
            self._last_results = []
            return []

        try:
            predictions = self._model.predict(
                crops,
                imgsz=self.imgsz,
                conf=self.conf,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            self.load_error = str(exc)
            return list(self._last_results)

        detections: List[Dict[str, Any]] = []
        for prediction, roi in zip(predictions, rois):
            hand_index, roi_x1, roi_y1, roi_x2, roi_y2 = roi
            boxes = getattr(prediction, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                if not 0 <= class_id < len(ACCESSORY_MODEL_CLASSES):
                    continue
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].cpu().tolist()]
                detections.append({
                    "class_id": class_id,
                    "class_name": ACCESSORY_DISPLAY_NAMES[class_id],
                    "model_class_name": ACCESSORY_MODEL_CLASSES[class_id],
                    "confidence": float(box.conf[0]),
                    "hand_index": hand_index,
                    "bbox": [
                        round(roi_x1 + x1),
                        round(roi_y1 + y1),
                        round(x2 - x1),
                        round(y2 - y1),
                    ],
                    "hand_roi": [roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1],
                })
        self._last_results = detections
        return list(detections)
