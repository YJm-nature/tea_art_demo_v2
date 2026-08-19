"""Display ROI OCR and stable numeric measurements for tea SOP observations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable, Deque, Dict, Iterable, Optional, Sequence
import re

import cv2
import numpy as np


DISPLAY_TO_MEASUREMENT = {
    "水壶显示屏": ("temperature", "celsius", 1.0),
    "电子秤显示屏": ("weight", "grams", 0.15),
}
PLAUSIBLE_RANGES = {
    "temperature": (-20.0, 150.0),
    "weight": (-1.0, 20.0),
}


def parse_numeric_text(text: str) -> Optional[float]:
    """Parse one temperature/weight number from noisy OCR text."""

    normalized = str(text).strip().upper().replace(",", ".")
    normalized = normalized.replace("O", "0").replace("I", "1")
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", normalized)
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


@dataclass
class NumericMeasurement:
    kind: str
    value: Optional[float]
    confidence: float
    stable: bool
    unit: str
    sample_count: int
    reason: str = ""
    bbox: Optional[list[float]] = None
    raw_values: list[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "confidence": round(float(self.confidence), 4),
            "stable": self.stable,
            "unit": self.unit,
            "sample_count": self.sample_count,
            "reason": self.reason,
            "bbox": self.bbox,
            "raw_values": self.raw_values,
        }


class StableNumericReader:
    """Require consecutive, mutually consistent OCR readings."""

    def __init__(self, samples: int = 5, tolerance: float = 1.0):
        self.samples = max(1, int(samples))
        self.tolerance = float(tolerance)
        self._history: Deque[tuple[float, float]] = deque(maxlen=self.samples)

    def update(self, value: Optional[float], confidence: float) -> tuple[Optional[float], float, bool, list[float]]:
        if value is None or not np.isfinite(value):
            self._history.clear()
            return None, 0.0, False, []
        self._history.append((float(value), max(0.0, min(1.0, float(confidence)))))
        values = [row[0] for row in self._history]
        if len(values) < self.samples:
            return None, float(np.mean([row[1] for row in self._history])), False, values
        center = float(median(values))
        spread = max(abs(value - center) for value in values)
        if spread > self.tolerance:
            return None, float(np.mean([row[1] for row in self._history])), False, values
        return center, float(np.mean([row[1] for row in self._history])), True, values

    def reset(self) -> None:
        self._history.clear()


class SevenSegmentRecognizer:
    """Small OpenCV recognizer for high-contrast seven-segment displays."""

    DIGITS = {
        (1, 1, 1, 0, 1, 1, 1): "0",
        (0, 0, 1, 0, 0, 1, 0): "1",
        (1, 0, 1, 1, 1, 0, 1): "2",
        (1, 0, 1, 1, 0, 1, 1): "3",
        (0, 1, 1, 1, 0, 1, 0): "4",
        (1, 1, 0, 1, 0, 1, 1): "5",
        (1, 1, 0, 1, 1, 1, 1): "6",
        (1, 0, 1, 0, 0, 1, 0): "7",
        (1, 1, 1, 1, 1, 1, 1): "8",
        (1, 1, 1, 1, 0, 1, 1): "9",
    }

    def recognize(self, roi: np.ndarray) -> tuple[Optional[str], float]:
        if roi is None or roi.size == 0:
            return None, 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        boxes = []
        dot_boxes = []
        image_h, image_w = binary.shape[:2]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h >= image_h * 0.35 and w >= max(3, image_w * 0.015):
                boxes.append((x, y, w, h))
            elif (
                y >= image_h * 0.55
                and 2 <= h <= image_h * 0.20
                and 2 <= w <= image_w * 0.10
                and cv2.contourArea(contour) >= 3
            ):
                dot_boxes.append((x, y, w, h))
        boxes.sort(key=lambda row: row[0])
        if not boxes:
            return None, 0.0
        digits: list[str] = []
        digit_boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []
        for x, y, w, h in boxes:
            symbol, confidence = self._digit(binary[y:y + h, x:x + w])
            if symbol is not None:
                digits.append(symbol)
                digit_boxes.append((x, y, w, h))
                confidences.append(confidence)
        if not digits:
            return None, 0.0
        if dot_boxes and len(digits) >= 2:
            dot_x = max(dot_boxes, key=lambda row: row[1] + row[3])[0]
            insertion = sum(1 for x, _, w, _ in digit_boxes if x + w <= dot_x)
            if 0 < insertion < len(digits):
                digits.insert(insertion, ".")
        return "".join(digits), float(np.mean(confidences))

    def _digit(self, image: np.ndarray) -> tuple[Optional[str], float]:
        h, w = image.shape[:2]
        if h < 12 or w < 4:
            return None, 0.0
        dw = max(1, int(w * 0.22))
        dh = max(1, int(h * 0.12))
        cy = h // 2
        segments = (
            (dw, 0, w - dw, dh),
            (0, dh, dw, cy),
            (w - dw, dh, w, cy),
            (dw, cy - dh // 2, w - dw, cy + dh // 2 + 1),
            (0, cy, dw, h - dh),
            (w - dw, cy, w, h - dh),
            (dw, h - dh, w - dw, h),
        )
        ratios = []
        for x1, y1, x2, y2 in segments:
            crop = image[y1:y2, x1:x2]
            ratios.append(float(np.count_nonzero(crop)) / max(1, crop.size))
        active = tuple(1 if value >= 0.35 else 0 for value in ratios)
        symbol = self.DIGITS.get(active)
        if symbol is None:
            return None, 0.0
        confidence = float(np.mean([value if flag else 1.0 - value for value, flag in zip(ratios, active)]))
        return symbol, confidence


class DisplayOcrService:
    """Crop display boxes from the original frame and publish stable readings."""

    def __init__(
        self,
        backend: Optional[Callable[[np.ndarray], tuple[Optional[str], float]]] = None,
        *,
        stable_samples: int = 5,
        enable_paddle_fallback: bool = False,
    ):
        self.seven_segment = SevenSegmentRecognizer()
        self.backend = backend
        self.enable_paddle_fallback = enable_paddle_fallback
        self._paddle = None
        self.readers = {
            kind: StableNumericReader(stable_samples, tolerance)
            for _, (kind, _, tolerance) in DISPLAY_TO_MEASUREMENT.items()
        }

    def process(
        self,
        source_frame: np.ndarray,
        detections: Iterable[Any],
        inference_shape: Sequence[int],
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        selected: Dict[str, Any] = {}
        for item in detections:
            name = str(getattr(item, "item_name", ""))
            if name not in DISPLAY_TO_MEASUREMENT:
                continue
            previous = selected.get(name)
            if previous is None or float(getattr(item, "confidence", 0.0)) > float(getattr(previous, "confidence", 0.0)):
                selected[name] = item
        observed_kinds = {DISPLAY_TO_MEASUREMENT[name][0] for name in selected}
        for kind, reader in self.readers.items():
            if kind not in observed_kinds:
                reader.reset()
        for name, item in selected.items():
            kind, unit, _ = DISPLAY_TO_MEASUREMENT[name]
            bbox = self._scale_bbox(getattr(item, "bbox"), inference_shape, source_frame.shape[:2])
            roi = self._rectify(self._crop(source_frame, bbox))
            text, confidence = self.seven_segment.recognize(roi)
            value = parse_numeric_text(text or "")
            plausible = PLAUSIBLE_RANGES[kind]
            if value is not None and not plausible[0] <= value <= plausible[1]:
                value = None
            if value is None and self.backend is not None:
                text, confidence = self.backend(self._enhance(roi))
                value = parse_numeric_text(text or "")
                if value is not None and not plausible[0] <= value <= plausible[1]:
                    value = None
            if value is None and self.enable_paddle_fallback:
                text, confidence = self._paddle_read(self._enhance(roi))
                value = parse_numeric_text(text or "")
                if value is not None and not plausible[0] <= value <= plausible[1]:
                    value = None
            stable_value, stable_conf, stable, raw_values = self.readers[kind].update(value, confidence)
            reason = "读数稳定" if stable else (
                "显示屏不可读" if value is None else "等待连续5次稳定读数"
            )
            measurement = NumericMeasurement(
                kind=kind,
                value=stable_value,
                confidence=stable_conf,
                stable=stable,
                unit=unit,
                sample_count=len(raw_values),
                reason=reason,
                bbox=[float(v) for v in bbox],
                raw_values=[round(v, 3) for v in raw_values],
            )
            results[kind] = measurement.to_dict()
        return results

    @staticmethod
    def _scale_bbox(bbox: Sequence[float], inference_shape: Sequence[int], source_shape: Sequence[int]) -> tuple[int, int, int, int]:
        x, y, w, h = [float(value) for value in bbox]
        inf_h, inf_w = int(inference_shape[0]), int(inference_shape[1])
        src_h, src_w = int(source_shape[0]), int(source_shape[1])
        return (
            int(round(x * src_w / max(1, inf_w))),
            int(round(y * src_h / max(1, inf_h))),
            int(round(w * src_w / max(1, inf_w))),
            int(round(h * src_h / max(1, inf_h))),
        )

    @staticmethod
    def _crop(frame: np.ndarray, bbox: Sequence[int]) -> np.ndarray:
        x, y, w, h = bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _enhance(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    @staticmethod
    def _rectify(roi: np.ndarray) -> np.ndarray:
        """Conservatively deskew the largest display-like quadrilateral."""
        if roi is None or roi.size == 0 or min(roi.shape[:2]) < 12:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        edges = cv2.Canny(gray, 50, 150)
        contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not contours:
            return roi
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < roi.shape[0] * roi.shape[1] * 0.20:
            return roi
        box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
        sums, differences = box.sum(axis=1), np.diff(box, axis=1).ravel()
        ordered = np.asarray([
            box[np.argmin(sums)], box[np.argmin(differences)],
            box[np.argmax(sums)], box[np.argmax(differences)],
        ], dtype=np.float32)
        tl, tr, br, bl = ordered
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if width < 8 or height < 8:
            return roi
        destination = np.asarray(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(ordered, destination)
        return cv2.warpPerspective(roi, transform, (width, height))

    def _paddle_read(self, roi: np.ndarray) -> tuple[Optional[str], float]:
        try:
            if self._paddle is None:
                from paddleocr import PaddleOCR
                self._paddle = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
            output = self._paddle.ocr(roi, cls=False)
            candidates = []
            for group in output or []:
                for row in group or []:
                    if len(row) >= 2 and len(row[1]) >= 2:
                        candidates.append((str(row[1][0]), float(row[1][1])))
            return max(candidates, key=lambda row: row[1]) if candidates else (None, 0.0)
        except Exception:
            return None, 0.0

    def reset(self) -> None:
        for reader in self.readers.values():
            reader.reset()
