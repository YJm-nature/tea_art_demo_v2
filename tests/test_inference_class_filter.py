import unittest

import numpy as np

from src.object_tracker import ByteTrackAdapter
from src.tea_detector import TeaDetector


class FakeYolo:
    def __init__(self):
        self.predict_kwargs = None
        self.track_kwargs = None

    def __call__(self, _frame, **kwargs):
        self.predict_kwargs = kwargs
        return []

    def track(self, _frame, **kwargs):
        self.track_kwargs = kwargs
        return []


class InferenceClassFilterTests(unittest.TestCase):
    def test_detector_passes_active_classes_to_yolo(self):
        model = FakeYolo()
        detector = TeaDetector(
            yolo_model=model,
            class_names=[f"class_{index}" for index in range(18)],
            active_class_ids=[0, 1, 2, 10, 14],
        )
        detector._detect_yolo(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertEqual(model.predict_kwargs["classes"], [0, 1, 2, 10, 14])

    def test_tracker_passes_active_classes_to_yolo(self):
        model = FakeYolo()
        tracker = ByteTrackAdapter(
            model,
            [f"class_{index}" for index in range(18)],
            active_class_ids=[0, 1, 2, 10, 14],
        )
        tracker.track(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertEqual(model.track_kwargs["classes"], [0, 1, 2, 10, 14])


if __name__ == "__main__":
    unittest.main()
