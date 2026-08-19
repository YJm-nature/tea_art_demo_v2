import unittest

from src.detection_memory import DetectionMemory
from src.tea_detector import DetectedItem


class DetectionMemoryTests(unittest.TestCase):
    def test_item_expires_from_recent_scoring_window(self):
        memory = DetectionMemory()
        item = DetectedItem(
            item_name="盖碗",
            confidence=0.9,
            bbox=(10, 10, 20, 20),
            centroid=(20, 20),
            contour_area=400,
            aspect_ratio=1.0,
            circularity=0.9,
        )
        memory.accumulate([item], frame_idx=1)
        config = {
            "盖碗": {
                "name_cn": "盖碗",
                "essential": True,
                "quantity_range": [1, 1],
            }
        }
        recent = memory.get_checklist(config, current_frame=30, max_age_frames=90)
        stale = memory.get_checklist(config, current_frame=100, max_age_frames=90)
        self.assertTrue(recent["盖碗"]["detected"])
        self.assertFalse(stale["盖碗"]["present"])
        self.assertFalse(stale["盖碗"]["detected"])


if __name__ == "__main__":
    unittest.main()
