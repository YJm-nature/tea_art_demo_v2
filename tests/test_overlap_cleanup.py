import unittest

from scripts.remove_overlapping_class_boxes import remove_overlaps


class OverlapCleanupTests(unittest.TestCase):
    def test_removes_only_wrong_class_overlapping_reference(self):
        rows = [
            (3, 0.5, 0.5, 0.2, 0.2),
            (10, 0.5, 0.5, 0.19, 0.19),
            (10, 0.8, 0.8, 0.25, 0.25),
        ]
        cleaned, scores = remove_overlaps(rows, wrong_class=10, reference_class=3, min_iou=0.75)
        self.assertEqual(len(scores), 1)
        self.assertEqual(cleaned, [rows[0], rows[2]])


if __name__ == "__main__":
    unittest.main()
