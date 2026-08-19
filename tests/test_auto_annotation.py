import tempfile
from pathlib import Path
import unittest

from src.auto_annotation import (
    Prediction,
    add_missing_legacy_split_boxes,
    apply_candidate_labels,
    box_iou,
    merge_baseline_with_predictions,
)
from src.dataset_rebuild import read_manifest, write_manifest


class AutoAnnotationTests(unittest.TestCase):
    def test_box_iou(self):
        row = (0, 0.5, 0.5, 0.2, 0.2)
        self.assertAlmostEqual(box_iou(row, row), 1.0)
        self.assertEqual(box_iou(row, (0, 0.1, 0.1, 0.1, 0.1)), 0.0)

    def test_merge_keeps_unsupported_and_adds_split_class(self):
        baseline = [
            (2, 0.5, 0.5, 0.2, 0.2),
            (17, 0.8, 0.8, 0.1, 0.1),
        ]
        predictions = [
            Prediction(2, 0.51, 0.5, 0.2, 0.2, 0.8),
            Prediction(0, 0.3, 0.3, 0.15, 0.15, 0.3),
            Prediction(3, 0.2, 0.2, 0.1, 0.1, 0.4),
        ]
        merged, report = merge_baseline_with_predictions(
            baseline, predictions, active_classes={0, 2, 3}
        )
        self.assertEqual({row[0] for row in merged}, {0, 2, 17})
        self.assertEqual(report["refined"], 1)
        self.assertEqual(report["added"], 1)

    def test_add_missing_legacy_split_boxes(self):
        rows = [(0, 0.5, 0.5, 0.2, 0.2)]
        legacy = [(0, 0.5, 0.5, 0.2, 0.2)]
        transforms = {0: (0.0, 0.0, 1.0, 1.0), 1: (0.0, -0.2, 0.7, 0.5)}
        merged, added = add_missing_legacy_split_boxes(rows, legacy, transforms)
        self.assertEqual(added, 1)
        self.assertEqual([row[0] for row in merged], [0, 1])

    def test_apply_preserves_rejected_review_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            candidates = Path(temp_dir) / "candidates"
            detect = workspace / "pool" / "labels" / "detect"
            detect.mkdir(parents=True)
            candidates.mkdir()
            (workspace / "classes.txt").write_text("one\n", encoding="utf-8")
            records = []
            for stem, status in (("keep", "rejected"), ("review", "accepted")):
                (detect / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
                (candidates / f"{stem}.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
                records.append({
                    "sample_id": stem,
                    "review_status": status,
                    "second_review_required": False,
                })
            write_manifest(workspace, records)
            report = apply_candidate_labels(workspace, candidates, "test")
            statuses = {row["sample_id"]: row["review_status"] for row in read_manifest(workspace)}
            self.assertEqual(statuses, {"keep": "rejected", "review": "needs_fix"})
            self.assertEqual(report["preserved_rejected"], 1)


if __name__ == "__main__":
    unittest.main()
