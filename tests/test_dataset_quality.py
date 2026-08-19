import tempfile
from pathlib import Path
import unittest

from src.dataset_quality import audit_dataset, infer_source


class DatasetQualityTests(unittest.TestCase):
    def test_source_normalizes_annotation_method(self):
        self.assertEqual(infer_source("office_auto__office_0012"), "office")
        self.assertEqual(infer_source("office_manual__office_0099"), "office")

    def test_audit_finds_source_and_neighbor_leakage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data.yaml").write_text(
                "path: .\ntrain: train/images\nval: val/images\nnc: 1\nnames:\n  0: 盖碗\n",
                encoding="utf-8",
            )
            self._sample(root, "train", "office_auto__office_0010")
            self._sample(root, "val", "office_manual__office_0011")

            report = audit_dataset(root, near_frame_gap=2)
            self.assertEqual(report["summary"]["train"]["instances"], 1)
            self.assertEqual(report["leakage"]["sources_in_both_splits"], ["office"])
            self.assertEqual(report["leakage"]["near_frame_pairs"]["count"], 1)

    @staticmethod
    def _sample(root: Path, split: str, stem: str):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{stem}.jpg").write_bytes(b"image-placeholder")
        (label_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
