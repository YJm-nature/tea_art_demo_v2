import json
from pathlib import Path
import tempfile
import unittest

from scripts.action_dataset import initialize_dataset, validate_dataset


class ActionDatasetTests(unittest.TestCase):
    def test_empty_initialized_workspace_is_valid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = initialize_dataset(Path(temp_dir) / "action_data")
            report = validate_dataset(root)

            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["counts"]["segments"], 0)
            self.assertEqual(report["counts"]["jewelry_rois"], 0)

    def test_validates_segments_and_jewelry_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = initialize_dataset(Path(temp_dir) / "action_data")
            self._write_assignments(root, [("session_a", "train")])
            video = root / "raw_sessions/session_a/side/videos/main.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video-placeholder")

            segment = self._segment("session_a", "train")
            self._write_jsonl(root / "annotations/segments.jsonl", [segment])

            image = root / "jewelry_roi/train/images/session_a_left_0001.jpg"
            label = root / "jewelry_roi/train/labels/session_a_left_0001.txt"
            image.write_bytes(b"image-placeholder")
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            jewelry = self._jewelry("session_a", "train")
            self._write_jsonl(root / "jewelry_roi/manifest.jsonl", [jewelry])

            report = validate_dataset(root, require_media=True)

            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["counts"]["segments"], 1)
            self.assertEqual(report["counts"]["jewelry_class_instances"]["ring"], 1)

    def test_rejects_session_split_leakage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = initialize_dataset(Path(temp_dir) / "action_data")
            self._write_assignments(root, [("session_a", "train")])
            records = [
                self._segment("session_a", "train", segment_id="segment_train"),
                self._segment("session_a", "val", segment_id="segment_val"),
            ]
            self._write_jsonl(root / "annotations/segments.jsonl", records)

            report = validate_dataset(root)

            self.assertFalse(report["valid"])
            self.assertTrue(
                any("session split leakage" in error for error in report["errors"]),
                report["errors"],
            )

    def test_rejects_invalid_yolo_roi_box(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = initialize_dataset(Path(temp_dir) / "action_data")
            self._write_assignments(root, [("session_a", "train")])
            image = root / "jewelry_roi/train/images/session_a_left_0001.jpg"
            label = root / "jewelry_roi/train/labels/session_a_left_0001.txt"
            image.write_bytes(b"image-placeholder")
            label.write_text("3 1.2 0.5 0.2 0.2\n", encoding="utf-8")
            self._write_jsonl(
                root / "jewelry_roi/manifest.jsonl",
                [self._jewelry("session_a", "train")],
            )

            report = validate_dataset(root)

            self.assertFalse(report["valid"])
            self.assertTrue(
                any("class id must be 0..2" in error for error in report["errors"]),
                report["errors"],
            )

    def test_requires_reviewer_for_non_pending_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = initialize_dataset(Path(temp_dir) / "action_data")
            self._write_assignments(root, [("session_a", "train")])
            record = self._segment("session_a", "train")
            record["review_status"] = "accepted"
            self._write_jsonl(root / "annotations/segments.jsonl", [record])

            report = validate_dataset(root)

            self.assertFalse(report["valid"])
            self.assertTrue(
                any("require reviewer and reviewed_at" in error for error in report["errors"]),
                report["errors"],
            )

    def test_rejects_review_datetime_without_timezone(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = initialize_dataset(Path(temp_dir) / "action_data")
            self._write_assignments(root, [("session_a", "train")])
            record = self._segment("session_a", "train")
            record.update({
                "review_status": "accepted",
                "reviewer": "reviewer_a",
                "reviewed_at": "2026-07-30T10:00:00",
            })
            self._write_jsonl(root / "annotations/segments.jsonl", [record])

            report = validate_dataset(root)

            self.assertFalse(report["valid"])
            self.assertTrue(
                any("ISO-8601 datetime" in error for error in report["errors"]),
                report["errors"],
            )

    @staticmethod
    def _segment(
        session_id: str, split: str, segment_id: str = "segment_0001"
    ) -> dict:
        return {
            "schema_version": "1.0",
            "segment_id": segment_id,
            "session_id": session_id,
            "split": split,
            "camera_role": "side",
            "video_path": f"raw_sessions/{session_id}/side/videos/main.mp4",
            "observation_point_id": "action_hold_lotus",
            "start_ms": 1000,
            "end_ms": 2200,
            "sample_kind": "positive",
            "target_utensil": "tea_lotus",
            "review_status": "pending",
            "reviewer": None,
            "reviewed_at": None,
            "review_note": "",
        }

    @staticmethod
    def _jewelry(session_id: str, split: str) -> dict:
        stem = f"{session_id}_left_0001"
        return {
            "schema_version": "1.0",
            "roi_id": stem,
            "session_id": session_id,
            "split": split,
            "image": f"jewelry_roi/{split}/images/{stem}.jpg",
            "label": f"jewelry_roi/{split}/labels/{stem}.txt",
            "source_video": f"raw_sessions/{session_id}/side/videos/main.mp4",
            "frame_ms": 1200,
            "hand_side": "left",
            "sample_kind": "positive",
            "review_status": "pending",
            "reviewer": None,
            "reviewed_at": None,
            "review_note": "",
        }

    @staticmethod
    def _write_assignments(root: Path, rows: list[tuple[str, str]]) -> None:
        body = "session_id,split\n" + "".join(f"{session},{split}\n" for session, split in rows)
        (root / "splits/session_assignments.csv").write_text(body, encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
