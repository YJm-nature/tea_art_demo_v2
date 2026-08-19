import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from src.dataset_rebuild import (
    parse_yolo_labels,
    mark_reviewed_absent_classes,
    prepare_review_workspace,
    publish_reviewed_dataset,
    publish_temporal_prototype_dataset,
    read_manifest,
    set_review_status,
    set_batch_review_status,
    set_second_review_status,
    validate_published_dataset,
    write_manifest,
    write_yolo_labels,
)


PROJECT = Path(__file__).resolve().parents[1]
ONTOLOGY = PROJECT / "config" / "ontology_v1.yaml"


class DatasetRebuildTests(unittest.TestCase):
    def test_temporal_prototype_uses_only_accepted_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            review = root / "review"
            published = root / "prototype"
            for index in range(30):
                split = "val" if index == 29 else "train"
                self._sample(source, split, f"office_auto__office_{index:04d}", index)
            prepare_review_workspace(source, review, ONTOLOGY)
            records = read_manifest(review)
            for index, record in enumerate(records):
                if index == 0:
                    continue
                label = review / record["detect_label"]
                rows = parse_yolo_labels(label, 18)
                rows.extend([(0, 0.5, 0.5, 0.2, 0.2), (1, 0.5, 0.4, 0.1, 0.1)])
                write_yolo_labels(label, rows)
                set_review_status(review, record["sample_id"], "accepted", "tester")

            report = publish_temporal_prototype_dataset(
                review, published, val_ratio=0.2, gap_frames=2
            )
            self.assertTrue(report["prototype_same_session_holdout"])
            self.assertEqual(report["unfinished_samples_not_published"], 1)
            self.assertEqual(report["split_counts"], {"train": 21, "val": 6})
            data_yaml = (published / "data.yaml").read_text(encoding="utf-8")
            self.assertIn("prototype_same_session_holdout: true", data_yaml)
            self.assertNotIn("test: test/images", data_yaml)
            manifest = (published / "manifest.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(records[0]["sample_id"], manifest)

    def test_batch_review_status_updates_only_after_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            review = root / "review"
            self._sample(source, "train", "office_auto__office_0010", 1)
            self._sample(source, "val", "office_auto__office_0020", 2)
            prepare_review_workspace(source, review, ONTOLOGY)
            records = read_manifest(review)
            with self.assertRaises(ValueError):
                set_batch_review_status(
                    review, [record["sample_id"] for record in records], "accepted", "tester"
                )
            self.assertTrue(all(
                record["review_status"] == "pending" for record in read_manifest(review)
            ))

            for record in records:
                label = review / record["detect_label"]
                rows = parse_yolo_labels(label, 18)
                rows.extend([(0, 0.5, 0.5, 0.2, 0.2), (1, 0.5, 0.4, 0.1, 0.1)])
                write_yolo_labels(label, rows)
            report = set_batch_review_status(
                review, [record["sample_id"] for record in records], "accepted", "tester"
            )
            self.assertEqual(report["updated"], 2)
            self.assertTrue(all(
                record["review_status"] == "accepted" for record in read_manifest(review)
            ))

    def test_reviewed_absent_class_can_pass_acceptance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            review = root / "review"
            self._sample(source, "train", "office_auto__office_0010", 1)
            self._sample(source, "val", "office_auto__office_0020", 2)
            prepare_review_workspace(source, review, ONTOLOGY)
            record = read_manifest(review)[0]
            with self.assertRaises(ValueError):
                set_review_status(review, record["sample_id"], "accepted", "tester")
            mark_reviewed_absent_classes(
                review, record["sample_id"], [0, 1], "tester", "not visible"
            )
            accepted = set_review_status(
                review, record["sample_id"], "accepted", "tester"
            )
            self.assertEqual(accepted["reviewed_absent_class_ids"], [0, 1])

    def test_prepare_review_and_publish_keep_sessions_isolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            review = root / "review"
            published = root / "published"
            samples = [
                ("train", "office_auto__office_0010"),
                ("val", "original__frame_0020"),
                ("val", "focus_manual__focus_0030"),
            ]
            for index, (split, stem) in enumerate(samples):
                self._sample(source, split, stem, index)

            summary = prepare_review_workspace(source, review, ONTOLOGY)
            self.assertEqual(summary["total"], 3)
            records = read_manifest(review)
            migrated = parse_yolo_labels(review / records[0]["detect_label"], 18)
            self.assertNotIn(0, {row[0] for row in migrated})
            self.assertIn(2, {row[0] for row in migrated})

            with self.assertRaises(ValueError):
                set_review_status(review, records[0]["sample_id"], "accepted", "tester")

            for record in records:
                label = review / record["detect_label"]
                rows = parse_yolo_labels(label, 18)
                rows.extend([
                    (0, 0.5, 0.5, 0.2, 0.2),
                    (1, 0.5, 0.4, 0.1, 0.1),
                ])
                write_yolo_labels(label, rows)
                set_review_status(review, record["sample_id"], "accepted", "tester")

            for record in read_manifest(review):
                if record["second_review_required"]:
                    with self.assertRaises(ValueError):
                        set_second_review_status(
                            review, record["sample_id"], "accepted", "tester"
                        )
                    set_second_review_status(
                        review, record["sample_id"], "accepted", "second_tester"
                    )

            report = publish_reviewed_dataset(
                review,
                published,
                explicit_assignments={"office": "train", "original": "val", "focus": "test"},
                allow_prototype=True,
            )
            self.assertEqual(report["split_counts"], {"train": 1, "val": 1, "test": 1})
            self.assertTrue((published / "data.yaml").exists())
            self.assertTrue((published / "test" / "images" / "focus_manual__focus_0030.jpg").exists())
            validation = validate_published_dataset(published, min_sessions_per_class=1)
            self.assertFalse(validation["valid"])
            self.assertIn("无实例的当前训练类别", validation["errors"][0])
            self.assertEqual(
                validation["active_class_ids"], [0, 1, 2, 3, 4, 5, 6, 7, 10, 14]
            )
            self.assertEqual(
                validation["deferred_class_ids"], [8, 9, 11, 12, 13, 15, 16, 17]
            )

    def test_prototype_candidate_filters_duplicates_and_deferred_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            review = root / "review"
            published = root / "published"
            samples = [
                ("train", "office_auto__office_0010", 1),
                ("val", "original__frame_0020", 1),
                ("val", "original__frame_0030", 3),
                ("val", "focus_manual__focus_0040", 2),
            ]
            for split, stem, marker in samples:
                self._sample(source, split, stem, marker)
            unique_image_path = source / "val" / "images" / "original__frame_0030.jpg"
            checker = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
            unique_image = np.repeat(checker[:, :, None], 3, axis=2)
            cv2.imwrite(str(unique_image_path), unique_image)

            prepare_review_workspace(source, review, ONTOLOGY)
            records = read_manifest(review)
            for index, record in enumerate(records):
                label = review / record["detect_label"]
                rows = parse_yolo_labels(label, 18)
                rows.extend([(0, 0.5, 0.5, 0.2, 0.2), (1, 0.5, 0.4, 0.1, 0.1)])
                if index == 0:
                    rows.append((8, 0.5, 0.6, 0.3, 0.2))
                write_yolo_labels(label, rows)
                set_review_status(review, record["sample_id"], "accepted", "tester")

            records = read_manifest(review)
            records[0]["second_review_required"] = True
            records[0]["second_review_status"] = "pending"
            write_manifest(review, records)
            with self.assertRaises(ValueError):
                publish_reviewed_dataset(
                    review,
                    published,
                    explicit_assignments={"office": "train", "original": "val", "focus": "test"},
                    allow_prototype=True,
                )

            report = publish_reviewed_dataset(
                review,
                published,
                explicit_assignments={"office": "train", "original": "val", "focus": "test"},
                allow_prototype=True,
                allow_pending_second_review=True,
            )
            self.assertEqual(report["samples"], 3)
            self.assertEqual(report["excluded_duplicate_candidates"], 1)
            self.assertEqual(report["excluded_deferred_instances"], {"茶盘": 1})
            self.assertTrue(report["provisional_pending_second_review"])
            published_rows = []
            for label in published.glob("*/labels/*.txt"):
                published_rows.extend(parse_yolo_labels(label, 18))
            self.assertNotIn(8, {row[0] for row in published_rows})

    @staticmethod
    def _sample(root: Path, split: str, stem: str, marker: int):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image = np.full((64, 64, 3), marker * 8, dtype=np.uint8)
        cv2.imwrite(str(image_dir / f"{stem}.jpg"), image)
        (label_dir / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
