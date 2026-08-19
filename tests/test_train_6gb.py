import tempfile
from pathlib import Path
import unittest

import yaml

from scripts.train_6gb import _validate_data_yaml


class Train6GBTests(unittest.TestCase):
    def test_prototype_can_train_without_claiming_an_independent_test(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_yaml = self._write_data(
                root, deferred=[8, 9, 11, 12, 13, 15, 16, 17]
            )
            data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
            data.pop("test")
            data["prototype_same_session_holdout"] = True
            data_yaml.write_text(
                yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
            )
            _validate_data_yaml(data_yaml, require_test=True)

    def test_accepts_current_phase_with_fixed_18_class_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_yaml = self._write_data(Path(temp_dir), deferred=[8, 9, 11, 12, 13, 15, 16, 17])
            _validate_data_yaml(data_yaml, require_test=True)

    def test_rejects_overlapping_phase_class_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_yaml = self._write_data(
                Path(temp_dir), deferred=[7, 8, 9, 11, 12, 13, 15, 16, 17]
            )
            with self.assertRaisesRegex(ValueError, "不能重叠"):
                _validate_data_yaml(data_yaml, require_test=True)

    @staticmethod
    def _write_data(root: Path, deferred: list[int]) -> Path:
        path = root / "data.yaml"
        path.write_text(yaml.safe_dump({
            "path": str(root),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": {index: f"class_{index}" for index in range(18)},
            "active_class_ids": [0, 1, 2, 3, 4, 5, 6, 7, 10, 14],
            "deferred_class_ids": deferred,
        }), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
