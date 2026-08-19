import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile, ZIP_DEFLATED

from scripts.apply_makesense_batch import main as apply_main


class MakeSenseBatchTests(unittest.TestCase):
    def test_import_validates_batch_and_preserves_needs_fix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            detect = workspace / "pool" / "labels" / "detect"
            detect.mkdir(parents=True)
            (workspace / "classes.txt").write_text("one\n", encoding="utf-8")
            (detect / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (workspace / "manifest.jsonl").write_text(json.dumps({
                "sample_id": "sample",
                "detect_label": "pool/labels/detect/sample.txt",
                "review_status": "needs_fix",
                "second_review_required": False,
            }) + "\n", encoding="utf-8")
            batch = root / "batch.json"
            batch.write_text(json.dumps({"sample_ids": ["sample"]}), encoding="utf-8")
            archive = root / "labels.zip"
            with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
                bundle.writestr("sample.txt", "0 0.6 0.5 0.3 0.2\n")
                bundle.writestr("labels.txt", "one\n")

            import sys
            old_argv = sys.argv
            sys.argv = ["apply_makesense_batch.py", str(workspace), str(batch), str(archive), "--run-name", "test_run"]
            try:
                self.assertEqual(apply_main(), 0)
            finally:
                sys.argv = old_argv
            self.assertIn("0.600000", (detect / "sample.txt").read_text(encoding="utf-8"))
            record = json.loads((workspace / "manifest.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["review_status"], "needs_fix")
            self.assertTrue((workspace / "pool/labels/detect_before_test_run/sample.txt").exists())


if __name__ == "__main__":
    unittest.main()
