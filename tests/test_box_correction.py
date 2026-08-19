from pathlib import Path
import tempfile
import unittest

from scripts.apply_reviewed_box_correction import apply_correction, learn_correction


class BoxCorrectionTests(unittest.TestCase):
    def test_learns_median_correction_and_applies_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            before = root / "before"
            after = root / "after"
            before.mkdir()
            after.mkdir()
            for index, dx in enumerate((0.01, 0.02, 0.03)):
                stem = f"sample_{index}"
                (before / f"{stem}.txt").write_text(
                    "1 0.500000 0.500000 0.200000 0.100000\n", encoding="utf-8"
                )
                (after / f"{stem}.txt").write_text(
                    f"1 {0.5 + dx:.6f} 0.510000 0.180000 0.110000\n", encoding="utf-8"
                )
            correction = learn_correction(
                before, after, [f"sample_{i}" for i in range(3)], 18, 1, 0.15
            )
            self.assertEqual(correction["dx"], 0.02)
            self.assertEqual(correction["dy"], 0.01)
            self.assertEqual(correction["width_ratio"], 0.9)
            self.assertEqual(correction["height_ratio"], 1.1)
            corrected = apply_correction(
                (1, 0.5, 0.5, 0.2, 0.1), correction
            )
            self.assertAlmostEqual(corrected[1], 0.52)
            self.assertAlmostEqual(corrected[2], 0.51)
            self.assertAlmostEqual(corrected[3], 0.18)
            self.assertAlmostEqual(corrected[4], 0.11)


if __name__ == "__main__":
    unittest.main()
