import tempfile
from pathlib import Path
import unittest

from src.scoring import ScoringEngine


class PreparationScoringTests(unittest.TestCase):
    def setUp(self):
        self.supported = [
            "盖碗", "公道杯", "品茗杯", "茶荷", "茶巾",
            "茶夹", "茶拨", "茶叶罐", "建水",
        ]
        self.checklist = {
            name: {
                "detected": True,
                "count": 3 if name == "品茗杯" else 1,
                "confidence": 0.9,
                "seen_frames": 15,
            }
            for name in self.supported
        }

    def test_full_observable_score_discloses_partial_requirement_coverage(self):
        report = ScoringEngine.evaluate_preparation_step(
            self.checklist, self.supported, placement_score=1.0
        )
        self.assertEqual(report.score, 100.0)
        self.assertEqual(report.requirement_coverage, 0.8)
        self.assertEqual(report.coverage_adjusted_score, 80.0)
        self.assertEqual(report.evidence_reliability, 90.0)
        self.assertEqual(report.score_status, "provisional")
        self.assertEqual(report.unsupported_requirements, ["茶盘", "烧水壶"])

    def test_missing_item_reduces_performance_and_reliability(self):
        self.checklist["茶夹"].update(detected=False, count=0, confidence=0, seen_frames=0)
        report = ScoringEngine.evaluate_preparation_step(
            self.checklist, self.supported, placement_score=1.0
        )
        self.assertEqual(report.score, 88.8)
        self.assertLess(report.evidence_reliability, 90.0)

    def test_missing_placement_dimension_is_renormalized(self):
        report = ScoringEngine.evaluate_preparation_step(
            self.checklist, self.supported, placement_score=None
        )
        self.assertEqual(report.score, 100.0)
        self.assertIsNone(report.dimension_scores["placement_heuristic"])

    def test_report_can_be_serialized(self):
        report = ScoringEngine.evaluate_preparation_step(
            self.checklist, self.supported, placement_score=0.5
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = report.save_json(str(Path(temp_dir) / "report.json"))
            self.assertTrue(output.exists())
            self.assertIn("schema_version", output.read_text(encoding="utf-8"))

    def test_stale_item_does_not_count_as_present(self):
        self.checklist["茶夹"].update(present=False, detected=False)
        report = ScoringEngine.evaluate_preparation_step(
            self.checklist, self.supported, placement_score=1.0
        )
        self.assertEqual(report.detected_essential, 7)
        self.assertEqual(report.dimension_scores["object_completeness"], 87.5)


if __name__ == "__main__":
    unittest.main()
