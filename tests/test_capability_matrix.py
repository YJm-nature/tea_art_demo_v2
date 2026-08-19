from collections import Counter
from pathlib import Path
import unittest

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "config" / "observation_capability_matrix.yaml"


class CapabilityMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))

    def test_detection_ontology_has_fixed_eighteen_classes(self):
        classes = self.matrix["detection_classes"]
        self.assertEqual(len(classes), 18)
        self.assertEqual([item["id"] for item in classes], list(range(18)))
        names = [item["name"] for item in classes]
        self.assertEqual(len(names), len(set(names)))

    def test_requirement_observations_cover_o01_through_o19(self):
        observations = self.matrix["requirement_observations"]
        expected = [f"O{number:02d}" for number in range(1, 20)]
        self.assertEqual([item["id"] for item in observations], expected)

    def test_runtime_observation_ids_are_unique(self):
        runtime = self.matrix["runtime_observations"]
        identifiers = [item["id"] for item in runtime]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for item in runtime:
            self.assertTrue((PROJECT_ROOT / item["code"]).is_file(), item["code"])

    def test_summary_counts_match_requirement_statuses(self):
        observations = self.matrix["requirement_observations"]
        status_counts = Counter(item["status"] for item in observations)
        summary = self.matrix["summary"]
        self.assertEqual(summary["requirement_observation_count"], len(observations))
        self.assertEqual(
            summary["available_or_partial_count"],
            status_counts["available"] + status_counts["partial"],
        )
        self.assertEqual(
            summary["experimental_count"], status_counts["experimental"]
        )
        self.assertEqual(
            summary["deferred_or_excluded_count"],
            status_counts["deferred"]
            + status_counts["deferred_optional"]
            + status_counts["excluded_current_scope"],
        )
        self.assertEqual(
            summary["formal_scoring_ready_count"],
            sum(bool(item["formal_scoring"]) for item in observations),
        )


if __name__ == "__main__":
    unittest.main()
