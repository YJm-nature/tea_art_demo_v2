from pathlib import Path
import unittest

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "sop_red_tea_v1.yaml"


class RedTeaSopConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_business_steps_are_complete_and_ordered(self):
        steps = self.config["steps"]
        self.assertEqual(len(steps), 6)
        self.assertEqual([step["order"] for step in steps], list(range(1, 7)))
        step_ids = [step["id"] for step in steps]
        self.assertEqual(len(step_ids), len(set(step_ids)))

    def test_runtime_node_references_are_valid(self):
        step_ids = {step["id"] for step in self.config["steps"]}
        nodes = self.config["runtime_nodes"]
        node_ids = {node["node_id"] for node in nodes}
        self.assertEqual(len(node_ids), len(nodes))
        for node in nodes:
            self.assertIn(node["business_step"], step_ids)
            for prerequisite in node.get("prerequisites", []):
                self.assertIn(prerequisite, node_ids)

    def test_current_scope_keeps_unavailable_features_deferred(self):
        steps = {step["id"]: step for step in self.config["steps"]}
        self.assertEqual(steps["step02_warm_clean"]["implementation_status"], "deferred")
        self.assertEqual(steps["step06_serve"]["implementation_status"], "deferred")
        excluded = steps["step06_serve"]["excluded_observations"]
        self.assertIn("action_hold_tray", excluded)
        self.assertIn("action_bow_serve", excluded)
        self.assertFalse(self.config["formal_acceptance"]["enabled"])


if __name__ == "__main__":
    unittest.main()
