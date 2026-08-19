import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.sop_config import build_sop_steps, load_sop_config
from src.sop_replay import load_event_records, replay_sop_events


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "sop_red_tea_v1.yaml"


def completed(observation_id, timestamp, confidence=0.9):
    return {
        "observation_id": observation_id,
        "phase": "completed",
        "end_time": timestamp,
        "confidence": confidence,
    }


CURRENT_EVENTS = [
    completed("action_tea_canister_to_lotus", 5.0),
    completed("action_hold_lotus", 10.0),
    completed("action_open_lid_smell", 15.0),
    completed("result_brew_wait_decant_partial", 20.0),
]


class SopConfigLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_sop_config(CONFIG_PATH)

    def test_default_filter_uses_current_capabilities(self):
        steps = build_sop_steps(self.config)
        self.assertEqual(
            [step.step_id for step in steps],
            ["tea_prepare", "hold_lotus", "smell", "brew_partial"],
        )
        self.assertEqual(steps[0].prerequisites, ())

    def test_include_deferred_builds_full_runtime_chain(self):
        steps = build_sop_steps(self.config, include_deferred=True)
        self.assertEqual(
            [step.step_id for step in steps],
            [
                "warm_clean",
                "tea_prepare",
                "hold_lotus",
                "smell",
                "brew_partial",
                "serve_layout",
            ],
        )
        self.assertEqual(steps[-1].prerequisites, ("brew_partial",))

    def test_include_disabled_and_deferred_covers_all_six_business_steps(self):
        steps = build_sop_steps(
            self.config, include_deferred=True, include_disabled=True
        )
        self.assertEqual(
            [step.step_id for step in steps],
            [
                "setup",
                "warm_clean",
                "tea_prepare",
                "hold_lotus",
                "smell",
                "brew_partial",
                "serve_layout",
            ],
        )
        self.assertEqual(steps[1].prerequisites, ("setup",))

    def test_explicit_observation_filter_removes_missing_prerequisite(self):
        steps = build_sop_steps(
            self.config, available_observation_ids=["action_hold_lotus"]
        )
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].step_id, "hold_lotus")
        self.assertEqual(steps[0].prerequisites, ())


class SopReplayTests(unittest.TestCase):
    def test_correct_strict_sequence_completes(self):
        report = replay_sop_events(
            CURRENT_EVENTS, config_path=CONFIG_PATH, mode="strict"
        )
        self.assertTrue(report["summary"]["is_complete"])
        self.assertEqual(report["summary"]["accepted_record_count"], 4)
        self.assertEqual(report["summary"]["ignored_record_count"], 0)
        self.assertTrue(report["scope"]["current_capabilities_only"])
        self.assertFalse(report["scope"]["formal_acceptance_enabled"])
        self.assertEqual(
            [item["step_id"] for item in report["scope"]["omitted_runtime_nodes"]],
            ["setup", "warm_clean", "serve_layout"],
        )

    def test_out_of_order_strict_event_is_ignored(self):
        report = replay_sop_events(
            [CURRENT_EVENTS[1]], config_path=CONFIG_PATH, mode="strict"
        )
        self.assertFalse(report["summary"]["is_complete"])
        self.assertEqual(report["summary"]["ignored_record_count"], 1)
        transition = report["records"][0]["transitions"][-1]
        self.assertEqual(transition["action"], "ignored")
        self.assertIn("not currently active", transition["reason"])

    def test_low_confidence_enters_manual_review(self):
        report = replay_sop_events(
            [completed("action_tea_canister_to_lotus", 5.0, confidence=0.2)],
            config_path=CONFIG_PATH,
            mode="strict",
        )
        self.assertEqual(report["summary"]["final_status"], "needs_review")
        self.assertEqual(report["summary"]["steps_needing_review"], ["tea_prepare"])
        self.assertEqual(report["summary"]["review_record_count"], 1)

    def test_review_and_skip_control_records(self):
        records = [
            completed("action_tea_canister_to_lotus", 5.0, confidence=0.2),
            {
                "operation": "review",
                "step_id": "tea_prepare",
                "timestamp": 6.0,
                "approved": True,
            },
            {
                "operation": "skip",
                "step_id": "hold_lotus",
                "timestamp": 7.0,
                "reason": "test skip",
                "force": True,
            },
            completed("action_open_lid_smell", 8.0),
            completed("result_brew_wait_decant_partial", 9.0),
        ]
        report = replay_sop_events(records, config_path=CONFIG_PATH, mode="strict")
        self.assertTrue(report["summary"]["is_complete"])
        self.assertEqual(report["summary"]["review_record_count"], 2)

    def test_tick_and_retry_controls_are_recorded(self):
        records = [
            {
                "observation_id": "action_tea_canister_to_lotus",
                "phase": "started",
                "timestamp": 0.0,
            },
            {"operation": "tick", "timestamp": 31.0},
            {"operation": "tick", "timestamp": 62.0},
            {
                "operation": "retry",
                "step_id": "tea_prepare",
                "timestamp": 63.0,
            },
        ]
        report = replay_sop_events(records, config_path=CONFIG_PATH, mode="strict")
        actions = [
            transition["action"]
            for record in report["records"]
            for transition in record["transitions"]
        ]
        self.assertIn("timeout_retry", actions)
        self.assertIn("timeout", actions)
        self.assertIn("retry_rejected", actions)

    def test_json_and_jsonl_loading(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "events.json"
            jsonl_path = root / "events.jsonl"
            json_path.write_text(
                json.dumps({"events": CURRENT_EVENTS}, ensure_ascii=False),
                encoding="utf-8",
            )
            jsonl_path.write_text(
                "\n".join(json.dumps(item) for item in CURRENT_EVENTS) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_event_records(json_path), CURRENT_EVENTS)
            self.assertEqual(load_event_records(jsonl_path), CURRENT_EVENTS)

    def test_sort_events_uses_timestamp_and_preserves_ties(self):
        report = replay_sop_events(
            list(reversed(CURRENT_EVENTS)),
            config_path=CONFIG_PATH,
            mode="strict",
            sort_events=True,
        )
        self.assertTrue(report["summary"]["is_complete"])


if __name__ == "__main__":
    unittest.main()
