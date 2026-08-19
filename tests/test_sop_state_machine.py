from dataclasses import dataclass
import json
import tempfile
from pathlib import Path
import unittest

from src.sop_state_machine import (
    SopMode,
    SopStateMachine,
    SopStepConfig,
    StepStatus,
)


@dataclass
class ObjectEvent:
    observation_id: str
    event_type: str
    timestamp: float
    confidence: float = 1.0
    reason: str = ""


def configs():
    return [
        SopStepConfig(
            "layout", "cup_layout", timeout_seconds=2.0, max_retries=1,
            skippable=True,
        ),
        SopStepConfig(
            "hold", "two_hand_hold", prerequisites=("layout",),
            timeout_seconds=3.0,
        ),
        SopStepConfig(
            "smell", "open_lid_smell", prerequisites=("hold",),
            min_confidence=0.8,
        ),
    ]


class SopStateMachineTests(unittest.TestCase):
    def test_free_mode_accepts_out_of_order_dict_and_object_events(self):
        machine = SopStateMachine(configs(), mode="free_observation")
        result = machine.process_event({
            "observation_id": "open_lid_smell",
            "event_type": "completed",
            "timestamp": 5.0,
            "confidence": 0.9,
        })
        self.assertTrue(result)
        self.assertEqual(machine.get_step_state("smell").status, StepStatus.COMPLETED)

        machine.process_event(ObjectEvent("cup_layout", "started", 6.0))
        machine.process_event(ObjectEvent("cup_layout", "completed", 7.0))
        self.assertEqual(machine.get_step_state("layout").status, StepStatus.COMPLETED)
        self.assertFalse(machine.is_complete)

    def test_strict_mode_rejects_later_event_and_unlocks_in_order(self):
        machine = SopStateMachine(configs(), mode=SopMode.STRICT)
        self.assertEqual(machine.current_step_id, "layout")
        ignored = machine.process_event(ObjectEvent("two_hand_hold", "completed", 1.0))
        self.assertFalse(ignored.accepted)
        self.assertEqual(machine.get_step_state("hold").status, StepStatus.PENDING)

        machine.process_event(ObjectEvent("cup_layout", "completed", 2.0))
        self.assertEqual(machine.current_step_id, "hold")
        machine.process_event(ObjectEvent("two_hand_hold", "completed", 3.0))
        self.assertEqual(machine.current_step_id, "smell")
        machine.process_event(ObjectEvent("open_lid_smell", "completed", 4.0, 0.9))
        self.assertTrue(machine.is_complete)

    def test_timeout_retries_then_fails(self):
        machine = SopStateMachine(configs(), mode="strict")
        machine.tick(10.0)
        first = machine.tick(12.0)
        self.assertEqual(first[0].action, "timeout_retry")
        self.assertEqual(machine.get_step_state("layout").attempts, 2)
        self.assertEqual(machine.get_step_state("layout").status, StepStatus.ACTIVE)

        second = machine.tick(14.0)
        self.assertEqual(second[0].action, "timeout")
        self.assertEqual(machine.get_step_state("layout").status, StepStatus.FAILED)
        self.assertEqual(machine.status, "failed")

    def test_configured_failure_records_result_and_continues(self):
        steps = [
            SopStepConfig(
                "timer", "brew_timer", continue_on_failure=True,
            ),
            SopStepConfig(
                "decant", "decant_action", prerequisites=("timer",),
            ),
        ]
        machine = SopStateMachine(steps, mode="strict")
        transition = machine.process_event(ObjectEvent(
            "brew_timer", "failed", 12.5, reason="冲泡时间超过12秒"
        ))
        self.assertEqual(transition.action, "failed_continue")
        self.assertEqual(machine.get_step_state("timer").status, StepStatus.SKIPPED)
        self.assertEqual(machine.current_step_id, "decant")

    def test_uncertain_and_low_confidence_require_review(self):
        machine = SopStateMachine(configs())
        result = machine.process_event(ObjectEvent(
            "open_lid_smell", "completed", 1.0, confidence=0.5
        ))
        self.assertEqual(result.action, "needs_review")
        self.assertTrue(machine.needs_review)
        approved = machine.resolve_review("smell", True, 2.0, "human confirmed")
        self.assertEqual(approved.action, "review_approved")
        self.assertEqual(machine.get_step_state("smell").status, StepStatus.COMPLETED)

        machine.process_event({
            "observation_id": "two_hand_hold",
            "event_type": "uncertain",
            "timestamp": 3.0,
            "confidence": 0.7,
            "reason": "one hand is occluded",
        })
        state = machine.get_step_state("hold")
        self.assertEqual(state.status, StepStatus.NEEDS_REVIEW)
        self.assertEqual(state.review_reason, "one hand is occluded")

    def test_explicit_skip_unlocks_dependent_step(self):
        machine = SopStateMachine(configs(), mode="strict")
        skipped = machine.skip_step("layout", "equipment unavailable", 1.0)
        self.assertTrue(skipped)
        self.assertEqual(machine.get_step_state("layout").status, StepStatus.SKIPPED)
        self.assertEqual(machine.current_step_id, "hold")

        rejected = machine.skip_step("hold", "not allowed", 2.0)
        self.assertFalse(rejected)
        self.assertEqual(machine.get_step_state("hold").status, StepStatus.ACTIVE)

    def test_failure_automatically_retries_within_budget(self):
        machine = SopStateMachine(configs())
        first = machine.process_event(ObjectEvent("cup_layout", "failed", 1.0))
        self.assertEqual(first.action, "failed_retry")
        self.assertEqual(machine.get_step_state("layout").attempts, 2)
        second = machine.process_event(ObjectEvent("cup_layout", "failed", 2.0))
        self.assertEqual(second.status, StepStatus.FAILED)

    def test_state_round_trip_and_file_serialization(self):
        machine = SopStateMachine(configs(), mode="strict")
        machine.process_event(ObjectEvent("cup_layout", "completed", 2.0))
        restored = SopStateMachine.from_json(machine.to_json())
        self.assertEqual(restored.mode, SopMode.STRICT)
        self.assertEqual(restored.current_step_id, "hold")
        self.assertEqual(restored.to_dict(), machine.to_dict())

        with tempfile.TemporaryDirectory() as temp_dir:
            path = machine.save_json(Path(temp_dir) / "state.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertEqual(payload["runtime"]["layout"]["status"], "completed")

    def test_reset_clears_progress_and_reactivates_first_strict_step(self):
        machine = SopStateMachine(configs(), mode="strict")
        machine.process_event(ObjectEvent("cup_layout", "completed", 2.0))
        self.assertEqual(machine.current_step_id, "hold")
        self.assertTrue(machine.transition_history)

        machine.reset()

        self.assertEqual(machine.current_step_id, "layout")
        self.assertEqual(machine.last_timestamp, None)
        self.assertEqual(machine.transition_history, [])
        self.assertEqual(
            machine.get_step_state("layout").status, StepStatus.ACTIVE
        )
        self.assertEqual(
            machine.get_step_state("hold").status, StepStatus.PENDING
        )

    def test_invalid_prerequisite_is_rejected(self):
        with self.assertRaises(ValueError):
            SopStateMachine([
                SopStepConfig("one", "obs_one", prerequisites=("missing",))
            ])


if __name__ == "__main__":
    unittest.main()
