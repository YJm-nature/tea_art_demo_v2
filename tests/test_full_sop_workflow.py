from pathlib import Path
import unittest

from src.observation_catalog import (
    build_available_observations,
    build_default_observations,
    observation_specs,
    registered_observation_ids,
)
from src.observation_runtime import CameraRole
from src.sop_config import load_sop_config
from src.sop_replay import replay_sop_events
from src.sop_runtime import build_sop_state_machine
from src.sop_state_machine import SopStateMachine, StepStatus


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "sop_red_tea_v1.yaml"

FULL_EVENT_IDS = [
    "obj_utensils_s1",
    "seq_warm_clean_order",
    "action_tea_canister_to_lotus",
    "action_hold_lotus",
    "action_open_lid_smell",
    "result_brew_wait_decant_partial",
    "result_filled_cup_tray_layout",
]


def completed(observation_id, timestamp, confidence=0.95):
    return {
        "observation_id": observation_id,
        "phase": "completed",
        "end_time": float(timestamp),
        "confidence": confidence,
    }


def full_machine():
    machine = build_sop_state_machine(
        config_path=CONFIG_PATH,
        mode="strict",
        include_deferred=True,
        include_disabled=True,
    )
    assert machine is not None
    return machine


class ObservationCatalogIntegrationTests(unittest.TestCase):
    def test_legacy_action_module_keeps_public_imports_compatible(self):
        from src.action_observations import CupLayoutObservation
        from src.observations.layout import CupLayoutObservation as OrganizedLayout

        self.assertIs(CupLayoutObservation, OrganizedLayout)

    def test_catalog_ids_and_instances_are_unique(self):
        specs = observation_specs()
        identifiers = [spec.observation_id for spec in specs]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(set(identifiers), set(registered_observation_ids()))
        instances = build_default_observations()
        self.assertEqual(
            {item.observation_id for item in instances}, set(identifiers)
        )

    def test_every_temporal_sop_node_has_a_registered_observer(self):
        config = load_sop_config(CONFIG_PATH)
        temporal_nodes = [
            node
            for node in config["runtime_nodes"]
            if node.get("runtime_enabled", True)
        ]
        registered = registered_observation_ids()
        missing = [
            node["observation_id"]
            for node in temporal_nodes
            if node["observation_id"] not in registered
        ]
        self.assertEqual(missing, [])
        setup = config["runtime_nodes"][0]
        self.assertEqual(setup["observation_id"], "obj_utensils_s1")
        self.assertFalse(setup["runtime_enabled"])

    def test_current_single_camera_builds_configured_runtime_subset(self):
        classes = {
            "盖碗碗身",
            "盖碗碗盖",
            "公道杯",
            "品茗杯",
            "茶荷",
            "茶巾",
            "茶夹",
            "茶拨",
            "建水",
            "茶叶罐",
        }
        observations = build_available_observations(classes, CameraRole.SINGLE)
        machine = build_sop_state_machine(
            config_path=CONFIG_PATH,
            mode="strict",
            available_observation_ids={item.observation_id for item in observations},
        )
        self.assertIsNotNone(machine)
        self.assertEqual(
            [step.step_id for step in machine.steps],
            ["tea_prepare", "hold_lotus", "smell", "brew_partial"],
        )


class FullConfiguredSopStateMachineTests(unittest.TestCase):
    def test_full_configuration_covers_six_business_steps(self):
        config = load_sop_config(CONFIG_PATH)
        node_steps = {
            node["business_step"]
            for node in config["runtime_nodes"]
        }
        self.assertEqual(
            node_steps,
            {
                "step01_setup",
                "step02_warm_clean",
                "step03_tea_preparation",
                "step04_add_tea_smell",
                "step05_brew",
                "step06_serve",
            },
        )

    def test_full_strict_sequence_completes_all_seven_nodes(self):
        machine = full_machine()
        self.assertEqual(machine.current_step_id, "setup")
        for timestamp, observation_id in enumerate(FULL_EVENT_IDS, start=1):
            transition = machine.process_event(completed(observation_id, timestamp))
            self.assertTrue(transition.accepted, transition.reason)
        self.assertTrue(machine.is_complete)
        self.assertEqual(machine.status, "completed")
        self.assertEqual(
            [state.status for state in machine.runtime.values()],
            [StepStatus.COMPLETED] * 7,
        )

    def test_out_of_order_event_does_not_advance_full_chain(self):
        machine = full_machine()
        transition = machine.process_event(
            completed("action_tea_canister_to_lotus", 1.0)
        )
        self.assertFalse(transition.accepted)
        self.assertEqual(transition.action, "ignored")
        self.assertEqual(machine.current_step_id, "setup")
        self.assertEqual(
            machine.get_step_state("tea_prepare").status, StepStatus.PENDING
        )

    def test_low_confidence_review_can_resume_exact_next_step(self):
        machine = full_machine()
        machine.process_event(completed("obj_utensils_s1", 1.0))
        transition = machine.process_event(
            completed("seq_warm_clean_order", 2.0, confidence=0.2)
        )
        self.assertEqual(transition.action, "needs_review")
        self.assertEqual(machine.current_step_id, "warm_clean")
        approved = machine.resolve_review("warm_clean", True, 3.0)
        self.assertEqual(approved.action, "review_approved")
        self.assertEqual(machine.current_step_id, "tea_prepare")

    def test_timeout_retries_then_blocks_the_full_chain(self):
        machine = full_machine()
        machine.tick(0.0)
        first_timeout = machine.tick(120.0)
        second_timeout = machine.tick(240.0)
        self.assertEqual(first_timeout[-1].action, "timeout_retry")
        self.assertEqual(second_timeout[-1].action, "timeout")
        self.assertEqual(machine.status, "failed")
        ignored = machine.process_event(completed("seq_warm_clean_order", 241.0))
        self.assertFalse(ignored.accepted)
        self.assertEqual(machine.get_step_state("warm_clean").status, StepStatus.PENDING)

    def test_serialized_mid_flow_resumes_without_replaying_completed_nodes(self):
        machine = full_machine()
        for timestamp, observation_id in enumerate(FULL_EVENT_IDS[:3], start=1):
            machine.process_event(completed(observation_id, timestamp))
        restored = SopStateMachine.from_json(machine.to_json())
        self.assertEqual(restored.current_step_id, "hold_lotus")
        duplicate = restored.process_event(completed("obj_utensils_s1", 4.0))
        self.assertFalse(duplicate.accepted)
        for timestamp, observation_id in enumerate(FULL_EVENT_IDS[3:], start=5):
            restored.process_event(completed(observation_id, timestamp))
        self.assertTrue(restored.is_complete)

    def test_offline_replay_reports_full_scope_but_not_formal_acceptance(self):
        records = [
            completed(observation_id, timestamp)
            for timestamp, observation_id in enumerate(FULL_EVENT_IDS, start=1)
        ]
        report = replay_sop_events(
            records,
            config_path=CONFIG_PATH,
            mode="strict",
            include_deferred=True,
            include_disabled=True,
        )
        self.assertTrue(report["summary"]["is_complete"])
        self.assertEqual(report["summary"]["accepted_record_count"], 7)
        self.assertFalse(report["scope"]["current_capabilities_only"])
        self.assertFalse(report["scope"]["formal_acceptance_enabled"])
        self.assertEqual(report["scope"]["omitted_runtime_nodes"], [])


if __name__ == "__main__":
    unittest.main()
