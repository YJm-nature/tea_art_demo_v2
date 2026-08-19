import tempfile
from pathlib import Path
import unittest

import numpy as np
import yaml

from scripts.train_6gb import _validate_data_yaml
from src.display_ocr import StableNumericReader, parse_numeric_text
from src.observation_catalog import build_available_observations, registered_observation_ids
from src.observation_runtime import (
    CameraRole, EventPhase, FrameContext, ObservationEvent, ObservationState,
)
from src.observations import (
    BrewDurationObservation, TeaLotusToGaiwanObservation,
    TeaWeightObservation, WaterInjectionObservation, WaterTemperatureObservation,
)
from src.sop_config import load_sop_config
from src.sop_runtime import DEFAULT_SOP_CONFIG
from src.sop_scoring import SopScoreLedger
from src.tea_detector import DetectedItem
from src.vessel_pose import PourInteractionAnalyzer


PROJECT = Path(__file__).resolve().parents[1]


def item(name, center, size=(80, 80), track_id=None):
    width, height = size
    return DetectedItem(
        bbox=(int(center[0] - width / 2), int(center[1] - height / 2), width, height),
        centroid=center,
        contour_area=width * height,
        confidence=0.95,
        item_name=name,
        track_id=track_id,
    )


def context(timestamp, classes=(), detections=(), extras=None):
    return FrameContext(
        frame_idx=int(timestamp * 10) + 1,
        timestamp=float(timestamp),
        camera_role=CameraRole.FRONT,
        frame_shape=(720, 1280),
        detections=list(detections),
        hand_results=[],
        pose_results=[],
        model_version="test",
        model_classes=set(classes),
        capabilities={"vessel_pose", "display_ocr"},
        extras=extras or {},
    )


def event(observation_id, phase, start, end, confidence=0.9):
    return ObservationEvent(
        observation_id=observation_id,
        name=observation_id,
        sop_step=5,
        phase=phase,
        start_time=start,
        end_time=end,
        confidence=confidence,
        camera_role="front",
    )


class OcrTests(unittest.TestCase):
    def test_numeric_parser_and_five_sample_stability(self):
        self.assertEqual(parse_numeric_text(" 95C"), 95.0)
        self.assertEqual(parse_numeric_text("3,8 g"), 3.8)
        reader = StableNumericReader(samples=5, tolerance=0.2)
        for value in (3.9, 4.0, 4.0, 4.1):
            stable_value, _, stable, _ = reader.update(value, 0.9)
            self.assertFalse(stable)
            self.assertIsNone(stable_value)
        stable_value, confidence, stable, values = reader.update(4.0, 0.9)
        self.assertTrue(stable)
        self.assertEqual(stable_value, 4.0)
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(confidence, 0.9)

    def test_temperature_and_weight_boundaries(self):
        temperature = WaterTemperatureObservation()
        snapshot, events = temperature.update(context(1.0, extras={
            "ocr_measurements": {"temperature": {
                "value": 95.0, "stable": True, "confidence": 0.99,
            }}
        }))
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual(events[0].phase, EventPhase.COMPLETED)

        weight = TeaWeightObservation()
        snapshot, events = weight.update(context(2.0, extras={
            "ocr_measurements": {"weight": {
                "value": 5.1, "stable": True, "confidence": 0.99,
            }}
        }))
        self.assertEqual(snapshot.state, ObservationState.FAILED)
        self.assertEqual(events[0].phase, EventPhase.FAILED)


class PoseAndActionTests(unittest.TestCase):
    def test_pose_analyzer_requires_lift_tilt_contact_and_alignment(self):
        analyzer = PourInteractionAnalyzer()
        target = item("盖碗碗身", (500, 400), (120, 100), track_id=7)
        pose = {
            "class_name": "烧水壶",
            "bbox": [300, 300, 160, 140],
            "confidence": 0.95,
            "keypoints": [[455, 390], [380, 360], [310, 350]],
            "keypoint_confidences": [0.9, 0.9, 0.9],
            "tilt_delta_degrees": 25.0,
            "lifted": True,
        }
        rows = analyzer.update(
            [target], [{"center": (380, 360)}], [pose], 1.0
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "烧水壶")
        self.assertEqual(rows[0]["target_track_id"], 7)
        self.assertFalse(rows[0]["liquid_verified"])
        self.assertEqual(rows[0]["target_center"], [500.0, 400.0])

        pose["lifted"] = False
        self.assertEqual(
            analyzer.update([target], [{"center": (380, 360)}], [pose], 1.2), []
        )

    def test_tea_lotus_to_gaiwan_debounces_pose_interaction(self):
        observer = TeaLotusToGaiwanObservation(stable_seconds=0.5, min_samples=3)
        interaction = {
            "source": "茶荷", "target": "盖碗碗身", "confidence": 0.9,
            "liquid_verified": False,
        }
        events = []
        for timestamp in (0.0, 0.25, 0.5):
            snapshot, emitted = observer.update(context(
                timestamp,
                {"茶荷", "盖碗碗身"},
                extras={"pour_interactions": [interaction]},
            ))
            events.extend(emitted)
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual([row.phase for row in events], [EventPhase.STARTED, EventPhase.COMPLETED])
        self.assertTrue(events[-1].metrics["gesture_only"])

    def test_rotating_injection_requires_orbit_then_completes(self):
        observer = WaterInjectionObservation(
            stable_seconds=0.5, min_samples=3, minimum_arc_degrees=10.0
        )
        classes = {"烧水壶", "盖碗碗身"}
        events = []
        for index, angle in enumerate((0, 10, 20, 30, 40, 50, 60)):
            radians = np.deg2rad(angle)
            interaction = {
                "source": "烧水壶", "target": "盖碗碗身", "confidence": 0.9,
                "outlet_point": [500 + 40 * np.cos(radians), 400 + 40 * np.sin(radians)],
                "target_center": [500, 400], "liquid_verified": False,
            }
            snapshot, emitted = observer.update(context(
                index * 0.25, classes, extras={"pour_interactions": [interaction]}
            ))
            events.extend(emitted)
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertTrue(snapshot.metrics["orbit_verified"])


class BrewAndConfigurationTests(unittest.TestCase):
    def test_brew_duration_accepts_8_to_12_seconds(self):
        observer = BrewDurationObservation()
        first = [
            event("action_water_injection", EventPhase.COMPLETED, 0.0, 0.5),
            event("action_gaiwan_lid_close_brew", EventPhase.COMPLETED, 0.6, 1.0),
        ]
        observer.update(context(1.0, extras={"frame_observation_events": first}))
        decant = event("action_gaiwan_to_pitcher", EventPhase.COMPLETED, 10.0, 10.5)
        snapshot, emitted = observer.update(context(
            10.5, extras={"frame_observation_events": [decant]}
        ))
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual(snapshot.metrics["wait_seconds"], 9.0)
        self.assertEqual(emitted[0].phase, EventPhase.COMPLETED)

    def test_front_config_nodes_are_registered(self):
        config = load_sop_config(DEFAULT_SOP_CONFIG)
        self.assertEqual(config["runtime"]["camera_mode"], "front")
        self.assertEqual(config["thresholds"]["brew_seconds"], [8.0, 12.0])
        enabled = [row for row in config["runtime_nodes"] if row.get("runtime_enabled", True)]
        self.assertEqual(len(enabled), 14)
        registered = registered_observation_ids()
        self.assertEqual(
            [row["observation_id"] for row in enabled if row["observation_id"] not in registered],
            [],
        )

    def test_front_builder_exposes_new_observers(self):
        classes = {
            "盖碗碗身", "盖碗碗盖", "公道杯", "品茗杯", "茶荷", "茶盘",
            "烧水壶", "建水", "电子秤", "茶叶罐", "水壶显示屏", "电子秤显示屏",
        }
        ids = {
            row.observation_id
            for row in build_available_observations(classes, CameraRole.FRONT)
        }
        self.assertTrue({
            "seq_warm_clean_front", "result_water_temperature", "result_tea_weight",
            "action_tea_lotus_to_gaiwan", "action_water_injection",
            "result_brew_time_8_12", "action_tea_distribution",
            "action_two_hand_serve_tray",
        } <= ids)

    def test_pose_training_yaml_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.yaml"
            path.write_text(yaml.safe_dump({
                "path": temp_dir,
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {0: "kettle", 1: "pitcher", 2: "gaiwan", 3: "lotus"},
                "kpt_shape": [3, 3],
                "flip_idx": [2, 1, 0],
            }), encoding="utf-8")
            _validate_data_yaml(path, require_test=True, task="pose")

    def test_scoring_ledger_is_evidence_only_without_weights(self):
        ledger = SopScoreLedger()
        ledger.consume([
            event("action_water_injection", EventPhase.COMPLETED, 1.0, 2.0),
            event("result_brew_time_8_12", EventPhase.FAILED, 2.0, 15.0),
        ])
        report = ledger.to_dict()
        self.assertEqual(report["score_status"], "evidence_only")
        self.assertIsNone(report["weighted_score"])
        self.assertEqual(report["summary"]["passed"], 1)
        self.assertEqual(report["summary"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()

