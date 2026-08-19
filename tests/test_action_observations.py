import unittest

import numpy as np

from src.observations import (
    CupLayoutObservation,
    BrewWaitTimerObservation,
    FilledCupTrayLayoutObservation,
    GaiwanLidClosureObservation,
    GaiwanToPitcherObservation,
    HandAccessoryObservation,
    LidOpenSmellObservation,
    TeaCanisterToLotusObservation,
    TwoHandHoldObservation,
    WarmCleanSequenceObservation,
)
from src.observation_catalog import build_available_observations
from src.observation_runtime import (
    CameraRole,
    EventPhase,
    FrameContext,
    ObservationEvent,
    ObservationEngine,
    ObservationState,
)
from src.sop_state_machine import SopStateMachine, SopStepConfig, StepStatus
from src.tea_detector import DetectedItem


def item(name, center, size=(50, 50), confidence=0.9, track_id=None):
    width, height = size
    x = center[0] - width / 2
    y = center[1] - height / 2
    return DetectedItem(
        bbox=(int(x), int(y), int(width), int(height)),
        centroid=center,
        contour_area=width * height,
        confidence=confidence,
        item_name=name,
        track_id=track_id,
    )


def pose(nose=(450, 210)):
    landmarks = np.zeros((33, 3), dtype=np.float32)
    landmarks[0, :2] = nose
    landmarks[11, :2] = (350, 200)
    landmarks[12, :2] = (550, 200)
    landmarks[13, :2] = (360, 350)
    landmarks[14, :2] = (540, 350)
    return {"landmarks": landmarks}


def hand(center, confidence=0.9):
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[:, :2] = center
    return {
        "center": center,
        "bbox": (int(center[0] - 20), int(center[1] - 20), 40, 40),
        "confidence": confidence,
        "landmarks": landmarks,
    }


def context(
    timestamp,
    detections=None,
    hands=None,
    poses=None,
    role=CameraRole.TABLETOP,
    model_classes=None,
    capabilities=None,
    extras=None,
):
    return FrameContext(
        frame_idx=int(timestamp * 10) + 1,
        timestamp=timestamp,
        camera_role=role,
        frame_shape=(720, 1280),
        detections=detections or [],
        hand_results=hands or [],
        pose_results=poses or [],
        model_version="test-model",
        model_classes=set(model_classes or []),
        capabilities=set(capabilities or []),
        extras=extras or {},
    )


class CupLayoutTests(unittest.TestCase):
    def test_single_camera_builds_only_supported_observers(self):
        observations = build_available_observations(
            ["盖碗", "品茗杯", "茶荷", "茶巾"], CameraRole.SINGLE
        )
        self.assertEqual(
            {item.observation_id for item in observations},
            {"result_cup_layout", "action_hold_lotus"},
        )

    def test_classifies_line_and_pin_layouts(self):
        line = [item("品茗杯", (300 + index * 100, 300)) for index in range(3)]
        pin = [item("品茗杯", point) for point in ((400, 250), (500, 250), (450, 350))]
        self.assertEqual(CupLayoutObservation.evaluate(line).label, "一字形")
        self.assertEqual(CupLayoutObservation.evaluate(pin).label, "品字形")

    def test_classifies_compact_upper_cup_as_pin_layout(self):
        cups = [
            item("品茗杯", (440, 360), size=(100, 100)),
            item("品茗杯", (560, 390), size=(100, 100)),
            item("品茗杯", (500, 300), size=(100, 100)),
        ]
        self.assertEqual(CupLayoutObservation.evaluate(cups).label, "品字形")

    def test_classifies_offset_third_cup_as_other_layout(self):
        cups = [
            item("品茗杯", (400, 625), size=(105, 105)),
            item("品茗杯", (507, 618), size=(105, 105)),
            item("品茗杯", (597, 700), size=(105, 105)),
        ]
        evaluation = CupLayoutObservation.evaluate(cups)
        self.assertIsNone(evaluation.label)
        self.assertEqual(evaluation.metrics["classification"], "其他布局")

        snapshot, events = CupLayoutObservation().update(context(0.0, cups))
        self.assertEqual(snapshot.state, ObservationState.IDLE)
        self.assertEqual(snapshot.value, "其他布局")
        self.assertFalse(events)

    def test_rejects_missing_and_irregular_layouts(self):
        missing = [item("品茗杯", (300, 300))]
        irregular = [item("品茗杯", point) for point in ((100, 100), (420, 140), (230, 510), (700, 300))]
        self.assertIsNone(CupLayoutObservation.evaluate(missing).label)
        self.assertIsNone(CupLayoutObservation.evaluate(irregular).label)

    def test_emits_only_after_stable_window(self):
        observation = CupLayoutObservation(stable_seconds=1.0, min_samples=5)
        cups = [item("品茗杯", (300 + index * 100, 300)) for index in range(3)]
        events = []
        for timestamp in (0.0, 0.25, 0.5, 0.75, 1.0):
            snapshot, emitted = observation.update(context(timestamp, cups))
            events.extend(emitted)
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual([event.phase for event in events], [EventPhase.STARTED, EventPhase.COMPLETED])
        self.assertEqual(events[-1].value, "一字形")

        # The real-time snapshot follows the current geometry after completion.
        snapshot, emitted = observation.update(context(1.25, []))
        self.assertEqual(snapshot.state, ObservationState.UNCERTAIN)
        self.assertIsNone(snapshot.value)
        self.assertEqual(emitted, [])

    def test_switches_from_line_to_pin_after_new_layout_stabilizes(self):
        observation = CupLayoutObservation(stable_seconds=1.0, min_samples=5)
        line = [item("品茗杯", (300 + index * 100, 300)) for index in range(3)]
        pin = [item("品茗杯", point) for point in ((400, 250), (500, 250), (450, 350))]
        events = []
        for timestamp in (0.0, 0.25, 0.5, 0.75, 1.0):
            snapshot, emitted = observation.update(context(timestamp, line))
            events.extend(emitted)
        self.assertEqual(snapshot.value, "一字形")

        for timestamp in (1.25, 1.5, 1.75, 2.0, 2.25):
            snapshot, emitted = observation.update(context(timestamp, pin))
            events.extend(emitted)
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual(snapshot.value, "品字形")
        completed_values = [event.value for event in events if event.phase is EventPhase.COMPLETED]
        self.assertEqual(completed_values, ["一字形", "品字形"])


class FilledCupTrayLayoutTests(unittest.TestCase):
    classes = {"茶盘", "品茗杯"}

    @staticmethod
    def extras(*track_ids):
        return {"filled_cup_track_ids": list(track_ids)}

    def test_filled_cups_on_tray_complete_line_layout(self):
        observation = FilledCupTrayLayoutObservation(stable_seconds=1.0, min_samples=5)
        tray = item("茶盘", (500, 400), size=(600, 320))
        cups = [
            item("品茗杯", (350 + index * 150, 400), size=(60, 60), track_id=index + 1)
            for index in range(3)
        ]
        events = []
        for timestamp in (0.0, 0.25, 0.5, 0.75, 1.0):
            snapshot, emitted = observation.update(context(
                timestamp, [tray, *cups], role=CameraRole.SINGLE,
                model_classes=self.classes, extras=self.extras(1, 2, 3),
            ))
            events.extend(emitted)
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual(snapshot.value, "一字形")
        self.assertTrue(snapshot.metrics["layout_valid"])
        self.assertEqual([event.phase for event in events], [EventPhase.STARTED, EventPhase.COMPLETED])

    def test_filled_cups_on_tray_complete_pin_layout(self):
        observation = FilledCupTrayLayoutObservation(stable_seconds=1.0, min_samples=5)
        tray = item("茶盘", (500, 400), size=(600, 360))
        points = ((430, 440), (570, 440), (500, 350))
        cups = [
            item("品茗杯", point, size=(70, 70), track_id=index + 1)
            for index, point in enumerate(points)
        ]
        for timestamp in (0.0, 0.25, 0.5, 0.75, 1.0):
            snapshot, _ = observation.update(context(
                timestamp, [tray, *cups], role=CameraRole.SINGLE,
                model_classes=self.classes, extras=self.extras(1, 2, 3),
            ))
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual(snapshot.value, "品字形")

    def test_irregular_filled_cups_are_reported_as_nonstandard(self):
        observation = FilledCupTrayLayoutObservation(stable_seconds=1.0, min_samples=5)
        tray = item("茶盘", (450, 420), size=(800, 650))
        points = ((160, 220), (430, 250), (280, 610), (720, 420))
        cups = [
            item("品茗杯", point, size=(70, 70), track_id=index + 1)
            for index, point in enumerate(points)
        ]
        for timestamp in (0.0, 0.25, 0.5, 0.75, 1.0):
            snapshot, _ = observation.update(context(
                timestamp, [tray, *cups], role=CameraRole.SINGLE,
                model_classes=self.classes, extras=self.extras(1, 2, 3, 4),
            ))
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual(snapshot.value, "不规范摆放")
        self.assertFalse(snapshot.metrics["layout_valid"])

    def test_missing_liquid_evidence_is_uncertain(self):
        observation = FilledCupTrayLayoutObservation()
        tray = item("茶盘", (500, 400), size=(600, 320))
        cups = [
            item("品茗杯", (400 + index * 120, 400), track_id=index + 1)
            for index in range(3)
        ]
        snapshot, events = observation.update(context(
            0.0, [tray, *cups], role=CameraRole.SINGLE,
            model_classes=self.classes,
        ))
        self.assertEqual(snapshot.state, ObservationState.UNCERTAIN)
        self.assertFalse(snapshot.metrics["liquid_evidence"])
        self.assertFalse(events)

    def test_builder_adds_tray_layout_but_not_tray_hold(self):
        observations = build_available_observations(self.classes, CameraRole.SINGLE)
        ids = {item.observation_id for item in observations}
        self.assertIn("result_filled_cup_tray_layout", ids)
        self.assertNotIn("action_hold_tray", ids)


class TeaCanisterToLotusTests(unittest.TestCase):
    classes = {"茶叶罐", "茶荷"}

    def test_hand_moves_from_canister_to_lotus_and_completes(self):
        observation = TeaCanisterToLotusObservation(
            source_dwell_seconds=0.4,
            target_dwell_seconds=0.5,
            min_samples=3,
        )
        canister = item("茶叶罐", (300, 400), size=(100, 100))
        lotus = item("茶荷", (650, 400), size=(120, 70))
        events = []
        for timestamp in (0.0, 0.2, 0.4):
            snapshot, emitted = observation.update(context(
                timestamp,
                [canister, lotus],
                [hand((300, 400))],
                role=CameraRole.SINGLE,
                model_classes=self.classes,
            ))
            events.extend(emitted)
        self.assertEqual(snapshot.state, ObservationState.ACTIVE)

        for timestamp, center in (
            (0.6, (450, 400)),
            (0.8, (650, 400)),
            (1.05, (650, 400)),
            (1.3, (650, 400)),
        ):
            snapshot, emitted = observation.update(context(
                timestamp,
                [canister, lotus],
                [hand(center)],
                role=CameraRole.SINGLE,
                model_classes=self.classes,
            ))
            events.extend(emitted)

        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertFalse(snapshot.metrics["content_verified"])
        self.assertEqual(
            [event.phase for event in events],
            [EventPhase.STARTED, EventPhase.COMPLETED],
        )

    def test_touching_only_lotus_does_not_start_transfer(self):
        observation = TeaCanisterToLotusObservation(
            source_dwell_seconds=0.4, min_samples=3
        )
        detections = [
            item("茶叶罐", (300, 400), size=(100, 100)),
            item("茶荷", (650, 400), size=(120, 70)),
        ]
        events = []
        for timestamp in (0.0, 0.2, 0.4, 0.6):
            snapshot, emitted = observation.update(context(
                timestamp,
                detections,
                [hand((650, 400))],
                role=CameraRole.SINGLE,
                model_classes=self.classes,
            ))
            events.extend(emitted)
        self.assertEqual(snapshot.state, ObservationState.IDLE)
        self.assertFalse(events)

    def test_transfer_timeout_requires_a_new_attempt(self):
        observation = TeaCanisterToLotusObservation(
            source_dwell_seconds=0.4,
            target_dwell_seconds=0.5,
            transfer_timeout_seconds=1.0,
            min_samples=3,
        )
        detections = [
            item("茶叶罐", (300, 400), size=(100, 100)),
            item("茶荷", (650, 400), size=(120, 70)),
        ]
        for timestamp in (0.0, 0.2, 0.4):
            observation.update(context(
                timestamp,
                detections,
                [hand((300, 400))],
                role=CameraRole.SINGLE,
                model_classes=self.classes,
            ))
        snapshot, events = observation.update(context(
            1.5,
            detections,
            [hand((500, 400))],
            role=CameraRole.SINGLE,
            model_classes=self.classes,
        ))
        self.assertEqual(snapshot.state, ObservationState.UNCERTAIN)
        self.assertTrue(snapshot.metrics["transfer_timeout"])
        self.assertFalse(events)

    def test_direct_leaf_signal_completes_without_hand_geometry(self):
        observation = TeaCanisterToLotusObservation(
            target_dwell_seconds=0.5, min_samples=3
        )
        detections = [
            item("茶叶罐", (300, 400), size=(100, 100)),
            item("茶荷", (650, 400), size=(120, 70)),
        ]
        extras = {"tea_transfer_interactions": [{
            "source": "茶叶罐",
            "target": "茶荷",
            "confidence": 0.95,
            "signal_source": "leaf_segmentation",
        }]}
        events = []
        for timestamp in (0.0, 0.25, 0.5):
            snapshot, emitted = observation.update(context(
                timestamp,
                detections,
                role=CameraRole.SINGLE,
                model_classes=self.classes,
                extras=extras,
            ))
            events.extend(emitted)
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertTrue(snapshot.metrics["content_verified"])
        self.assertEqual(
            [event.phase for event in events],
            [EventPhase.STARTED, EventPhase.COMPLETED],
        )
        snapshot, emitted = observation.update(context(
            0.75,
            detections,
            role=CameraRole.SINGLE,
            model_classes=self.classes,
        ))
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertTrue(snapshot.metrics["content_verified"])
        self.assertFalse(emitted)

    def test_builder_requires_canister_and_lotus(self):
        available = build_available_observations(self.classes, CameraRole.SINGLE)
        missing = build_available_observations({"茶荷"}, CameraRole.SINGLE)
        self.assertIn(
            "action_tea_canister_to_lotus",
            {item.observation_id for item in available},
        )
        self.assertNotIn(
            "action_tea_canister_to_lotus",
            {item.observation_id for item in missing},
        )


class TwoHandHoldTests(unittest.TestCase):
    def test_lotus_hold_belongs_to_step_four_and_defaults_to_five_seconds(self):
        observation = TwoHandHoldObservation("茶荷")
        self.assertEqual(observation.sop_step, 4)
        self.assertEqual(observation.stable_seconds, 5.0)
        self.assertIn("赏茶", observation.name)

    def test_two_hands_on_opposite_sides_complete_hold(self):
        observation = TwoHandHoldObservation("茶荷", stable_seconds=0.8)
        lotus = item("茶荷", (450, 275), size=(100, 50))
        hands = [hand((395, 275)), hand((505, 275))]
        events = []
        for timestamp in (0.0, 0.2, 0.4, 0.6, 0.8):
            snapshot, emitted = observation.update(context(
                timestamp, [lotus], hands, [pose()], role=CameraRole.SIDE
            ))
            events.extend(emitted)
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual([event.phase for event in events], [EventPhase.STARTED, EventPhase.COMPLETED])

    def test_one_hand_or_missing_pose_is_uncertain(self):
        observation = TwoHandHoldObservation("茶荷")
        lotus = item("茶荷", (450, 275), size=(100, 50))
        snapshot, events = observation.update(context(
            0.0, [lotus], [hand((395, 275))], [], role=CameraRole.SIDE
        ))
        self.assertEqual(snapshot.state, ObservationState.UNCERTAIN)
        self.assertFalse(events)

    def test_single_camera_completes_without_pose_using_hand_target_geometry(self):
        observation = TwoHandHoldObservation("茶荷", stable_seconds=0.8)
        resting_lotus = item("茶荷", (450, 400), size=(100, 50))
        for timestamp in (0.0, 0.1, 0.2, 0.3, 0.4):
            observation.update(context(
                timestamp, [resting_lotus], [], [], role=CameraRole.SINGLE
            ))
        lotus = item("茶荷", (450, 300), size=(100, 50))
        hands = [hand((395, 300)), hand((505, 300))]
        events = []
        for timestamp in (0.6, 0.8, 1.0, 1.2, 1.4):
            snapshot, emitted = observation.update(context(
                timestamp, [lotus], hands, [], role=CameraRole.SINGLE
            ))
            events.extend(emitted)
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertTrue(snapshot.value)
        self.assertFalse(snapshot.metrics["chest_region_verified"])
        self.assertTrue(snapshot.metrics["lift_verified"])
        self.assertTrue(snapshot.metrics["hands_contact_target"])
        self.assertEqual(snapshot.metrics["geometry_mode"], "single_camera_relational")
        self.assertEqual(
            [event.phase for event in events],
            [EventPhase.STARTED, EventPhase.COMPLETED],
        )

    def test_single_camera_rejects_two_hands_on_same_side(self):
        observation = TwoHandHoldObservation("茶荷", stable_seconds=0.8)
        resting_lotus = item("茶荷", (450, 400), size=(100, 50))
        for timestamp in (0.0, 0.1, 0.2, 0.3, 0.4):
            observation.update(context(
                timestamp, [resting_lotus], [], [], role=CameraRole.SINGLE
            ))
        lotus = item("茶荷", (450, 275), size=(100, 50))
        hands = [hand((500, 250)), hand((520, 300))]
        snapshot, _ = observation.update(context(
            0.0, [lotus], hands, [], role=CameraRole.SINGLE
        ))
        self.assertEqual(snapshot.state, ObservationState.IDLE)
        self.assertIn("分处茶荷两侧", snapshot.reason)

    def test_single_camera_rejects_hands_around_lotus_still_on_table(self):
        observation = TwoHandHoldObservation("茶荷", stable_seconds=0.8)
        lotus = item("茶荷", (450, 400), size=(100, 50))
        for timestamp in (0.0, 0.1, 0.2, 0.3, 0.4):
            observation.update(context(
                timestamp, [lotus], [], [], role=CameraRole.SINGLE
            ))
        hands = [hand((395, 400)), hand((505, 400))]
        snapshot, _ = observation.update(context(
            0.6, [lotus], hands, [], role=CameraRole.SINGLE
        ))
        self.assertEqual(snapshot.state, ObservationState.IDLE)
        self.assertFalse(snapshot.metrics["lift_verified"])
        self.assertIn("桌面位置", snapshot.reason)

    def test_single_camera_rejects_hands_that_do_not_contact_target(self):
        observation = TwoHandHoldObservation("茶荷", stable_seconds=0.8)
        resting_lotus = item("茶荷", (450, 400), size=(100, 50))
        for timestamp in (0.0, 0.1, 0.2, 0.3, 0.4):
            observation.update(context(
                timestamp, [resting_lotus], [], [], role=CameraRole.SINGLE
            ))
        lifted_lotus = item("茶荷", (450, 300), size=(100, 50))
        separated_hands = [hand((360, 300)), hand((540, 300))]
        snapshot, _ = observation.update(context(
            0.7, [lifted_lotus], separated_hands, [], role=CameraRole.SINGLE
        ))
        self.assertEqual(snapshot.state, ObservationState.IDLE)
        self.assertTrue(snapshot.metrics["lift_verified"])
        self.assertFalse(snapshot.metrics["hands_contact_target"])
        self.assertIn("接触", snapshot.reason)

    def test_completed_hold_returns_to_idle_after_target_is_put_down(self):
        observation = TwoHandHoldObservation(
            "茶荷", stable_seconds=0.8, release_seconds=0.5
        )
        resting_lotus = item("茶荷", (450, 400), size=(100, 50))
        for timestamp in (0.0, 0.1, 0.2, 0.3, 0.4):
            observation.update(context(
                timestamp, [resting_lotus], [], [], role=CameraRole.SINGLE
            ))
        contact_hands = [hand((395, 400)), hand((505, 400))]
        observation.update(context(
            0.6, [resting_lotus], contact_hands, [], role=CameraRole.SINGLE
        ))
        held_lotus = item("茶荷", (450, 300), size=(100, 50))
        hands = [hand((395, 300)), hand((505, 300))]
        for timestamp in (0.8, 1.0, 1.2, 1.4, 1.6):
            snapshot, _ = observation.update(context(
                timestamp, [held_lotus], hands, [], role=CameraRole.SINGLE
            ))
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)

        snapshot, _ = observation.update(context(
            1.8, [resting_lotus], [], [], role=CameraRole.SINGLE
        ))
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        snapshot, _ = observation.update(context(
            2.4, [resting_lotus], [], [], role=CameraRole.SINGLE
        ))
        self.assertEqual(snapshot.state, ObservationState.UNCERTAIN)
        self.assertIsNone(snapshot.value)


class WarmCleanSequenceTests(unittest.TestCase):
    classes = {"烧水壶", "盖碗碗身", "公道杯", "品茗杯"}

    @staticmethod
    def signal(source, target, target_id=None):
        row = {
            "source": source,
            "target": target,
            "confidence": 0.95,
            "signal_source": "test_liquid",
        }
        if target_id is not None:
            row["target_track_id"] = target_id
        return {"pour_interactions": [row]}

    def test_correct_order_and_two_distinct_cups_complete(self):
        observation = WarmCleanSequenceObservation(
            stable_seconds=0.5, min_samples=3, min_cups=2
        )
        rows = [
            (0.0, self.signal("烧水壶", "盖碗碗身")),
            (0.25, self.signal("烧水壶", "盖碗碗身")),
            (0.5, self.signal("烧水壶", "盖碗碗身")),
            (1.0, self.signal("盖碗碗身", "公道杯")),
            (1.25, self.signal("盖碗碗身", "公道杯")),
            (1.5, self.signal("盖碗碗身", "公道杯")),
            (2.0, self.signal("公道杯", "品茗杯", 101)),
            (2.25, self.signal("公道杯", "品茗杯", 101)),
            (2.5, self.signal("公道杯", "品茗杯", 101)),
            (3.0, self.signal("公道杯", "品茗杯", 102)),
            (3.25, self.signal("公道杯", "品茗杯", 102)),
            (3.5, self.signal("公道杯", "品茗杯", 102)),
        ]
        events = []
        for timestamp, extras in rows:
            snapshot, emitted = observation.update(context(
                timestamp,
                role=CameraRole.SINGLE,
                model_classes=self.classes,
                extras=extras,
            ))
            events.extend(emitted)

        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual(snapshot.value, "盖碗→公道杯→品茗杯")
        self.assertEqual(snapshot.metrics["cup_targets_completed"], 2)
        self.assertEqual(
            [event.phase for event in events],
            [EventPhase.STARTED, EventPhase.COMPLETED],
        )

    def test_repeated_same_cup_does_not_complete(self):
        observation = WarmCleanSequenceObservation(
            stable_seconds=0.5, min_samples=3, min_cups=2
        )
        prefixes = (
            ("烧水壶", "盖碗碗身", None),
            ("盖碗碗身", "公道杯", None),
        )
        timestamp = 0.0
        for source, target, target_id in prefixes:
            for _ in range(3):
                observation.update(context(
                    timestamp,
                    role=CameraRole.SINGLE,
                    model_classes=self.classes,
                    extras=self.signal(source, target, target_id),
                ))
                timestamp += 0.25
        for _ in range(6):
            snapshot, _ = observation.update(context(
                timestamp,
                role=CameraRole.SINGLE,
                model_classes=self.classes,
                extras=self.signal("公道杯", "品茗杯", 101),
            ))
            timestamp += 0.25
        self.assertNotEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual(snapshot.metrics["cup_targets_completed"], 1)

    def test_stable_out_of_order_interaction_is_uncertain(self):
        observation = WarmCleanSequenceObservation(
            stable_seconds=0.5, min_samples=3
        )
        for timestamp in (0.0, 0.25, 0.5):
            snapshot, events = observation.update(context(
                timestamp,
                role=CameraRole.SINGLE,
                model_classes=self.classes,
                extras=self.signal("盖碗碗身", "公道杯"),
            ))
        self.assertEqual(snapshot.state, ObservationState.UNCERTAIN)
        self.assertTrue(snapshot.metrics["out_of_order"])
        self.assertIn("盖碗", snapshot.reason)
        self.assertFalse(events)

    def test_missing_kettle_capability_is_uncertain(self):
        observation = WarmCleanSequenceObservation()
        snapshot, events = observation.update(context(
            0.0,
            role=CameraRole.SINGLE,
            model_classes={"盖碗碗身", "公道杯", "品茗杯"},
        ))
        self.assertEqual(snapshot.state, ObservationState.UNCERTAIN)
        self.assertIn("烧水壶", snapshot.reason)
        self.assertFalse(events)

    def test_builder_enables_warm_clean_only_with_required_classes(self):
        observations = build_available_observations(
            self.classes, CameraRole.SINGLE
        )
        self.assertIn(
            "seq_warm_clean_order",
            {item.observation_id for item in observations},
        )

    def test_geometry_fallback_requires_source_motion_and_hand_contact(self):
        observation = WarmCleanSequenceObservation(
            stable_seconds=0.5, min_samples=3, min_cups=1
        )
        body = item("盖碗碗身", (500, 400), size=(100, 80))
        far_kettle = item("烧水壶", (220, 420), size=(120, 120))
        observation.update(context(
            0.0,
            [far_kettle, body],
            [hand((220, 420))],
            role=CameraRole.SIDE,
            model_classes=self.classes,
        ))
        near_kettle = item("烧水壶", (440, 310), size=(120, 120))
        for timestamp in (0.25, 0.5, 0.75):
            snapshot, _ = observation.update(context(
                timestamp,
                [near_kettle, body],
                [hand((440, 310))],
                role=CameraRole.SIDE,
                model_classes=self.classes,
            ))
        self.assertEqual(snapshot.state, ObservationState.ACTIVE)
        self.assertEqual(snapshot.metrics["completed_targets"], ["盖碗"])
        self.assertEqual(snapshot.metrics["expected_target"], "公道杯")

    def test_static_close_utensils_do_not_trigger_geometry_fallback(self):
        observation = WarmCleanSequenceObservation(
            stable_seconds=0.5, min_samples=3
        )
        body = item("盖碗碗身", (500, 400), size=(100, 80))
        kettle = item("烧水壶", (440, 310), size=(120, 120))
        for timestamp in (0.0, 0.25, 0.5, 0.75):
            snapshot, events = observation.update(context(
                timestamp,
                [kettle, body],
                [hand((440, 310))],
                role=CameraRole.SIDE,
                model_classes=self.classes,
            ))
        self.assertEqual(snapshot.state, ObservationState.IDLE)
        self.assertFalse(events)


class LidOpenSmellTests(unittest.TestCase):
    classes = {"盖碗碗身", "盖碗碗盖"}

    def test_missing_part_capability_is_uncertain(self):
        observation = LidOpenSmellObservation()
        snapshot, events = observation.update(context(0, role=CameraRole.SIDE))
        self.assertEqual(snapshot.state, ObservationState.UNCERTAIN)
        self.assertIn("不能区分", snapshot.reason)
        self.assertFalse(events)

    def test_closed_open_near_closed_sequence_completes(self):
        observation = LidOpenSmellObservation(smell_seconds=0.5)
        body = item("盖碗碗身", (450, 290), size=(100, 80))
        closed_lid = item("盖碗碗盖", (450, 270), size=(80, 40))
        open_lid = item("盖碗碗盖", (570, 270), size=(80, 40))
        base = {"role": CameraRole.SIDE, "model_classes": self.classes, "capabilities": {"gaiwan_parts"}}

        observation.update(context(0.0, [body, closed_lid], poses=[pose()], **base))
        observation.update(context(1.0, [body, open_lid], poses=[pose()], **base))
        observation.update(context(1.2, [body, open_lid], poses=[pose()], **base))
        observation.update(context(1.8, [body, open_lid], poses=[pose()], **base))
        snapshot, events = observation.update(context(2.0, [body, closed_lid], poses=[pose()], **base))

        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual([event.phase for event in events], [EventPhase.COMPLETED])
        self.assertEqual(events[0].observation_id, "action_open_lid_smell")


class BrewPartialObservationTests(unittest.TestCase):
    classes = {"盖碗碗身", "盖碗碗盖", "公道杯"}

    def test_lid_closure_requires_open_then_closed(self):
        observation = GaiwanLidClosureObservation(
            open_seconds=0.4, close_seconds=0.5, min_samples=3
        )
        body = item("盖碗碗身", (450, 400), size=(100, 80))
        open_lid = item("盖碗碗盖", (600, 400), size=(80, 30))
        closed_lid = item("盖碗碗盖", (450, 380), size=(80, 30))
        for timestamp in (0.0, 0.2, 0.4):
            snapshot, _ = observation.update(context(
                timestamp, [body, open_lid], role=CameraRole.SINGLE,
                model_classes=self.classes,
            ))
        self.assertEqual(snapshot.state, ObservationState.ACTIVE)
        for timestamp in (0.6, 0.85, 1.1):
            snapshot, events = observation.update(context(
                timestamp, [body, closed_lid], role=CameraRole.SINGLE,
                model_classes=self.classes,
            ))
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual([event.phase for event in events], [EventPhase.COMPLETED])

    def test_static_closed_lid_does_not_trigger(self):
        observation = GaiwanLidClosureObservation(min_samples=3)
        body = item("盖碗碗身", (450, 400), size=(100, 80))
        lid = item("盖碗碗盖", (450, 380), size=(80, 30))
        for timestamp in (0.0, 0.2, 0.4, 0.6):
            snapshot, events = observation.update(context(
                timestamp, [body, lid], role=CameraRole.SINGLE,
                model_classes=self.classes,
            ))
        self.assertEqual(snapshot.state, ObservationState.IDLE)
        self.assertFalse(events)

    def test_gaiwan_to_pitcher_geometry_completes_after_lift(self):
        observation = GaiwanToPitcherObservation(stable_seconds=0.5, min_samples=3)
        pitcher = item("公道杯", (650, 450), size=(120, 120))
        resting = item("盖碗碗身", (350, 450), size=(100, 80))
        for timestamp in (0.0, 0.2, 0.4, 0.6, 0.8):
            observation.update(context(
                timestamp, [resting, pitcher], role=CameraRole.SINGLE,
                model_classes=self.classes,
            ))
        lifted = item("盖碗碗身", (580, 330), size=(100, 80))
        for timestamp in (1.0, 1.25, 1.5):
            snapshot, events = observation.update(context(
                timestamp, [lifted, pitcher], [hand((580, 330))],
                role=CameraRole.SINGLE, model_classes=self.classes,
            ))
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertFalse(snapshot.metrics["liquid_verified"])
        self.assertEqual([event.phase for event in events], [EventPhase.COMPLETED])

    @staticmethod
    def brew_event(observation_id, phase, start, end, confidence=0.9):
        return ObservationEvent(
            observation_id=observation_id,
            name=observation_id,
            sop_step=5,
            phase=phase,
            start_time=start,
            end_time=end,
            confidence=confidence,
            camera_role=CameraRole.SINGLE.value,
        )

    def test_brew_timer_accepts_ten_seconds_and_marks_partial(self):
        observation = BrewWaitTimerObservation(minimum_wait_seconds=10.0)
        events = [
            self.brew_event("action_gaiwan_lid_close_brew", EventPhase.COMPLETED, 0.5, 1.0),
            self.brew_event("action_gaiwan_to_pitcher", EventPhase.COMPLETED, 11.2, 11.5),
        ]
        snapshot, emitted = observation.update(context(
            11.5, role=CameraRole.SINGLE, extras={"frame_observation_events": events}
        ))
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertGreaterEqual(snapshot.metrics["wait_seconds"], 10.0)
        self.assertFalse(snapshot.metrics["injection_verified"])
        self.assertEqual([event.phase for event in emitted], [EventPhase.COMPLETED])

    def test_brew_timer_rejects_early_decant(self):
        observation = BrewWaitTimerObservation(minimum_wait_seconds=10.0)
        events = [
            self.brew_event("action_gaiwan_lid_close_brew", EventPhase.COMPLETED, 0.5, 1.0),
            self.brew_event("action_gaiwan_to_pitcher", EventPhase.COMPLETED, 5.0, 5.3),
        ]
        snapshot, emitted = observation.update(context(
            5.3, role=CameraRole.SINGLE, extras={"frame_observation_events": events}
        ))
        self.assertEqual(snapshot.state, ObservationState.FAILED)
        self.assertEqual([event.phase for event in emitted], [EventPhase.FAILED])

    def test_formal_brew_timer_waits_for_injection_event(self):
        observation = BrewWaitTimerObservation(
            minimum_wait_seconds=10.0, require_injection=True
        )
        lid = self.brew_event(
            "action_gaiwan_lid_close_brew", EventPhase.COMPLETED, 0.5, 1.0
        )
        snapshot, _ = observation.update(context(
            1.0, role=CameraRole.SINGLE,
            extras={"frame_observation_events": [lid]},
        ))
        self.assertEqual(snapshot.state, ObservationState.UNCERTAIN)
        self.assertFalse(snapshot.metrics["injection_verified"])


class AccessoryAndEngineTests(unittest.TestCase):
    def test_three_positive_windows_report_accessory(self):
        observation = HandAccessoryObservation()
        hands = [hand((300, 300)), hand((500, 300))]
        events = []
        for index, positive in enumerate((True, False, True, False, True)):
            rows = [{"class_name": "戒指", "confidence": 0.9}] if positive else []
            snapshot, emitted = observation.update(context(
                float(index), hands=hands, poses=[pose()], role=CameraRole.SIDE,
                extras={"accessory_detector_configured": True, "accessory_detections": rows},
            ))
            events.extend(emitted)
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertTrue(snapshot.value["present"])
        self.assertEqual(len(events), 1)

    def test_engine_routes_by_camera_and_state_machine_accepts_event(self):
        layout = CupLayoutObservation(stable_seconds=1.0, min_samples=5)
        engine = ObservationEngine([layout])
        machine = SopStateMachine([
            SopStepConfig("layout", "result_cup_layout")
        ])
        cups = [item("品茗杯", (300 + index * 100, 300)) for index in range(3)]
        emitted = []
        for timestamp in (0.0, 0.25, 0.5, 0.75, 1.0):
            _, new_events = engine.process(context(timestamp, cups))
            emitted.extend(new_events)
        for event in emitted:
            machine.process_event(event)
        self.assertEqual(machine.get_step_state("layout").status, StepStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
