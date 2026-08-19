import unittest
from pathlib import Path

import numpy as np

from src.observation_runtime import CameraRole, EventPhase, FrameContext, ObservationState
from src.observations.sop_v2 import (
    LotusAppreciationObservation,
    RelaxedLidClosureObservation,
    ReturnAwareDecantObservation,
    ReturnAwareDistributionObservation,
    SetupReadyObservation,
    SimpleWaterInjectionObservation,
    SmellObservation,
    TeaPreparationObservation,
)
from src.tea_detector import DetectedItem
from src.sop_config import build_sop_steps, load_sop_config


def item(name, center, size=(80, 80), track_id=None):
    width, height = size
    return DetectedItem(
        bbox=(int(center[0] - width / 2), int(center[1] - height / 2), width, height),
        centroid=center,
        contour_area=width * height,
        confidence=0.9,
        item_name=name,
        track_id=track_id,
    )


def hand(center):
    return {
        "center": center,
        "bbox": (int(center[0] - 25), int(center[1] - 25), 50, 50),
        "confidence": 0.9,
    }


def pose(nose=(500, 180)):
    landmarks = np.zeros((33, 3), dtype=np.float32)
    landmarks[0, :2] = nose
    landmarks[11, :2] = (420, 220)
    landmarks[12, :2] = (580, 220)
    landmarks[23, :2] = (440, 430)
    landmarks[24, :2] = (560, 430)
    return {"landmarks": landmarks, "visibility": np.ones(33, dtype=np.float32)}


def context(timestamp, detections=(), hands=(), poses=(), extras=None):
    return FrameContext(
        frame_idx=int(timestamp * 10) + 1,
        timestamp=float(timestamp),
        camera_role=CameraRole.FRONT,
        frame_shape=(720, 1280),
        detections=list(detections),
        hand_results=list(hands),
        pose_results=list(poses),
        model_version="test",
        model_classes={
            "盖碗碗身", "盖碗碗盖", "公道杯", "品茗杯", "茶荷", "茶巾",
            "茶夹", "茶拨", "烧水壶", "建水", "茶叶罐",
        },
        capabilities={"gaiwan_parts", "vessel_pose"},
        extras=extras or {},
    )


class SixStepRulesTests(unittest.TestCase):
    def test_action_config_contains_six_business_steps_and_twelve_nodes(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "sop_red_tea_front_action_test_v1.yaml"
        )
        config = load_sop_config(config_path)
        steps = build_sop_steps(config, include_deferred=True)
        self.assertEqual(len(config["steps"]), 6)
        self.assertEqual(len(steps), 12)
        self.assertEqual(
            [row["name"] for row in config["steps"]],
            ["备具布席", "温杯洁具", "投茶准备", "投茶闻香", "注水冲泡", "分茶与奉茶"],
        )

    def test_setup_requires_utensils_and_upright_person(self):
        observer = SetupReadyObservation(stable_seconds=0.5)
        detections = [
            item("盖碗碗身", (100, 300)), item("盖碗碗盖", (130, 280)),
            item("公道杯", (220, 300)), item("茶荷", (320, 300)),
            item("茶巾", (400, 400)), item("茶夹", (500, 400)),
            item("茶拨", (600, 400)), item("烧水壶", (700, 300)),
            item("建水", (800, 300)), item("茶叶罐", (900, 300)),
            item("品茗杯", (300, 500)), item("品茗杯", (400, 500)),
            item("品茗杯", (500, 500)),
        ]
        events = []
        for timestamp in (0.0, 0.25, 0.55):
            snapshot, emitted = observer.update(context(timestamp, detections, poses=[pose()]))
            events.extend(emitted)
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual(events[-1].phase, EventPhase.COMPLETED)

    def test_tea_preparation_accepts_near_canister_and_requires_return(self):
        observer = TeaPreparationObservation(stable_seconds=0.25)
        lotus = item("茶荷", (500, 350), (120, 70))
        far_canister = item("茶叶罐", (250, 350))
        near_canister = item("茶叶罐", (430, 350))
        far_pick = item("茶拨", (720, 380), (140, 30))
        near_pick = item("茶拨", (470, 360), (140, 30))
        sequence = [
            (0.0, far_canister, far_pick, [hand((250, 350))]),
            (0.3, far_canister, far_pick, [hand((250, 350))]),
            (0.6, near_canister, far_pick, [hand((430, 350))]),
            (0.9, near_canister, far_pick, [hand((430, 350))]),
            (1.2, near_canister, near_pick, [hand((470, 360))]),
            (1.5, near_canister, near_pick, [hand((470, 360))]),
            (1.8, far_canister, far_pick, []),
            (2.3, far_canister, far_pick, []),
        ]
        events = []
        for timestamp, canister, tea_pick, hands in sequence:
            snapshot, emitted = observer.update(
                context(timestamp, [canister, lotus, tea_pick], hands=hands)
            )
            events.extend(emitted)
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertNotIn("距离过近", snapshot.reason)
        self.assertEqual(events[-1].phase, EventPhase.COMPLETED)

    def test_lotus_appreciation_uses_hand_boxes_and_direction_not_chest(self):
        observer = LotusAppreciationObservation()
        rest = item("茶荷", (330, 500), (140, 70))
        for timestamp in (0.0, 0.1, 0.2, 0.3, 0.4):
            observer.update(context(timestamp, [rest]))
        def held(timestamp, x):
            lotus = item("茶荷", (x, 420), (140, 70))
            return observer.update(context(
                timestamp, [lotus], hands=[hand((x - 70, 420)), hand((x + 70, 420))]
            ))
        held(0.6, 340)
        held(0.9, 420)
        held(1.1, 420)
        held(1.3, 380)
        snapshot, events = held(1.6, 375)
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertFalse(snapshot.metrics["chest_height_required"])
        self.assertEqual(events[-1].phase, EventPhase.COMPLETED)

    def test_smell_uses_mediapipe_nose_and_tolerates_part_occlusion(self):
        observer = SmellObservation()
        body = item("盖碗碗身", (500, 330), (100, 80))
        open_lid = item("盖碗碗盖", (650, 330), (90, 50))
        observer.update(context(0.0, [body, open_lid], poses=[pose()]))
        observer.update(context(0.2, [], hands=[hand((500, 190))], poses=[pose()]))
        observer.update(context(0.8, [], hands=[hand((500, 190))], poses=[pose()]))
        observer.update(context(1.0, [], hands=[hand((800, 500))], poses=[pose()]))
        snapshot, events = observer.update(
            context(1.4, [], hands=[hand((800, 500))], poses=[pose()])
        )
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual(snapshot.metrics["nose_source"], "mediapipe_pose_index_0")
        self.assertEqual(events[-1].phase, EventPhase.COMPLETED)

    def test_smell_can_use_hand_motion_when_lid_is_occluded(self):
        observer = SmellObservation()
        body = item("盖碗碗身", (500, 330), (100, 80))
        lid = item("盖碗碗盖", (505, 320), (90, 50))
        observer.update(
            context(0.0, [body, lid], hands=[hand((500, 330))], poses=[pose()])
        )
        observer.update(
            context(0.2, [body], hands=[hand((500, 180))], poses=[pose()])
        )
        observer.update(
            context(0.8, [body], hands=[hand((500, 180))], poses=[pose()])
        )
        observer.update(
            context(1.1, [body], hands=[hand((800, 500))], poses=[pose()])
        )
        snapshot, events = observer.update(
            context(1.5, [body], hands=[hand((800, 500))], poses=[pose()])
        )
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual(snapshot.metrics["open_signal"], "hand_from_gaiwan_to_nose")
        self.assertEqual(events[-1].phase, EventPhase.COMPLETED)

    def test_lid_closure_does_not_require_reobserving_open_state(self):
        observer = RelaxedLidClosureObservation()
        body = item("盖碗碗身", (500, 400), (120, 90))
        lid = item("盖碗碗盖", (510, 390), (100, 50))
        observer.update(context(0.0, [body, lid]))
        snapshot, events = observer.update(context(0.5, [body, lid]))
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertFalse(snapshot.metrics["open_prerequisite_required"])
        self.assertEqual(events[-1].phase, EventPhase.COMPLETED)

    def test_lid_closure_falls_back_to_stable_body_after_hand_release(self):
        observer = RelaxedLidClosureObservation()
        body = item("盖碗碗身", (500, 400), (120, 90))
        observer.update(context(0.0, [body], hands=[hand((500, 400))]))
        observer.update(context(0.4, [body]))
        observer.update(context(0.8, [body]))
        snapshot, events = observer.update(context(1.3, [body]))
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual(snapshot.metrics["closure_signal"], "stable_body_after_hand_release")
        self.assertEqual(events[-1].phase, EventPhase.COMPLETED)

    def test_injection_needs_stable_pose_interaction_but_no_orbit(self):
        observer = SimpleWaterInjectionObservation()
        interaction = {
            "source": "烧水壶", "target": "盖碗碗身", "confidence": 0.8,
            "outlet_point": [500, 350], "tilt_delta_degrees": 10,
        }
        observer.update(context(0.0, extras={"pour_interactions": [interaction]}))
        snapshot, events = observer.update(
            context(0.6, extras={"pour_interactions": [interaction]})
        )
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertFalse(snapshot.metrics["orbit_required"])
        self.assertEqual(events[-1].phase, EventPhase.COMPLETED)

    def test_decant_starts_on_grip_and_completes_after_return(self):
        observer = ReturnAwareDecantObservation()
        pitcher = item("公道杯", (600, 420), (100, 120))
        rest_body = item("盖碗碗身", (350, 420), (110, 90))
        snapshot, events = observer.update(
            context(0.0, [rest_body, pitcher], hands=[hand((350, 420))])
        )
        self.assertEqual(events[0].phase, EventPhase.STARTED)
        pour_body = item("盖碗碗身", (540, 350), (110, 90))
        interaction = {"source": "盖碗碗身", "target": "公道杯", "confidence": 0.8}
        observer.update(context(0.2, [pour_body, pitcher], hands=[hand((540, 350))], extras={"pour_interactions": [interaction]}))
        observer.update(context(0.8, [pour_body, pitcher], hands=[hand((540, 350))], extras={"pour_interactions": [interaction]}))
        observer.update(context(1.0, [rest_body, pitcher]))
        snapshot, events = observer.update(context(1.5, [rest_body, pitcher]))
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual(events[-1].phase, EventPhase.COMPLETED)

    def test_decant_armed_for_brew_requires_release_before_new_grip(self):
        observer = ReturnAwareDecantObservation()
        observer.arm_for_brew()
        pitcher = item("公道杯", (600, 420), (100, 120))
        body = item("盖碗碗身", (350, 420), (110, 90))
        _, events = observer.update(
            context(0.0, [body, pitcher], hands=[hand((350, 420))])
        )
        self.assertEqual(events, [])
        observer.update(context(0.2, [body, pitcher], hands=[]))
        _, events = observer.update(
            context(8.5, [body, pitcher], hands=[hand((350, 420))])
        )
        self.assertEqual(events[-1].phase, EventPhase.STARTED)

    def test_distribution_requires_three_cups_and_pitcher_return(self):
        observer = ReturnAwareDistributionObservation()
        pitcher = item("公道杯", (250, 400), (100, 120))
        cups = [item("品茗杯", (x, 500), track_id=i) for i, x in enumerate((450, 550, 650), 1)]
        observer.update(context(0.0, [pitcher, *cups], hands=[hand((250, 400))]))
        timestamp = 0.2
        for cup in cups:
            interaction = {
                "source": "公道杯", "target": "品茗杯", "confidence": 0.8,
                "target_track_id": cup.track_id, "target_center": list(cup.centroid),
            }
            observer.update(context(timestamp, [pitcher, *cups], hands=[hand((250, 400))], extras={"pour_interactions": [interaction]}))
            timestamp += 0.5
            observer.update(context(timestamp, [pitcher, *cups], hands=[hand((250, 400))], extras={"pour_interactions": [interaction]}))
            timestamp += 0.2
        observer.update(context(timestamp, [pitcher, *cups]))
        snapshot, events = observer.update(context(timestamp + 0.5, [pitcher, *cups]))
        self.assertEqual(snapshot.state, ObservationState.COMPLETED)
        self.assertEqual(snapshot.metrics["target_count"], 3)
        self.assertEqual(events[-1].phase, EventPhase.COMPLETED)


if __name__ == "__main__":
    unittest.main()
