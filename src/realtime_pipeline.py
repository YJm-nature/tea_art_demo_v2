"""
实时处理管线 — 单帧检测、匹配、记忆、评分和观测点路由。
"""

from dataclasses import dataclass, field
from copy import copy
from typing import Any, Dict, List, Optional
import time

import numpy as np

from .detection_memory import DetectionMemory
from .item_matcher import ItemMatcher
from .object_tracker import ByteTrackAdapter
from .observation_point import OBSERVATION_REGISTRY
from .observation_runtime import CameraRole, FrameContext, ObservationEngine
from .scoring import ScoreReport, ScoringEngine
from .sop_scoring import SopScoreLedger
from .tea_detector import DetectedItem, TeaDetector


@dataclass
class FrameResult:
    frame_idx: int
    detections: List[DetectedItem]
    matched_items: List[DetectedItem]
    hand_results: List[dict]
    pose_results: List[dict]
    hand_bboxes: List[tuple]
    arm_bboxes: List[tuple]
    checklist: Dict[str, dict]
    essential_found: int
    total_essential: int
    item_score: float
    placement_score: float
    normality_score: float
    weighted_score: float
    grade: str
    grade_color: str
    accessory_detections: List[dict] = field(default_factory=list)
    vessel_pose_results: List[dict] = field(default_factory=list)
    ocr_measurements: Dict[str, dict] = field(default_factory=dict)
    observation_results: Dict[str, Any] = field(default_factory=dict)
    new_events: List[Any] = field(default_factory=list)
    sop_state: Optional[Dict[str, Any]] = None
    sop_score_data: Optional[Dict[str, Any]] = None
    observation_result: Optional[Any] = None
    score_report: Optional[ScoreReport] = None


class RealtimePipeline:
    """完整实时检测管线。"""

    def __init__(
        self,
        detector: TeaDetector,
        matcher: ItemMatcher,
        memory: DetectionMemory,
        tracker: Optional[ByteTrackAdapter] = None,
        hand_detector: Optional[object] = None,
        pose_detector: Optional[object] = None,
        accessory_detector: Optional[object] = None,
        vessel_pose_detector: Optional[object] = None,
        pour_analyzer: Optional[object] = None,
        display_ocr: Optional[object] = None,
        observation_engine: Optional[ObservationEngine] = None,
        state_machine: Optional[object] = None,
        camera_role: CameraRole | str = CameraRole.TABLETOP,
        model_version: str = "unknown",
        model_classes: Optional[List[str]] = None,
        tracking_enabled: bool = True,
        recent_evidence_frames: int = 90,
    ):
        self.detector = detector
        self.matcher = matcher
        self.memory = memory
        self.tracker = tracker
        self.hand_detector = hand_detector
        self.pose_detector = pose_detector
        self.accessory_detector = accessory_detector
        self.vessel_pose_detector = vessel_pose_detector
        self.pour_analyzer = pour_analyzer
        self.display_ocr = display_ocr
        self.observation_engine = observation_engine
        self.state_machine = state_machine
        self.camera_role = CameraRole(camera_role)
        self.model_version = model_version
        self.model_classes = set(model_classes or [])
        self.tracking_enabled = tracking_enabled
        self.recent_evidence_frames = recent_evidence_frames
        self.sop_score_ledger = SopScoreLedger()
        self._active_observation_id: Optional[str] = None

    def set_tracking_enabled(self, enabled: bool):
        self.tracking_enabled = enabled

    def set_confidence(self, conf: float):
        self.detector.conf = conf
        if self.tracker is not None:
            self.tracker.set_thresholds(conf=conf)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: Optional[float] = None,
        source_frame: Optional[np.ndarray] = None,
    ) -> FrameResult:
        frame_timestamp = time.monotonic() if timestamp is None else float(timestamp)
        if self.tracking_enabled and self.tracker is not None:
            detections = self.tracker.track(frame)
        else:
            detections = self.detector.detect(frame)

        # ItemMatcher may rewrite item names for preparation scoring. Temporal
        # observations need the detector's original fine-grained classes.
        action_detections = [copy(item) for item in detections]
        matched = self.matcher.match(detections, frame.shape[:2])
        placement_score = self.matcher.get_placement_score(matched, frame.shape[:2])

        pose_results = self.pose_detector.detect(frame) if self.pose_detector else []
        arm_bboxes = []
        for pose in pose_results:
            arm_bboxes.extend(pose.get("arm_bboxes", []))

        hand_rois = self._pose_hand_rois(pose_results, frame.shape[:2])
        hand_results = (
            self.hand_detector.detect(frame, roi_bboxes=hand_rois)
            if self.hand_detector else []
        )
        hand_bboxes = [h["bbox"] for h in hand_results]

        accessory_detections: List[dict] = []
        if self.accessory_detector is not None and self.camera_role in {
            CameraRole.FRONT, CameraRole.SIDE, CameraRole.SINGLE
        }:
            accessory_detections = self.accessory_detector.detect(frame, hand_results, frame_idx)

        vessel_pose_results = (
            self.vessel_pose_detector.detect(frame, reference_detections=action_detections)
            if self.vessel_pose_detector is not None else []
        )
        pour_interactions = (
            self.pour_analyzer.update(
                action_detections, hand_results, vessel_pose_results, frame_timestamp
            )
            if self.pour_analyzer is not None else []
        )
        ocr_measurements = (
            self.display_ocr.process(
                source_frame if source_frame is not None else frame,
                action_detections,
                frame.shape[:2],
            )
            if self.display_ocr is not None else {}
        )

        self.memory.accumulate(
            matched,
            frame_idx,
            hand_bboxes=hand_bboxes,
            arm_bboxes=arm_bboxes,
        )
        checklist = self.memory.get_checklist(
            self.matcher.items_config,
            current_frame=frame_idx,
            max_age_frames=self.recent_evidence_frames,
        )
        ess, tot, item_score = self.matcher.compute_score(checklist)
        grade, color = self.matcher.get_verdict(ess, tot)
        normality_score = self.matcher.get_area_normality_score(
            matched,
            occluded_count=self.memory.occluded_count,
        )
        score_report = ScoringEngine.evaluate_preparation_step(
            checklist=checklist,
            supported_items=self.matcher.all_item_names,
            placement_score=placement_score,
        )
        weighted_score = score_report.score
        grade, color = score_report.grade, score_report.grade_color

        observation_result = None
        obs = OBSERVATION_REGISTRY.get("obj_utensils_s1")
        if obs is not None:
            observation_result = obs.detect(frame, context={
                "checklist": checklist,
                "essential_found": ess,
                "total_essential": tot,
                "score": item_score,
                "grade": grade,
                "grade_color": color,
                "placement_score": placement_score,
                "normality_score": normality_score,
            })

        observation_results: Dict[str, Any] = {}
        new_events: List[Any] = []
        sop_state = None
        if self.observation_engine is not None:
            active_observation_id = None
            if self.state_machine is not None:
                current_config = self.state_machine.current_step_config
                if current_config is not None:
                    active_observation_id = current_config.observation_id
            if active_observation_id != self._active_observation_id:
                entering_brew_timer = bool(
                    active_observation_id is not None
                    and active_observation_id.startswith("result_brew_time")
                )
                leaving_brew_for_decant = bool(
                    active_observation_id == "action_gaiwan_to_pitcher"
                    and self._active_observation_id is not None
                    and self._active_observation_id.startswith("result_brew_time")
                )
                if entering_brew_timer:
                    # Arm the following decant observer before the wait starts.
                    # It then learns the table position and only accepts a new
                    # grip after the lid-closing hand has first moved away.
                    self.observation_engine.prepare_observation(
                        "action_gaiwan_to_pitcher", "arm_for_brew"
                    )
                elif active_observation_id is not None and not leaving_brew_for_decant:
                    self.observation_engine.reset_observation(
                        active_observation_id
                    )
                self._active_observation_id = active_observation_id
            capabilities = set()
            body_names = {"盖碗碗身", "盖碗（碗身）"}
            lid_names = {"盖碗碗盖", "盖碗（碗盖）"}
            if body_names & self.model_classes and lid_names & self.model_classes:
                capabilities.add("gaiwan_parts")
            accessory_configured = bool(
                self.accessory_detector is not None
                and getattr(self.accessory_detector, "configured", False)
            )
            if accessory_configured:
                capabilities.add("hand_accessory_detector")
            if self.vessel_pose_detector is not None:
                capabilities.add("vessel_pose")
            if self.display_ocr is not None:
                capabilities.add("display_ocr")
            frame_context = FrameContext(
                frame_idx=frame_idx,
                timestamp=frame_timestamp,
                camera_role=self.camera_role,
                frame_shape=frame.shape[:2],
                detections=action_detections,
                hand_results=hand_results,
                pose_results=pose_results,
                model_version=self.model_version,
                model_classes=set(self.model_classes),
                capabilities=capabilities,
                extras={
                    "accessory_detector_configured": accessory_configured,
                    "accessory_detections": accessory_detections,
                    "vessel_pose_results": vessel_pose_results,
                    "pour_interactions": pour_interactions,
                    "ocr_measurements": ocr_measurements,
                },
            )
            observation_results, new_events = self.observation_engine.process(frame_context)
            self.sop_score_ledger.consume(new_events)
            if self.state_machine is not None:
                self.state_machine.tick(frame_timestamp)
                for event in new_events:
                    self.state_machine.process_event(event)
                sop_state = self.state_machine.to_dict()

        return FrameResult(
            frame_idx=frame_idx,
            # Keep detector ontology names for visualization. ItemMatcher
            # rewrites 盖碗碗身 to the legacy scoring alias 盖碗 in matched.
            detections=action_detections,
            matched_items=matched,
            hand_results=hand_results,
            pose_results=pose_results,
            hand_bboxes=hand_bboxes,
            arm_bboxes=arm_bboxes,
            checklist=checklist,
            essential_found=ess,
            total_essential=tot,
            item_score=item_score,
            placement_score=placement_score,
            normality_score=normality_score,
            weighted_score=weighted_score,
            grade=grade,
            grade_color=color,
            accessory_detections=accessory_detections,
            vessel_pose_results=vessel_pose_results,
            ocr_measurements=ocr_measurements,
            observation_results=observation_results,
            new_events=new_events,
            sop_state=sop_state,
            sop_score_data=self.sop_score_ledger.to_dict(),
            observation_result=observation_result,
            score_report=score_report,
        )

    def reset(self) -> None:
        self.memory.reset()
        if self.tracker is not None:
            self.tracker.reset()
        if self.observation_engine is not None:
            self.observation_engine.reset()
        if self.state_machine is not None and hasattr(self.state_machine, "reset"):
            self.state_machine.reset()
        if self.pour_analyzer is not None:
            self.pour_analyzer.reset()
        if self.display_ocr is not None:
            self.display_ocr.reset()
        self.sop_score_ledger.reset()
        self._active_observation_id = None

    @staticmethod
    def _pose_hand_rois(pose_results: List[dict], frame_shape) -> List[tuple]:
        if not pose_results:
            return []
        frame_h, frame_w = frame_shape
        landmarks = np.asarray(pose_results[0].get("landmarks", []))
        if len(landmarks) < 17:
            return []
        shoulder_width = float(np.linalg.norm(landmarks[11, :2] - landmarks[12, :2]))
        roi_size = int(max(96, shoulder_width * 0.7))
        rois = []
        for wrist_index in (15, 16):
            cx, cy = landmarks[wrist_index, :2]
            x1 = max(0, int(cx - roi_size / 2))
            y1 = max(0, int(cy - roi_size / 2))
            x2 = min(frame_w, x1 + roi_size)
            y2 = min(frame_h, y1 + roi_size)
            rois.append((x1, y1, x2 - x1, y2 - y1))
        return rois
