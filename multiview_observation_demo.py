"""Sequential dual-camera observation demo for a 6 GB GPU."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np

from src.accessory_detector import HandAccessoryDetector
from src.observation_catalog import build_default_observations
from src.capture_source import open_capture
from src.detection_memory import DetectionMemory
from src.draw_utils import (
    draw_accessory_detections,
    draw_detections,
    draw_hand_skeleton,
    draw_observation_panel,
    draw_pose_skeleton,
)
from src.hand_detector import HandDetector
from src.item_matcher import ItemMatcher
from src.model_config import load_yolo_with_profile
from src.object_tracker import ByteTrackAdapter
from src.observation_runtime import CameraRole, ObservationEngine
from src.pose_detector import PoseDetector
from src.realtime_pipeline import RealtimePipeline
from src.sop_runtime import DEFAULT_SOP_CONFIG, build_sop_state_machine
from src.sop_state_machine import SopStateMachine
from src.tea_detector import TeaDetector


WINDOW_NAME = "Tea SOP Dual View Observations"


def parse_args():
    parser = argparse.ArgumentParser(description="桌面+正侧面串行动作观测")
    parser.add_argument("--table-video", help="桌面机位视频；不提供时使用摄像头")
    parser.add_argument("--side-video", help="正侧面机位视频；不提供时使用摄像头")
    parser.add_argument("--table-camera", type=int, default=0)
    parser.add_argument("--side-camera", type=int, default=1)
    parser.add_argument("--model", default=None)
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--accessory-model", default=None)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--process-every", type=int, default=2)
    parser.add_argument("--sync-tolerance-ms", type=float, default=150.0)
    parser.add_argument("--no-track", action="store_true")
    parser.add_argument("--no-dshow", action="store_true")
    parser.add_argument("--report-dir", default="output/reports")
    parser.add_argument("--headless", action="store_true", help="不打开窗口，用于离线批处理和自动测试")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理的同步帧对；0表示不限制")
    parser.add_argument("--observation-mode", choices=["free_observation", "strict"], default="free_observation")
    parser.add_argument("--sop-config", default=str(DEFAULT_SOP_CONFIG), help="红茶SOP YAML配置文件")
    return parser.parse_args()


def build_state_machine(
    observation_ids, mode="free_observation", config_path=DEFAULT_SOP_CONFIG
) -> SopStateMachine:
    machine = build_sop_state_machine(
        config_path=config_path,
        mode=mode,
        available_observation_ids=observation_ids,
    )
    assert machine is not None
    return machine


def build_pipeline(
    args,
    role: CameraRole,
    engine: ObservationEngine,
    state_machine: SopStateMachine,
):
    loaded = load_yolo_with_profile(args.model, requested_profile=args.profile, strict=True)
    detector = TeaDetector(
        use_yolo=True,
        model_path=loaded.model_path,
        class_names=loaded.profile.class_names,
        yolo_model=loaded.yolo_model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        active_class_ids=loaded.profile.active_class_ids,
    )
    tracker = ByteTrackAdapter(
        loaded.yolo_model,
        loaded.profile.class_names,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        active_class_ids=loaded.profile.active_class_ids,
    )
    matcher = ItemMatcher(
        supported_items=loaded.profile.scoring_items,
        item_aliases=loaded.profile.item_aliases,
    )
    hand_detector = HandDetector(detect_every_n_frames=3) if role is CameraRole.SIDE else None
    pose_detector = PoseDetector(detect_every_n_frames=3) if role is CameraRole.SIDE else None
    accessory_detector = (
        HandAccessoryDetector(args.accessory_model)
        if role is CameraRole.SIDE and args.accessory_model else None
    )
    pipeline = RealtimePipeline(
        detector=detector,
        tracker=tracker,
        matcher=matcher,
        memory=DetectionMemory(),
        hand_detector=hand_detector,
        pose_detector=pose_detector,
        accessory_detector=accessory_detector,
        observation_engine=engine,
        state_machine=state_machine,
        camera_role=role,
        model_version=str(loaded.model_path),
        model_classes=loaded.profile.active_class_names,
        tracking_enabled=not args.no_track,
    )
    return pipeline, loaded, hand_detector, pose_detector


def open_role_capture(video, camera_id, args):
    return open_capture(
        source="video" if video else "camera",
        camera_id=camera_id,
        video_path=video,
        width=args.width,
        height=args.height,
        use_dshow=not args.no_dshow,
    )


def read_timestamped(cap, info):
    ok, frame = cap.read()
    if not ok:
        return False, None, 0.0
    timestamp = time.monotonic() if info.is_camera else max(0.0, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    return True, frame, timestamp


def align_pair(table, side, tolerance_seconds):
    table_cap, table_info, table_frame, table_ts = table
    side_cap, side_info, side_frame, side_ts = side
    while abs(table_ts - side_ts) > tolerance_seconds:
        if table_ts < side_ts:
            ok, table_frame, table_ts = read_timestamped(table_cap, table_info)
        else:
            ok, side_frame, side_ts = read_timestamped(side_cap, side_info)
        if not ok:
            return None
    return table_frame, table_ts, side_frame, side_ts


def render_view(frame, result, role):
    output = draw_detections(frame, result.matched_items)
    if result.hand_results:
        output = draw_hand_skeleton(output, result.hand_results)
    if result.pose_results:
        output = draw_pose_skeleton(output, result.pose_results)
    if result.accessory_detections:
        output = draw_accessory_detections(output, result.accessory_detections)
    output = draw_observation_panel(output, result.observation_results, role.value)
    return cv2.resize(output, (640, 360))


def save_report(args, engine, state_machine, table_loaded, side_loaded, frame_idx):
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    output = report_dir / f"multiview_{now.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "schema_version": "2.0",
        "generated_at": now.isoformat(),
        "session": {
            "table_source": args.table_video or f"camera:{args.table_camera}",
            "side_source": args.side_video or f"camera:{args.side_camera}",
            "processed_pairs": frame_idx,
            "sync_tolerance_ms": args.sync_tolerance_ms,
            "table_model": str(table_loaded.model_path),
            "side_model": str(side_loaded.model_path),
            "inference": "sequential",
        },
        "observation_snapshots": {
            key: value.to_dict() for key, value in engine.snapshots.items()
        },
        "observation_events": engine.events_as_dicts(),
        "sop_state": state_machine.to_dict(),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main():
    args = parse_args()
    tolerance = max(0.0, args.sync_tolerance_ms / 1000.0)
    engine = ObservationEngine(build_default_observations())
    state_machine = build_state_machine(
        engine.observations.keys(), args.observation_mode, args.sop_config
    )
    table_pipeline, table_loaded, _, _ = build_pipeline(args, CameraRole.TABLETOP, engine, state_machine)
    side_pipeline, side_loaded, side_hands, side_pose = build_pipeline(args, CameraRole.SIDE, engine, state_machine)
    table_cap, table_info = open_role_capture(args.table_video, args.table_camera, args)
    side_cap, side_info = open_role_capture(args.side_video, args.side_camera, args)

    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    frame_idx = 0
    last_table_result = None
    last_side_result = None
    try:
        while True:
            table_ok, table_frame, table_ts = read_timestamped(table_cap, table_info)
            side_ok, side_frame, side_ts = read_timestamped(side_cap, side_info)
            if not table_ok or not side_ok:
                break
            aligned = align_pair(
                (table_cap, table_info, table_frame, table_ts),
                (side_cap, side_info, side_frame, side_ts),
                tolerance,
            )
            if aligned is None:
                break
            table_frame, table_ts, side_frame, side_ts = aligned
            frame_idx += 1
            table_frame = cv2.resize(table_frame, (args.width, args.height))
            side_frame = cv2.resize(side_frame, (args.width, args.height))
            if last_table_result is None or (frame_idx - 1) % max(1, args.process_every) == 0:
                last_table_result = table_pipeline.process_frame(table_frame, frame_idx, table_ts)
                last_side_result = side_pipeline.process_frame(side_frame, frame_idx, side_ts)
                for event in last_table_result.new_events + last_side_result.new_events:
                    print(f"[EVENT] {event.observation_id} {event.phase.value} {event.confidence:.2f}")

            if not args.headless:
                display = np.hstack([
                    render_view(table_frame, last_table_result, CameraRole.TABLETOP),
                    render_view(side_frame, last_side_result, CameraRole.SIDE),
                ])
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("e"):
                    path = save_report(args, engine, state_machine, table_loaded, side_loaded, frame_idx)
                    print(f"观测报告: {path}")
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        table_cap.release()
        side_cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if side_hands is not None:
            side_hands.close()
        if side_pose is not None:
            side_pose.close()
    path = save_report(args, engine, state_machine, table_loaded, side_loaded, frame_idx)
    print(f"观测结束，报告: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
