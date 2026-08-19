"""
茶具实时检测 + 目标跟踪 Demo

默认使用摄像头，集成 YOLO 茶具检测、ByteTrack 目标跟踪、跨帧记忆、
手/姿态遮挡检测与三维评分。
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.capture_source import open_capture
from src.detection_memory import DetectionMemory
from src.draw_utils import (
    draw_accessory_detections,
    draw_controls,
    draw_detections,
    draw_hand_skeleton,
    draw_info_panel,
    draw_observation_panel,
    draw_pose_skeleton,
    draw_step_details_panel,
    draw_sop_panel,
    draw_tracks,
    draw_vessel_pose,
)
from src.hand_detector import HandDetector
from src.accessory_detector import HandAccessoryDetector
from src.observation_catalog import build_available_observations
from src.item_matcher import ItemMatcher
from src.model_config import ModelConfigError, load_yolo_with_profile
from src.object_tracker import ByteTrackAdapter, TrajectoryStore
from src.pose_detector import PoseDetector
from src.realtime_pipeline import RealtimePipeline
from src.observation_runtime import CameraRole, ObservationEngine
from src.sop_runtime import build_sop_state_machine
from src.tea_detector import TeaDetector
from src.display_ocr import DisplayOcrService
from src.vessel_pose import PourInteractionAnalyzer, YoloV8PoseDetector


WINDOW_NAME = "Tea Ware Realtime Detection + Tracking"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT_DIR = PROJECT_ROOT / "output" / "reports"
DEFAULT_ACTION_SOP_CONFIG = (
    PROJECT_ROOT / "config" / "sop_red_tea_front_action_test_v1.yaml"
)


def resolve_project_path(value: str | Path | None) -> str | None:
    """Resolve user-supplied relative paths from the project root, not the shell cwd."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def parse_args():
    parser = argparse.ArgumentParser(description="茶具摄像头实时检测 + ByteTrack 目标跟踪 Demo")
    parser.add_argument("--source", choices=["camera", "video"], default="camera", help="输入源类型")
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头编号")
    parser.add_argument("--video", default=None, help="视频文件路径")
    parser.add_argument("--model", default=None, help="YOLO 权重路径；默认自动查找")
    parser.add_argument(
        "--profile",
        default="auto",
        help="模型类别 profile：auto/tea9/tea13/tea18/tea18_warm_clean/tea18_front",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO 置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU 阈值")
    parser.add_argument("--width", type=int, default=1280, help="推理宽度")
    parser.add_argument("--height", type=int, default=720, help="推理高度")
    parser.add_argument("--imgsz", type=int, default=832, help="YOLO推理尺寸，6GB显存建议832")
    parser.add_argument("--process-every", type=int, default=2, help="每N帧运行一次完整感知，6GB建议2")
    parser.add_argument("--track", dest="track", action="store_true", default=True, help="开启目标跟踪")
    parser.add_argument("--no-track", dest="track", action="store_false", help="关闭目标跟踪")
    parser.add_argument("--no-hand", action="store_true", help="关闭手部检测")
    parser.add_argument("--no-pose", action="store_true", help="关闭姿态检测")
    parser.add_argument("--no-dshow", action="store_true", help="摄像头不使用 Windows DirectShow 后端")
    parser.add_argument("--hand-every", type=int, default=3, help="手部检测帧间隔")
    parser.add_argument("--pose-every", type=int, default=3, help="姿态检测帧间隔")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="评分报告输出目录")
    parser.add_argument(
        "--camera-role",
        choices=["front", "tabletop", "side", "single"],
        default="front",
        help="front为新版固定正面机位；其余选项仅兼容旧数据",
    )
    parser.add_argument("--pose-model", default=None, help="器具YOLOv8n-pose关键点权重")
    parser.add_argument("--pose-imgsz", type=int, default=640, help="关键点模型推理尺寸")
    parser.add_argument("--pose-conf", type=float, default=0.45, help="器具关键点模型置信度阈值")
    parser.add_argument("--paddle-ocr", action="store_true", help="七段识别失败时启用CPU PaddleOCR回退")
    parser.add_argument("--accessory-model", default=None, help="独立手部饰品YOLO权重；未提供时饰品观测为不确定")
    parser.add_argument(
        "--show-observations",
        action="store_true",
        help="显示左上角全部观测器状态；默认仅显示当前SOP动作",
    )
    parser.add_argument(
        "--show-score-panel",
        action="store_true",
        help="显示右上角旧版备具评分面板；动作联调默认隐藏",
    )
    parser.add_argument("--observation-mode", choices=["free_observation", "strict"], default="strict", help="动作观测事件模式")
    parser.add_argument("--sop-config", default=str(DEFAULT_ACTION_SOP_CONFIG), help="红茶SOP YAML配置文件")
    return parser.parse_args()


def save_session_report(result, args, loaded, frame_idx: int, observation_engine, state_machine) -> Path:
    """保存包含运行上下文和评分证据的 JSON 报告。"""
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone()
    output = report_dir / f"observation_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "schema_version": "2.0",
        "generated_at": timestamp.isoformat(),
        "session": {
            "source_type": args.source,
            "source": args.video if args.source == "video" else f"camera:{args.camera_id}",
            "frame_count": frame_idx,
            "model_path": str(loaded.model_path),
            "model_profile": loaded.profile.name,
            "model_classes": loaded.profile.class_names,
            "active_class_ids": loaded.profile.active_class_ids,
            "active_class_names": loaded.profile.active_class_names,
            "confidence_threshold": args.conf,
            "tracking_enabled": args.track,
            "camera_role": args.camera_role,
            "observation_mode": args.observation_mode,
        },
        "score_report": result.score_report.to_dict(),
        "observation_snapshots": {
            key: value.to_dict() for key, value in result.observation_results.items()
        },
        "observation_events": observation_engine.events_as_dicts(),
        "sop_state": state_machine.to_dict() if state_machine is not None else None,
        "sop_score_data": result.sop_score_data,
        "notice": "动作观测与步骤一评分相互独立；实验观测未达到正式验收门槛",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main():
    args = parse_args()
    args.video = resolve_project_path(args.video)
    args.report_dir = resolve_project_path(args.report_dir)
    args.sop_config = resolve_project_path(args.sop_config)
    args.pose_model = resolve_project_path(args.pose_model)
    args.accessory_model = resolve_project_path(args.accessory_model)

    try:
        loaded = load_yolo_with_profile(
            args.model,
            requested_profile=args.profile,
            strict=True,
        )
    except ModelConfigError as exc:
        print(f"❌ 模型配置错误:\n{exc}")
        return 1

    print(f"📦 模型: {loaded.model_path}")
    print(f"🏷 Profile: {loaded.profile.name} ({len(loaded.profile.class_names)}类)")
    print(f"   类别: {', '.join(loaded.profile.class_names)}")

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
    trajectories = TrajectoryStore(max_trail=30)
    matcher = ItemMatcher(
        supported_items=loaded.profile.scoring_items,
        item_aliases=loaded.profile.item_aliases,
    )
    memory = DetectionMemory()
    action_role = args.camera_role in {"front", "side", "single"}
    hand_detector = (
        HandDetector(detect_every_n_frames=max(1, args.hand_every))
        if action_role and not args.no_hand else None
    )
    pose_detector = (
        PoseDetector(detect_every_n_frames=max(1, args.pose_every))
        if action_role and not args.no_pose else None
    )
    accessory_detector = (
        HandAccessoryDetector(args.accessory_model)
        if action_role and args.accessory_model else None
    )
    vessel_pose_detector = (
        YoloV8PoseDetector(
            args.pose_model, conf=args.pose_conf, imgsz=args.pose_imgsz
        )
        if args.pose_model else None
    )
    pour_analyzer = PourInteractionAnalyzer() if vessel_pose_detector else None
    display_ocr = DisplayOcrService(
        enable_paddle_fallback=args.paddle_ocr
    ) if args.camera_role == "front" else None
    observations = build_available_observations(
        loaded.profile.active_class_names,
        CameraRole(args.camera_role),
        accessory_configured=bool(
            accessory_detector is not None
            and getattr(accessory_detector, "configured", False)
        ),
    )
    observation_engine = ObservationEngine(observations)
    available_observation_ids = {item.observation_id for item in observations}
    state_machine = build_sop_state_machine(
        config_path=args.sop_config,
        mode=args.observation_mode,
        available_observation_ids=available_observation_ids,
        allow_empty=True,
    )

    pipeline = RealtimePipeline(
        detector=detector,
        tracker=tracker,
        matcher=matcher,
        memory=memory,
        hand_detector=hand_detector,
        pose_detector=pose_detector,
        accessory_detector=accessory_detector,
        vessel_pose_detector=vessel_pose_detector,
        pour_analyzer=pour_analyzer,
        display_ocr=display_ocr,
        observation_engine=observation_engine,
        state_machine=state_machine,
        camera_role=CameraRole(args.camera_role),
        model_version=str(loaded.model_path),
        model_classes=loaded.profile.active_class_names,
        tracking_enabled=args.track,
    )

    try:
        cap, cap_info = open_capture(
            source=args.source,
            camera_id=args.camera_id,
            video_path=args.video,
            width=args.width,
            height=args.height,
            use_dshow=not args.no_dshow,
        )
    except Exception as exc:
        print(f"❌ 打开视频源失败: {exc}")
        return 1

    print(f"🎥 输入源: {cap_info.source}")
    print(f"✅ 就绪 | 跟踪:{'ON' if args.track else 'OFF'} | CONF:{args.conf:.2f} | IoU:{args.iou:.2f}")
    print("   Q/Esc=退出  Space=暂停  S=截图  T=追踪开关  +/-=调阈值  R=视频重播  N=跳过当前步骤  X=重置SOP")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1400, 900)

    os.makedirs(os.path.join("output", "screenshots"), exist_ok=True)

    frame_idx = 0
    fps_display = 0.0
    paused = False
    tracking_enabled = args.track
    last_annotated = None
    last_result = None

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if cap_info.can_replay:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        pipeline.reset()
                        trajectories.clear()
                        continue
                    break
                frame_idx += 1
            elif last_annotated is None:
                key = cv2.waitKey(30) & 0xFF
                if key in [ord('q'), 27]:
                    break
                continue

            if not paused:
                frame_inf = cv2.resize(frame, (args.width, args.height))
                t0 = time.perf_counter()
                should_process = (
                    last_result is None
                    or (frame_idx - 1) % max(1, args.process_every) == 0
                )
                if should_process:
                    source_timestamp = (
                        time.monotonic() if cap_info.is_camera
                        else max(0.0, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                    )
                    result = pipeline.process_frame(
                        frame_inf, frame_idx, timestamp=source_timestamp,
                        source_frame=frame,
                    )
                    last_result = result
                    for event in result.new_events:
                        print(f"\n[EVENT] {event.observation_id} {event.phase.value} conf={event.confidence:.2f}")
                else:
                    result = last_result
                if tracking_enabled and should_process:
                    trajectories.update(result.matched_items, frame_idx)

                # Draw the model's fine-grained ontology labels. The matched
                # list is reserved for legacy preparation scoring aliases.
                annotated = draw_detections(frame_inf, result.detections)
                annotated = draw_tracks(annotated, trajectories.get_trails())
                if result.hand_results:
                    annotated = draw_hand_skeleton(annotated, result.hand_results)
                if result.pose_results:
                    annotated = draw_pose_skeleton(annotated, result.pose_results)
                if result.vessel_pose_results:
                    annotated = draw_vessel_pose(
                        annotated, result.vessel_pose_results
                    )
                if result.accessory_detections:
                    annotated = draw_accessory_detections(annotated, result.accessory_detections)

                panel = {
                    "detected_count": result.essential_found,
                    "total_count": result.total_essential,
                    "score": result.weighted_score,
                    "grade": result.grade,
                    "grade_color": result.grade_color,
                    "fps": fps_display,
                    "frame_idx": frame_idx,
                    "mode": "TRACK" if tracking_enabled else "DETECT",
                    "profile": loaded.profile.name,
                    "requirement_coverage": result.score_report.requirement_coverage,
                    "evidence_reliability": result.score_report.evidence_reliability,
                }
                if args.show_score_panel:
                    annotated = draw_info_panel(annotated, panel)
                # Strict SOP must remain step-scoped.  The all-observation
                # panel is useful for free-observation debugging, but showing
                # later observers during an active SOP step is misleading.
                if args.show_observations and args.observation_mode != "strict":
                    annotated = draw_observation_panel(
                        annotated, result.observation_results, args.camera_role,
                        position="top_left",
                    )
                else:
                    annotated = draw_step_details_panel(
                        annotated,
                        result.sop_state,
                        result.observation_results,
                    )
                annotated = draw_sop_panel(
                    annotated,
                    result.sop_state,
                    result.observation_results,
                )
                annotated = draw_controls(
                    annotated,
                    paused=paused,
                    tracking_enabled=tracking_enabled,
                    conf=args.conf,
                    show_replay=cap_info.can_replay,
                )
                last_annotated = annotated

                t1 = time.perf_counter()
                fps_display = 0.85 * fps_display + 0.15 / max(t1 - t0, 0.001)

                if frame_idx % 30 == 0:
                    occ_info = f" 遮挡:{memory.occluded_count}件" if memory.occluded_count > 0 else ""
                    print(
                        f"\r[Frame {frame_idx}] 检出:{result.item_score:.0f} 摆放:{result.placement_score*100:.0f} "
                        f"证据:{result.score_report.evidence_reliability:.0f} 覆盖:{result.score_report.requirement_coverage*100:.0f}% "
                        f"→ 可观测得分:{result.weighted_score:.0f}{occ_info} "
                        f"FPS:{fps_display:.1f}",
                        end="",
                        flush=True,
                    )

            display = cv2.resize(last_annotated, (1400, 788))
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key in [ord('q'), 27]:
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(' '):
                paused = not paused
            elif key == ord('s') and last_annotated is not None:
                screenshot_dir = PROJECT_ROOT / "output" / "screenshots"
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = screenshot_dir / f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(screenshot_path), last_annotated)
                print(f"\n📸 截图: {screenshot_path}")
            elif key == ord('e') and last_result is not None:
                report_path = save_session_report(
                    last_result, args, loaded, frame_idx, observation_engine, state_machine
                )
                print(f"\n观测报告: {report_path}")
            elif key == ord('t'):
                tracking_enabled = not tracking_enabled
                pipeline.set_tracking_enabled(tracking_enabled)
                trajectories.clear()
                tracker.reset()
                print(f"\n🔍 追踪: {'ON' if tracking_enabled else 'OFF'}")
            elif key in [ord('+'), ord('=')]:
                args.conf = min(0.9, args.conf + 0.05)
                pipeline.set_confidence(args.conf)
                print(f"\n🔧 置信度: {args.conf:.2f}")
            elif key == ord('-'):
                args.conf = max(0.1, args.conf - 0.05)
                pipeline.set_confidence(args.conf)
                print(f"\n🔧 置信度: {args.conf:.2f}")
            elif key == ord('r') and cap_info.can_replay:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                pipeline.reset()
                trajectories.clear()
                paused = False
                print("\n▶ 重播视频")
            elif key == ord('n') and state_machine is not None:
                current_step_id = state_machine.current_step_id
                if current_step_id is not None:
                    timestamp = state_machine.last_timestamp or time.monotonic()
                    transition = state_machine.skip_step(
                        current_step_id,
                        "动作联调期间人工跳过",
                        timestamp,
                        force=True,
                    )
                    print(
                        f"\n⏭ 跳过步骤: {current_step_id} "
                        f"({'成功' if transition.accepted else '未执行'})"
                    )
            elif key == ord('x'):
                pipeline.reset()
                trajectories.clear()
                print("\n↺ 已重置观测器和SOP状态")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if hand_detector is not None:
            hand_detector.close()
        if pose_detector is not None:
            pose_detector.close()

    print("\n✅ 检测结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
