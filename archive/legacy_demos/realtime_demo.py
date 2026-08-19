"""
茶艺红茶 AI实时检测 — GPU加速版
直接OpenCV显示，真实时（~48 FPS）
"""
import cv2, sys, os, time
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
VIDEO_INBOX = PROJECT / "dataset" / "tea_sop_modular_v1" / "raw_videos" / "00_inbox"
sys.path.insert(0, str(PROJECT))
from src.tea_detector import TeaDetector
from src.item_matcher import ItemMatcher
from src.draw_utils import draw_detections, draw_info_panel, draw_hand_skeleton, draw_pose_skeleton
from src.detection_memory import DetectionMemory
from src.observation_point import OBSERVATION_REGISTRY
from src.hand_detector import HandDetector
from src.pose_detector import PoseDetector

# 视频选择
if len(sys.argv) > 1:
    VIDEO = sys.argv[1]
else:
    # 默认用最长的
    candidates = [
        VIDEO_INBOX / "VID_20260612_094722.mp4",
        VIDEO_INBOX / "VID_20260612_095428.mp4",
    ]
    VIDEO = str(next((v for v in candidates if v.exists()), candidates[0]))

print(f"Video: {os.path.basename(VIDEO)}")
print(f"GPU: GTX 1060 | Target: 48 FPS")
print()

detector = TeaDetector(use_yolo=True)
matcher = ItemMatcher()
memory = DetectionMemory()  # 跨帧累积记忆
hand_detector = HandDetector(detect_every_n_frames=2)
pose_detector = PoseDetector(detect_every_n_frames=2)

cap = cv2.VideoCapture(VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_dur = total_frames / video_fps if video_fps > 0 else 0

cv2.namedWindow("Tea Art AI Detection (GPU)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tea Art AI Detection (GPU)", 1400, 900)

fps_display = 0
frame_idx = 0
paused = False

def draw_controls(img):
    """在画面底部绘制控制按钮栏"""
    h, w = img.shape[:2]
    bar_h = 36
    y0 = h - bar_h
    cv2.rectangle(img, (0, y0), (w, h), (30, 30, 35), -1)

    controls = [
        ("Q 退出", (60, 180, 100)),
        ("Space 暂停", (255, 200, 60) if not paused else (60, 255, 100)),
        ("S 截图", (120, 180, 255)),
        ("R 重播", (200, 140, 255)),
    ]
    x_pos = 10
    for label, color in controls:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x_pos, y0 + 4), (x_pos + tw + 12, y0 + th + 10), color, -1)
        cv2.putText(img, label, (x_pos + 6, y0 + th + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        x_pos += tw + 18

    # 进度条
    bar_w = w - x_pos - 20
    progress = frame_idx / max(total_frames, 1)
    cv2.rectangle(img, (x_pos, y0 + 10), (x_pos + bar_w, y0 + 26), (80, 80, 80), -1)
    filled_w = int(bar_w * progress)
    cv2.rectangle(img, (x_pos, y0 + 10), (x_pos + filled_w, y0 + 26), (0, 200, 100), -1)
    pct_text = f"{progress*100:.0f}%"
    cv2.putText(img, pct_text, (x_pos + bar_w//2 - 15, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

while True:
    # 读帧（到末尾自动循环）
    if not paused:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            ret, frame = cap.read()
            if not ret: break
        frame_idx += 1

    frame_inf = cv2.resize(frame, (1280, 720))

    t0 = time.perf_counter()

    items = detector.detect(frame_inf)
    matched = matcher.match(items, frame_inf.shape[:2])

    # ── 摆放合理性 + 操作规范性评分 ──
    placement_score = matcher.get_placement_score(matched, frame_inf.shape[:2])

    # ── 手部 + 姿态检测 ──
    hand_results = hand_detector.detect(frame_inf)
    hand_bboxes = [h["bbox"] for h in hand_results]
    pose_results = pose_detector.detect(frame_inf)
    arm_bboxes = pose_detector.get_arm_bboxes(frame_inf)

    # ── 累积记忆（打分用，只升不降） ──
    memory.accumulate(
        matched, frame_idx,
        hand_bboxes=hand_bboxes,
        arm_bboxes=arm_bboxes,
    )
    checklist_scoring = memory.get_checklist(matcher.items_config)
    ess, tot, score = matcher.compute_score(checklist_scoring)
    grade, color = matcher.get_verdict(ess, tot)

    # ── 操作规范性（需在 accumulate 之后，因为依赖 occlusion_count） ──
    normality_score = matcher.get_area_normality_score(
        matched, occluded_count=memory.occluded_count
    )

    # ── 路由到观测点框架 ──
    obs = OBSERVATION_REGISTRY.get("obj_utensils_s1")
    if obs is not None:
        obs_result = obs.detect(frame_inf, context={
            "checklist": checklist_scoring,
            "essential_found": ess,
            "total_essential": tot,
            "score": score,
            "grade": grade,
            "grade_color": color,
            "placement_score": placement_score,
            "normality_score": normality_score,
        })

    # ── 三维加权得分 ──
    weighted_score = (
        score * 0.40
        + placement_score * 100 * 0.30
        + normality_score * 100 * 0.30
    )

    annotated = draw_detections(frame_inf, matched)
    if hand_results:
        annotated = draw_hand_skeleton(annotated, hand_results)
    if pose_results:
        annotated = draw_pose_skeleton(annotated, pose_results)
    panel = {
        "detected_count": ess, "total_count": tot,
        "score": weighted_score, "grade": grade, "grade_color": color,
        "fps": fps_display, "frame_idx": frame_idx,
    }
    annotated = draw_info_panel(annotated, panel)
    draw_controls(annotated)

    # 每30帧输出一次三维评分到控制台
    if frame_idx % 30 == 0:
        occ_info = f" 遮挡:{memory.occluded_count}件" if memory.occluded_count > 0 else ""
        print(
            f"\r[Frame {frame_idx}] 检出:{score:.0f} 摆放:{placement_score*100:.0f} "
            f"规范:{normality_score*100:.0f} → 综合:{weighted_score:.0f}{occ_info}  "
            f"FPS:{fps_display:.1f}",
            end="", flush=True,
        )

    t1 = time.perf_counter()
    fps_display = 0.85 * fps_display + 0.15 / max(t1 - t0, 0.001)

    # 显示——放大到更大窗口
    display = cv2.resize(annotated, (1400, 788))
    cv2.imshow("Tea Art AI Detection (GPU)", display)

    key = cv2.waitKey(10) & 0xFF
    # 多种退出方式
    if key == ord('q') or key == 27:  # Q or Esc
        break
    elif cv2.getWindowProperty("Tea Art AI Detection (GPU)", cv2.WND_PROP_VISIBLE) < 1:
        break  # 窗口被关闭
    elif key == ord(' '):
        paused = not paused
    elif key == ord('s'):
        fname = f"screenshot_{time.strftime('%H%M%S')}.jpg"
        cv2.imwrite(fname, annotated)
        print(f"Saved: {fname}")
    elif key == ord('r'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        paused = False
        print("Replaying...")

cap.release()
cv2.destroyAllWindows()
print("Done.")
