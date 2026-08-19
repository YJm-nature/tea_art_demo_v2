"""
茶具实时检测 + 目标追踪 — ByteTrack 版

跟踪原理:
  YOLO每帧检测 → ByteTrack卡尔曼滤波预测上一帧目标在新帧的位置
  → 匈牙利算法匹配检测框与预测框 → 同一目标保持同一ID

使用方式:
  python realtime_demo_fast.py                          # 默认茶艺视频
  python realtime_demo_fast.py <video_path>             # 指定视频
  python realtime_demo_fast.py camera                   # 摄像头
"""
import cv2, sys, os, time
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[2]
VIDEO_INBOX = PROJECT / "dataset" / "tea_sop_modular_v1" / "raw_videos" / "00_inbox"

# ===== 9类茶具 =====
CLASS_NAMES = ["盖碗", "公道杯", "品茗杯", "茶荷", "茶巾", "茶夹", "茶拨", "茶盘", "建水"]
COLORS = [
    (0, 255, 128), (255, 128, 0), (0, 200, 255),
    (255, 0, 128),   (128, 0, 255),  (0, 255, 255),
    (255, 255, 0),   (128, 255, 0),  (255, 80, 80),
]

# ===== 加载模型 =====
model_paths = [
    PROJECT / "models" / "tea_ware_train" / "weights" / "best.pt",
    PROJECT / "models" / "tea_ware_best.pt",
]
MODEL_PATH = None
for mp in model_paths:
    if os.path.exists(mp):
        MODEL_PATH = mp
        break

if MODEL_PATH is None:
    print("❌ 未找到模型文件！请先训练模型。")
    print(f"   查找路径: {model_paths}")
    exit(1)

print(f"📦 加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# ===== 视频源 =====
if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg == "camera":
        VIDEO = 0
        print("📹 摄像头模式")
    else:
        VIDEO = arg
else:
    VIDEO = str(VIDEO_INBOX / "茶具检测.MP4")
    if not os.path.exists(VIDEO):
        VIDEO = 0

# ===== 初始化 =====
cap = cv2.VideoCapture(VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(VIDEO, str) else 0
video_fps = cap.get(cv2.CAP_PROP_FPS)

cv2.namedWindow("Tea Utensil Detection + Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tea Utensil Detection + Tracking", 1400, 900)

fps_avg = 0
frame_idx = 0
paused = False
CONF = 0.35
IOU = 0.45

# 轨迹历史: track_id → [(center_x, center_y), ...]
trajectories = defaultdict(list)
MAX_TRAIL = 30  # 最多保留30帧轨迹

print(f"✅ 就绪 | 置信度:{CONF} | IoU:{IOU}")
print(f"   Q=退出  Space=暂停  S=截图  T=切换追踪/纯检测  +/-=调阈值")
print()

tracking_enabled = True

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            if isinstance(VIDEO, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                trajectories.clear()
                continue
            else:
                break
        frame_idx += 1

    frame_inf = cv2.resize(frame, (1280, 720))

    # ── 推理 (track vs detect) ──
    t0 = time.perf_counter()
    if tracking_enabled:
        # ByteTrack: 检测 + 关联上一帧ID
        results = model.track(
            frame_inf, conf=CONF, iou=IOU,
            persist=True,          # 跨帧保持ID
            tracker="bytetrack.yaml",
            verbose=False,
        )
    else:
        results = model(frame_inf, conf=CONF, iou=IOU, verbose=False)
    t1 = time.perf_counter()
    fps_avg = 0.9 * fps_avg + 0.1 / max(t1 - t0, 0.001)

    # ── 绘制 ──
    annotated = frame_inf.copy()
    detected = set()

    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                # 追踪ID (track模式下才有)
                track_id = None
                if boxes.id is not None and i < len(boxes.id):
                    track_id = int(boxes.id[i])

                if cls_id >= len(CLASS_NAMES):
                    continue

                name = CLASS_NAMES[cls_id]
                color = COLORS[cls_id]
                detected.add(name)

                # 更新轨迹
                if track_id is not None:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    trajectories[track_id].append((cx, cy))
                    if len(trajectories[track_id]) > MAX_TRAIL:
                        trajectories[track_id] = trajectories[track_id][-MAX_TRAIL:]

                # ── 画检测框 ──
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # ── 画标签 ──
                if track_id is not None:
                    label = f"#{track_id} {name} {conf:.2f}"
                else:
                    label = f"{name} {conf:.2f}"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # ── 画轨迹线 ──
    for tid, trail in trajectories.items():
        if len(trail) < 2:
            continue
        # 用对应类别颜色（取最后一次检测的类别）
        for i in range(1, len(trail)):
            alpha = i / len(trail)  # 越近越亮
            thickness = max(1, int(alpha * 3))
            cv2.line(annotated, trail[i-1], trail[i],
                    (150, 150, 150), thickness, cv2.LINE_AA)

    # ── 信息面板 ──
    h, w = annotated.shape[:2]
    panel_w = 260
    cv2.rectangle(annotated, (w - panel_w, 0), (w, h), (30, 30, 35), -1)

    y_pos = 30
    mode_text = "TRACK" if tracking_enabled else "DETECT"
    mode_color = (0, 255, 200) if tracking_enabled else (200, 200, 200)
    cv2.putText(annotated, f"[{mode_text}] FPS: {fps_avg:.1f}",
               (w - panel_w + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1)
    y_pos += 30
    cv2.putText(annotated, f"Frame: {frame_idx}",
               (w - panel_w + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    y_pos += 25
    cv2.putText(annotated, f"Track IDs: {len(trajectories)}",
               (w - panel_w + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 255), 1)
    y_pos += 25
    cv2.putText(annotated, f"检出: {len(detected)}/9",
               (w - panel_w + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    y_pos += 20
    cv2.line(annotated, (w - panel_w + 10, y_pos), (w - 10, y_pos), (80, 80, 80), 1)
    y_pos += 15

    for i, cn in enumerate(CLASS_NAMES):
        found = cn in detected
        prefix = "●" if found else "○"
        clr = (100, 255, 100) if found else (130, 130, 130)
        cv2.putText(annotated, f"{prefix} {cn}", (w - panel_w + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1)
        y_pos += 22

    # ── 底部控制栏 ──
    bar_h = 32
    cv2.rectangle(annotated, (0, h - bar_h), (w, h), (25, 25, 30), -1)
    tips = [
        ("Q 退出", (80, 200, 80)),
        ("Space 暂停", (200, 180, 60)),
        ("S 截图", (120, 140, 220)),
        ("T 追踪开关", (255, 160, 60) if tracking_enabled else (150, 150, 150)),
        ("+/- 阈值", (180, 140, 220)),
    ]
    x_pos = 8
    for tip, color in tips:
        (tw, th), _ = cv2.getTextSize(tip, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(annotated, (x_pos, h - bar_h + 4), (x_pos + tw + 10, h - 6), color, -1)
        cv2.putText(annotated, tip, (x_pos + 5, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        x_pos += tw + 16
    cv2.putText(annotated, f"CONF:{CONF:.2f}", (x_pos + 10, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # ── 显示 ──
    display = cv2.resize(annotated, (1400, 788))
    cv2.imshow("Tea Utensil Detection + Tracking", display)

    # ── 键盘控制 ──
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord(' '):
        paused = not paused
        print(f"{'⏸ 暂停' if paused else '▶ 继续'}")
    elif key == ord('s'):
        fname = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(fname, annotated)
        print(f"📸 截图: {fname}")
    elif key == ord('t'):
        tracking_enabled = not tracking_enabled
        trajectories.clear()
        print(f"🔍 追踪: {'ON' if tracking_enabled else 'OFF (纯检测)'}")
    elif key in [ord('+'), ord('=')]:
        CONF = min(0.9, CONF + 0.05)
        print(f"🔧 置信度: {CONF:.2f}")
    elif key == ord('-'):
        CONF = max(0.1, CONF - 0.05)
        print(f"🔧 置信度: {CONF:.2f}")

    # 窗口关闭检测
    if cv2.getWindowProperty("Tea Utensil Detection + Tracking", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ 检测结束")
