"""
茶具实时检测 — OpenCV 高性能版 (~45+ FPS on GPU)

直接 OpenCV 窗口显示，无 Streamlit 开销，最大化推理帧率。

使用方式:
  python realtime_demo_fast.py                          # 默认茶艺视频
  python realtime_demo_fast.py <video_path>             # 指定视频
  python realtime_demo_fast.py camera                   # 摄像头
"""
import cv2, sys, os, time
import numpy as np
from pathlib import Path
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

cv2.namedWindow("Tea Utensil Detection (GPU)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tea Utensil Detection (GPU)", 1400, 900)

fps_avg = 0
frame_idx = 0
paused = False
CONF = 0.35
IOU = 0.45

print(f"✅ 就绪 | 置信度:{CONF} | IoU:{IOU}")
print(f"   Q=退出  Space=暂停  S=截图  +/-=调阈值")
print()

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            if isinstance(VIDEO, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue
            else:
                break
        frame_idx += 1

    frame_inf = cv2.resize(frame, (1280, 720))

    # ── 推理 ──
    t0 = time.perf_counter()
    results = model(frame_inf, conf=CONF, iou=IOU, verbose=False)
    t1 = time.perf_counter()
    fps_avg = 0.9 * fps_avg + 0.1 / max(t1 - t0, 0.001)

    # ── 绘制检测框 ──
    annotated = frame_inf.copy()
    detected = set()

    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                if cls_id < len(CLASS_NAMES):
                    name = CLASS_NAMES[cls_id]
                    color = COLORS[cls_id]
                    detected.add(name)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(annotated, label, (x1 + 3, y1 - 3),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # ── 信息面板 ──
    h, w = annotated.shape[:2]
    panel_w = 240
    cv2.rectangle(annotated, (w - panel_w, 0), (w, h), (30, 30, 35), -1)

    y_pos = 30
    cv2.putText(annotated, f"FPS: {fps_avg:.1f}", (w - panel_w + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
    y_pos += 35
    cv2.putText(annotated, f"Frame: {frame_idx}", (w - panel_w + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_pos += 35
    cv2.putText(annotated, f"检出: {len(detected)}/9", (w - panel_w + 10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    y_pos += 25
    cv2.line(annotated, (w - panel_w + 10, y_pos), (w - 10, y_pos), (80, 80, 80), 1)
    y_pos += 20

    for i, cn in enumerate(CLASS_NAMES):
        found = cn in detected
        icon = "●" if found else "○"
        color = (100, 255, 100) if found else (150, 150, 150)
        cv2.putText(annotated, f"{icon} {cn}", (w - panel_w + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y_pos += 22

    # ── 底部控制栏 ──
    bar_h = 32
    cv2.rectangle(annotated, (0, h - bar_h), (w, h), (25, 25, 30), -1)
    tips = [
        ("Q 退出", (80, 200, 80)),
        ("Space 暂停", (200, 180, 60)),
        ("S 截图", (120, 140, 220)),
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
    cv2.imshow("Tea Utensil Detection (GPU)", display)

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
    elif key in [ord('+'), ord('=')]:
        CONF = min(0.9, CONF + 0.05)
        print(f"🔧 置信度: {CONF:.2f}")
    elif key == ord('-'):
        CONF = max(0.1, CONF - 0.05)
        print(f"🔧 置信度: {CONF:.2f}")

    # 窗口关闭检测
    if cv2.getWindowProperty("Tea Utensil Detection (GPU)", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ 检测结束")
