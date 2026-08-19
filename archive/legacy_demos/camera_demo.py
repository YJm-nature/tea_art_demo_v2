"""
摄像头实时茶具检测 — 即开即用

使用现有模型 (13类) 过滤到 9 类显示
"""
import cv2, time, sys
from pathlib import Path
from ultralytics import YOLO

# 模型: 13类 → 只显示9类
PROJECT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT / "models" / "tea_ware_train2" / "weights" / "best.pt"

# 新模型精确9类
ALL_CLASSES = ["盖碗", "公道杯", "品茗杯", "茶荷", "茶巾", "茶夹", "茶拨", "茶盘", "建水"]
SKIP_IDS = set()  # 无需过滤
COLORS = [
    (0, 255, 128), (255, 128, 0), (0, 200, 255),
    (255, 0, 128), (128, 0, 255), (0, 255, 255),
    (255, 255, 0), (128, 255, 0), (255, 80, 80),
]

print("Loading model...")
model = YOLO(str(MODEL_PATH))
print("Model loaded!\n")

# 摄像头
CAM_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Tea Utensils - Camera (Q=quit, S=save)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tea Utensils - Camera (Q=quit, S=save)", 1280, 800)

fps_avg = 0
CONF = 0.3

print(f"Camera {CAM_ID} ready. Q=quit S=save")
print(f"Detecting 9 classes, conf={CONF}\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed!")
        break

    frame = cv2.resize(frame, (1280, 720))
    t0 = time.perf_counter()

    results = model(frame, conf=CONF, iou=0.45, verbose=False)

    t1 = time.perf_counter()
    fps_avg = 0.9 * fps_avg + 0.1 / max(t1 - t0, 0.001)

    # 绘制
    annotated = frame.copy()
    detected = set()
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            cid = int(box.cls[0])
            if cid in SKIP_IDS:
                continue
            conf = float(box.conf[0])
            if cid >= len(ALL_CLASSES):
                continue
            name = ALL_CLASSES[cid]
            color = COLORS[cid]
            detected.add(name)

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.putText(annotated, f"FPS:{fps_avg:.1f} Found:{len(detected)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 右上角检出清单
    y = 60
    for i, cn in enumerate(ALL_CLASSES):
        if i in SKIP_IDS:
            continue
        found = cn in detected
        clr = (0, 255, 0) if found else (80, 80, 80)
        cv2.putText(annotated, f"{'O' if found else '-'} {cn}",
                   (annotated.shape[1] - 170, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1)
        y += 24

    cv2.imshow("Tea Utensils - Camera (Q=quit, S=save)", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('s'):
        fname = f"camera_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(fname, annotated)
        print(f"Saved: {fname}")
    elif key == ord('+') or key == ord('='):
        CONF = min(0.9, CONF + 0.05)
        print(f"Conf: {CONF:.2f}")
    elif key == ord('-'):
        CONF = max(0.1, CONF - 0.05)
        print(f"Conf: {CONF:.2f}")

cap.release()
cv2.destroyAllWindows()
print("Done.")
