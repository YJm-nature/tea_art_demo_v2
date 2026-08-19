"""
从茶艺视频中提取多样化训练帧

策略：
- 多视频源采样
- 帧差法去重（相似度>95%跳过）
- 保持不同光照/角度/物品组合的多样性
"""
import cv2, os, sys
import numpy as np
from pathlib import Path

# 输出目录
PROJECT = Path(__file__).resolve().parents[2]
VIDEO_INBOX = PROJECT / "dataset" / "tea_sop_modular_v1" / "raw_videos" / "00_inbox"
OUT_DIR = PROJECT / "training_data" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 视频源及采样间隔
VIDEOS = [
    (VIDEO_INBOX / "VID_20260612_094722.mp4", 15),   # 130s, 每15帧→~260
    (VIDEO_INBOX / "VID_20260612_095428.mp4", 50),   # 394s, 每50帧→~236
    (VIDEO_INBOX / "VID_20260612_094215.mp4", 3),    # 15s, 每3帧→~148
    (VIDEO_INBOX / "VID_20260612_093947.mp4", 2),    # 9s, 每2帧→~139
]

TARGET_WIDTH = 1280
SIMILARITY_THRESHOLD = 0.92  # 低于此值视为新场景
TARGET_COUNT = 200

extracted = 0
last_gray = None

print(f"提取目标: {TARGET_COUNT} 张多样化帧")
print(f"输出目录: {OUT_DIR}")
print()

for video_path, step in VIDEOS:
    if extracted >= TARGET_COUNT:
        break

    if not os.path.exists(video_path):
        print(f"跳过: {video_path}")
        continue

    vname = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 跳过前面1秒（可能还在调整镜头）
    start_frame = int(fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    video_extracted = 0

    print(f"[{vname}] {total} 帧, 采样间隔={step}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        # 缩放到目标宽度
        h, w = frame.shape[:2]
        scale = TARGET_WIDTH / w
        new_h = int(h * scale)
        frame_small = cv2.resize(frame, (TARGET_WIDTH, new_h))

        # 帧差去重
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))  # 缩到128x128加速比较

        if last_gray is not None:
            # 归一化互相关
            corr = np.corrcoef(gray.flatten(), last_gray.flatten())[0, 1]
            if corr > SIMILARITY_THRESHOLD:
                frame_idx += 1
                continue  # 太相似，跳过

        last_gray = gray

        # 保存
        out_path = OUT_DIR / f"{vname}_f{frame_idx:05d}.jpg"
        cv2.imwrite(str(out_path), frame_small, [cv2.IMWRITE_JPEG_QUALITY, 92])

        extracted += 1
        video_extracted += 1

        if extracted % 20 == 0:
            print(f"  已提取 {extracted} 张...")

        frame_idx += 1
        if extracted >= TARGET_COUNT:
            break

    cap.release()
    print(f"  -> 本视频提取 {video_extracted} 张 (累计 {extracted})")

print(f"\n✅ 完成！共提取 {extracted} 张训练帧")
print(f"   目录: {OUT_DIR}")
print(f"\n下一步: 用 labelImg 标注这 {extracted} 张图片")
