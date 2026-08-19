"""
视频源模块 — 统一摄像头和视频文件读取。
"""

from dataclasses import dataclass
from typing import Optional
import os
import sys

import cv2


@dataclass
class CaptureInfo:
    source: str
    is_camera: bool
    width: int
    height: int
    fps: float
    total_frames: int
    can_replay: bool


def open_capture(
    source: str = "camera",
    camera_id: int = 0,
    video_path: Optional[str] = None,
    width: int = 1280,
    height: int = 720,
    use_dshow: bool = True,
):
    """打开摄像头或视频文件。"""
    if source == "camera":
        backend = cv2.CAP_DSHOW if use_dshow and sys.platform.startswith("win") else 0
        if backend:
            cap = cv2.VideoCapture(camera_id, backend)
        else:
            cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        label = f"camera:{camera_id}"
        is_camera = True
    elif source == "video":
        if not video_path:
            raise ValueError("视频模式需要提供 --video 路径")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        cap = cv2.VideoCapture(video_path)
        label = video_path
        is_camera = False
    else:
        raise ValueError(f"未知 source: {source}")

    if not cap.isOpened():
        hint = "摄像头可能被占用、编号错误或权限不足。" if is_camera else "请检查视频路径和编码格式。"
        raise RuntimeError(f"无法打开视频源: {label}\n{hint}")

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else 0

    info = CaptureInfo(
        source=label,
        is_camera=is_camera,
        width=actual_width,
        height=actual_height,
        fps=fps,
        total_frames=total_frames,
        can_replay=not is_camera and total_frames > 0,
    )
    return cap, info
