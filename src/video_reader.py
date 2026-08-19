"""
视频帧读取模块

从视频文件逐帧读取，支持4K→720p缩放和帧采样。
"""

import cv2
import time
from typing import Optional, Tuple, Generator
import numpy as np


class VideoReader:
    """视频帧读取器，支持预处理（缩放、采样、ROI裁切）"""

    def __init__(
        self,
        video_path: str,
        target_width: int = 1280,
        target_height: int = 720,
        sample_every_n: int = 2,
    ):
        """
        Args:
            video_path: 视频文件路径
            target_width: 推理分辨率宽度
            target_height: 推理分辨率高度
            sample_every_n: 每N帧处理1帧
        """
        self.video_path = video_path
        self.target_width = target_width
        self.target_height = target_height
        self.sample_every_n = sample_every_n

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        self.frame_idx = 0
        self.processed_idx = 0

    # ─── 属性 ──────────────────────────────────────────

    @property
    def resolution_str(self) -> str:
        return f"{self.original_width}x{self.original_height}"

    @property
    def progress(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return min(self.frame_idx / self.total_frames, 1.0)

    # ─── 帧读取 ────────────────────────────────────────

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        读取下一帧。

        Returns:
            (has_frame, frame_original, frame_inference):
                has_frame: 是否还有帧
                frame_original: 原始分辨率帧（用于显示）
                frame_inference: 缩放后帧（用于推理）
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None

        self.frame_idx += 1

        # 帧采样
        if self.frame_idx % self.sample_every_n != 0:
            return True, None, None  # 跳过此帧推理，但仍推进视频

        self.processed_idx += 1

        frame_original = frame.copy()

        # 缩放到推理分辨率
        h, w = frame.shape[:2]
        if w != self.target_width or h != self.target_height:
            frame_inference = cv2.resize(
                frame, (self.target_width, self.target_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            frame_inference = frame

        return True, frame_original, frame_inference

    def read_all_frames(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        遍历所有需处理的帧。

        Yields:
            (frame_original, frame_inference)
        """
        while True:
            has_frame, original, inference = self.read_frame()
            if not has_frame:
                break
            if inference is not None:
                yield original, inference

    def seek(self, position_sec: float):
        """跳转到指定时间位置"""
        frame_no = int(position_sec * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        self.frame_idx = frame_no

    def reset(self):
        """重置到视频开头"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_idx = 0
        self.processed_idx = 0

    def close(self):
        """释放视频资源"""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return (
            f"<VideoReader: {self.original_width}x{self.original_height}, "
            f"{self.fps:.0f}fps, {self.total_frames} frames, "
            f"{self.duration:.1f}s, every {self.sample_every_n}th frame>"
        )
