"""
手部检测模块 — MediaPipe Hands 封装（Tasks API）

输出每只手的关键点、包围框、左右手标签。
用于遮挡检测（替代"未知物品"间接推断）和后续动作识别。

兼容 MediaPipe >= 0.10.0（Tasks API）。
"""

import os
import urllib.request
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ─── 模型自动下载 ─────────────────────────────────────

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "hand_landmarker.task")


def _download_model():
    """下载 MediaPipe 手部关键点模型到本地 models/ 目录"""
    if os.path.exists(_MODEL_PATH):
        return
    os.makedirs(_MODEL_DIR, exist_ok=True)
    print(f"[HandDetector] 下载手部关键点模型...")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[HandDetector] 模型已保存: {_MODEL_PATH}")
    except Exception as e:
        print(f"[HandDetector] 模型下载失败: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════

class HandDetector:
    """
    MediaPipe Hands 轻量封装（Tasks API）。

    特性：
    - 延迟加载：首次 detect() 调用时才加载模型
    - 自动下载模型文件到 models/ 目录
    - 每 N 帧运行一次（可配置）
    - BGR 输入自动转 RGB
    - bbox 由 21 个 landmarks 计算（紧凑包围框）
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        detect_every_n_frames: int = 2,
    ):
        """
        Args:
            max_num_hands: 最多检测手数
            min_detection_confidence: 检测置信度阈值
            min_tracking_confidence: 追踪置信度阈值
            detect_every_n_frames: 每 N 次 detect() 调用才真正运行一次
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.detect_every_n_frames = detect_every_n_frames

        self._landmarker = None    # HandLandmarker 实例（延迟加载）
        self._loaded = False
        self._call_count = 0
        self._last_results: List[Dict[str, Any]] = []  # 缓存最近一次结果

    # ─── 延迟加载 ──────────────────────────────────────

    def _ensure_loaded(self):
        """首次调用时加载 MediaPipe HandLandmarker"""
        if self._loaded:
            return
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            # 确保模型文件存在
            _download_model()

            base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
            self._loaded = True
            print("[HandDetector] 手部关键点模型已加载")
        except ImportError:
            print("[HandDetector] mediapipe 未安装，手部检测不可用")
            self._loaded = True
        except Exception as e:
            print(f"[HandDetector] 加载失败: {e}")
            self._loaded = True

    @property
    def is_loaded(self) -> bool:
        self._ensure_loaded()
        return self._landmarker is not None

    # ─── 主检测接口 ────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        roi_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        检测画面中的手部。

        Args:
            frame: BGR 格式图像 (H, W, 3)

        Returns:
            [
                {
                    "bbox": (x, y, w, h),       # 手部紧凑包围框（像素坐标）
                    "handedness": "Left" | "Right",
                    "confidence": float,          # 检测置信度 0-1
                    "landmarks": np.ndarray,      # (21, 3) — x, y, z 像素坐标
                    "center": (cx, cy),           # 包围框中心
                },
                ...
            ]
        """
        self._ensure_loaded()
        self._call_count += 1

        # 帧采样：跳过不处理的帧，返回缓存结果
        if self._call_count % self.detect_every_n_frames != 0:
            return self._last_results

        if self._landmarker is None:
            return []

        frame_h, frame_w = frame.shape[:2]
        results: List[Dict[str, Any]] = []

        # A front/side overview often makes hands too small for full-frame Hands.
        # Pose-guided crops enlarge each wrist while preserving full-frame output.
        for roi in roi_bboxes or []:
            x, y, w, h = roi
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(frame_w, int(x + w)), min(frame_h, int(y + h))
            if x2 - x1 < 32 or y2 - y1 < 32:
                continue
            results.extend(self._detect_region(frame[y1:y2, x1:x2], x1, y1, frame_w, frame_h))

        if not roi_bboxes or len(results) < 2:
            results.extend(self._detect_region(frame, 0, 0, frame_w, frame_h))

        self._last_results = self._deduplicate(results)[: self.max_num_hands]
        return self._last_results

    def _detect_region(
        self,
        region: np.ndarray,
        offset_x: int,
        offset_y: int,
        frame_w: int,
        frame_h: int,
    ) -> List[Dict[str, Any]]:
        region_h, region_w = region.shape[:2]
        if region_h == 0 or region_w == 0:
            return []
        rgb = np.ascontiguousarray(region[:, :, ::-1])
        try:
            from mediapipe import Image, ImageFormat
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            mp_result = self._landmarker.detect(mp_image)
        except Exception:
            return []

        results: List[Dict[str, Any]] = []
        for index, landmarks_list in enumerate(mp_result.hand_landmarks or []):
            handedness = "Unknown"
            confidence = 0.9
            if mp_result.handedness and index < len(mp_result.handedness):
                handedness = mp_result.handedness[index][0].category_name
                confidence = float(mp_result.handedness[index][0].score)
            landmarks_px = np.array([
                [
                    offset_x + landmark.x * region_w,
                    offset_y + landmark.y * region_h,
                    landmark.z * region_w,
                ]
                for landmark in landmarks_list
            ], dtype=np.float32)
            xs, ys = landmarks_px[:, 0], landmarks_px[:, 1]
            margin = 15
            x1 = max(0, int(xs.min()) - margin)
            y1 = max(0, int(ys.min()) - margin)
            x2 = min(frame_w, int(xs.max()) + margin)
            y2 = min(frame_h, int(ys.max()) + margin)
            results.append({
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "handedness": handedness,
                "confidence": confidence,
                "landmarks": landmarks_px,
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                "source": "pose_roi" if offset_x or offset_y else "full_frame",
            })
        return results

    @staticmethod
    def _deduplicate(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for candidate in sorted(results, key=lambda row: float(row.get("confidence", 0)), reverse=True):
            cx, cy = candidate["center"]
            duplicate = False
            for existing in kept:
                ex, ey = existing["center"]
                ew, eh = existing["bbox"][2:]
                if np.hypot(cx - ex, cy - ey) <= max(ew, eh, 20) * 0.6:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept

    # ─── 便捷方法 ──────────────────────────────────────

    def get_hand_bboxes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """只返回手部 bbox 列表（遮挡检测用）"""
        return [h["bbox"] for h in self.detect(frame)]

    def get_hand_count(self, frame: np.ndarray) -> int:
        """返回检测到的手部数量"""
        return len(self.detect(frame))

    def has_hands(self, frame: np.ndarray) -> bool:
        """画面中是否有手"""
        return len(self.detect(frame)) > 0

    # ─── 释放 ──────────────────────────────────────────

    def close(self):
        """释放 MediaPipe 资源"""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            self._loaded = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        status = "loaded" if (self._loaded and self._landmarker) else "not loaded"
        return f"<HandDetector: {status}, frame_skip={self.detect_every_n_frames}>"
