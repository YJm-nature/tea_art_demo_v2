"""
人体姿态检测模块 — MediaPipe Pose 封装（Tasks API）

输出全身 33 点关键点，重点提取上肢区域（肩→肘→腕→手）用于遮挡检测。
与 HandDetector 互补：Pose 覆盖肩膀到手腕，Hands 覆盖手腕到指尖。

兼容 MediaPipe >= 0.10.0（Tasks API）。
"""

import os
import urllib.request
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ─── 模型自动下载 ─────────────────────────────────────

_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
_POSE_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker_lite.task")


def _download_pose_model():
    """下载 MediaPipe 姿态关键点模型到本地 models/ 目录"""
    if os.path.exists(_POSE_MODEL_PATH):
        return
    os.makedirs(_MODEL_DIR, exist_ok=True)
    print("[PoseDetector] 下载姿态关键点模型...")
    try:
        urllib.request.urlretrieve(_POSE_MODEL_URL, _POSE_MODEL_PATH)
        print(f"[PoseDetector] 模型已保存: {_POSE_MODEL_PATH}")
    except Exception as e:
        print(f"[PoseDetector] 模型下载失败: {e}")
        raise


# ─── 上肢关键点索引 ─────────────────────────────────────
# MediaPipe Pose 33 点中与上肢相关的索引

# 躯干 / 头部（用于画骨架参考线）
SHOULDER_LEFT = 11
SHOULDER_RIGHT = 12

# 左臂
ELBOW_LEFT = 13
WRIST_LEFT = 15

# 右臂
ELBOW_RIGHT = 14
WRIST_RIGHT = 16

# 上肢索引列表（用于提取关键点坐标）
_ARM_INDICES = {
    "left_upper":  [SHOULDER_LEFT, ELBOW_LEFT],        # 左上臂
    "left_forearm": [ELBOW_LEFT, WRIST_LEFT],           # 左前臂
    "right_upper": [SHOULDER_RIGHT, ELBOW_RIGHT],       # 右上臂
    "right_forearm": [ELBOW_RIGHT, WRIST_RIGHT],        # 右前臂
}

# 用于骨架绘制的关键连接
_POSE_ARM_CONNECTIONS = [
    # 躯干
    (SHOULDER_LEFT, SHOULDER_RIGHT),
    # 左臂
    (SHOULDER_LEFT, ELBOW_LEFT),
    (ELBOW_LEFT, WRIST_LEFT),
    # 右臂
    (SHOULDER_RIGHT, ELBOW_RIGHT),
    (ELBOW_RIGHT, WRIST_RIGHT),
]


# ══════════════════════════════════════════════════════════════════════

class PoseDetector:
    """
    MediaPipe Pose 轻量封装（Tasks API）。

    特性：
    - 延迟加载 + 自动下载模型
    - 提取上肢区域（上臂 + 前臂）作为遮挡检测的 bbox
    - 返回全身 33 点关键点（供后续动作识别使用）
    - 每 N 帧运行一次
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        detect_every_n_frames: int = 2,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.detect_every_n_frames = detect_every_n_frames

        self._landmarker = None
        self._loaded = False
        self._call_count = 0
        self._last_results: List[Dict[str, Any]] = []
        self._last_arm_bboxes: List[Tuple[int, int, int, int]] = []

    # ─── 延迟加载 ──────────────────────────────────────

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            _download_pose_model()

            base_options = mp_python.BaseOptions(model_asset_path=_POSE_MODEL_PATH)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=self.min_tracking_confidence,
                output_segmentation_masks=False,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self._loaded = True
            print("[PoseDetector] 姿态关键点模型已加载")
        except ImportError:
            print("[PoseDetector] mediapipe 未安装")
            self._loaded = True
        except Exception as e:
            print(f"[PoseDetector] 加载失败: {e}")
            self._loaded = True

    @property
    def is_loaded(self) -> bool:
        self._ensure_loaded()
        return self._landmarker is not None

    # ─── 主检测接口 ────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测人体姿态，提取全身关键点和上肢 bbox。

        Args:
            frame: BGR 格式图像 (H, W, 3)

        Returns:
            [{
                "landmarks": np.ndarray,      # (33, 3) 像素坐标
                "arm_bboxes": [(x,y,w,h),...], # 上肢分段 bbox 列表
                "pose_bbox": (x,y,w,h),        # 全身包围框
            }]
        """
        self._ensure_loaded()
        self._call_count += 1

        if self._call_count % self.detect_every_n_frames != 0:
            return self._last_results

        if self._landmarker is None:
            return []

        h, w = frame.shape[:2]
        rgb = frame[:, :, ::-1]

        try:
            from mediapipe import Image, ImageFormat
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            mp_result = self._landmarker.detect(mp_image)
        except Exception:
            return self._last_results

        results: List[Dict[str, Any]] = []

        if mp_result.pose_landmarks:
            for pose_landmarks in mp_result.pose_landmarks:
                # 33 个关键点 → 像素坐标
                landmarks_px = np.array([
                    [lm.x * w, lm.y * h, lm.z * w] for lm in pose_landmarks
                ], dtype=np.float32)
                visibility = np.array([
                    float(getattr(lm, "visibility", 1.0)) for lm in pose_landmarks
                ], dtype=np.float32)
                presence = np.array([
                    float(getattr(lm, "presence", 1.0)) for lm in pose_landmarks
                ], dtype=np.float32)

                # 提取上肢 bbox（每段独立，用于精确遮挡检测）
                arm_bboxes = self._extract_arm_bboxes(landmarks_px, w, h)

                # 全身 bbox
                xs, ys = landmarks_px[:, 0], landmarks_px[:, 1]
                pose_bbox = (
                    max(0, int(xs.min()) - 10),
                    max(0, int(ys.min()) - 10),
                    min(w, int(xs.max()) + 10) - max(0, int(xs.min()) - 10),
                    min(h, int(ys.max()) + 10) - max(0, int(ys.min()) - 10),
                )

                results.append({
                    "landmarks": landmarks_px,
                    "visibility": visibility,
                    "presence": presence,
                    "arm_bboxes": arm_bboxes,
                    "pose_bbox": pose_bbox,
                })

        self._last_results = results
        return results

    # ─── 上肢 bbox 提取 ────────────────────────────────

    @staticmethod
    def _extract_arm_bboxes(
        landmarks_px: np.ndarray,
        frame_w: int,
        frame_h: int,
        margin: int = 25,
    ) -> List[Tuple[int, int, int, int]]:
        """
        从 33 点关键点中提取上肢各段的包围框。

        返回 4 段：左上臂、左前臂、右上臂、右前臂。
        每段由两个端点扩展为矩形。

        同时加入一个从肩到腕的整体胳膊 bbox（更宽松，兜底）。
        """
        arm_bboxes: List[Tuple[int, int, int, int]] = []

        # 分段 bbox（精细）
        for name, (idx_a, idx_b) in _ARM_INDICES.items():
            pts = landmarks_px[[idx_a, idx_b]]
            bbox = _pts_to_bbox(pts, frame_w, frame_h, margin=margin)
            arm_bboxes.append(bbox)

        # 整臂 bbox（肩 → 腕，更宽松，兜底遮挡检测）
        whole_arms = [
            (SHOULDER_LEFT, ELBOW_LEFT, WRIST_LEFT),
            (SHOULDER_RIGHT, ELBOW_RIGHT, WRIST_RIGHT),
        ]
        for shoulder, elbow, wrist in whole_arms:
            pts = landmarks_px[[shoulder, elbow, wrist]]
            bbox = _pts_to_bbox(pts, frame_w, frame_h, margin=margin + 10)
            arm_bboxes.append(bbox)

        return arm_bboxes

    # ─── 便捷方法 ──────────────────────────────────────

    def get_arm_bboxes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """返回所有上肢 bbox（用于遮挡检测）"""
        results = self.detect(frame)
        all_bboxes: List[Tuple[int, int, int, int]] = []
        for r in results:
            all_bboxes.extend(r.get("arm_bboxes", []))
        return all_bboxes

    def get_all_landmarks(self, frame: np.ndarray) -> List[np.ndarray]:
        """返回所有检测到的人体关键点"""
        return [r["landmarks"] for r in self.detect(frame) if "landmarks" in r]

    # ─── 释放 ──────────────────────────────────────────

    def close(self):
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
        return f"<PoseDetector: {status}, frame_skip={self.detect_every_n_frames}>"


# ══════════════════════════════════════════════════════════════════════
# 辅助
# ══════════════════════════════════════════════════════════════════════

def _pts_to_bbox(
    pts: np.ndarray,
    frame_w: int,
    frame_h: int,
    margin: int = 25,
) -> Tuple[int, int, int, int]:
    """将一组关键点转为带边距的矩形 bbox"""
    xs, ys = pts[:, 0], pts[:, 1]
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(frame_w, int(xs.max()) + margin)
    y2 = min(frame_h, int(ys.max()) + margin)
    return (x1, y1, x2 - x1, y2 - y1)
