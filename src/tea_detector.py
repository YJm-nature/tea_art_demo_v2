"""
茶具检测模块 — YOLOv8 + HSV颜色分割（双模）

- YOLO模式：使用自定义训练的茶具检测模型
- HSV模式：颜色分割+轮廓分析（YOLO不可用时的回退）
- 类别映射由当前模型 profile 决定，避免 9 类 / 13 类权重错映射
"""

from typing import List, Tuple, Optional
import os
import numpy as np
import cv2


# ─── 茶具类别 Profile ───────────────────────────────────

TEA9_CLASSES = [
    "盖碗", "公道杯", "品茗杯", "茶荷", "茶巾", "茶夹", "茶拨", "茶叶罐", "建水",
]

TEA13_CLASSES = [
    "盖碗", "公道杯", "品茗杯", "茶荷", "茶巾", "茶夹", "茶拨",
    "茶盘", "烧水壶", "建水", "电子秤", "温度计", "计时器",
]

# 兼容旧代码引用；新代码应使用 ModelProfile.class_names
TEA_CLASSES = TEA13_CLASSES


# ─── 检测结果数据结构 ──────────────────────────────────

class DetectedItem:
    """单个检测到的物品。"""

    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        contour_area: float = 0,
        centroid: Tuple[float, float] = (0, 0),
        aspect_ratio: float = 0,
        circularity: float = 0,
        solidity: float = 0,
        mean_color_bgr: Tuple[float, float, float] = (255, 255, 255),
        confidence: float = 0.5,
        item_name: str = "未知",
        class_id: Optional[int] = None,
        track_id: Optional[int] = None,
        source: Optional[str] = None,
    ):
        self.bbox = bbox
        self.x, self.y, self.w, self.h = bbox
        self.contour_area = contour_area
        self.centroid = centroid
        self.aspect_ratio = aspect_ratio
        self.circularity = circularity
        self.solidity = solidity
        self.mean_color_bgr = mean_color_bgr
        self.confidence = confidence
        self.item_name = item_name
        self.class_id = class_id
        self.track_id = track_id
        self.source = source

    @property
    def bbox_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def to_dict(self) -> dict:
        data = {
            "item_name": self.item_name,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 3),
        }
        if self.class_id is not None:
            data["class_id"] = self.class_id
        if self.track_id is not None:
            data["track_id"] = self.track_id
        if self.source is not None:
            data["source"] = self.source
        return data

    def __repr__(self):
        tid = f" track=#{self.track_id}" if self.track_id is not None else ""
        return f"<DetectedItem: {self.item_name} conf={self.confidence:.2f}{tid}>"


# ─── YOLO类别 → 茶具映射 ─────────────────────────────

# COCO预训练模型中可能与茶具相关的类别
YOLO_TO_TEA_MAP = {
    "cup": "品茗杯",
    "wine glass": "品茗杯",
    "bowl": "盖碗",
    "vase": "公道杯",
    "bottle": "烧水壶",
    "cell phone": "电子秤",
    "remote": "电子秤",
    "book": "茶盘",
    "scissors": "茶夹",
    "knife": "茶拨",
    "spoon": "茶拨",
    "fork": "茶拨",
    "mouse": "建水",
    "clock": "计时器",
}


# ─── 茶具检测器 ──────────────────────────────────────

class TeaDetector:
    """
    茶具检测器 — YOLO优先，HSV回退。

    Args:
        use_yolo: 是否启用 YOLO。
        model_path: YOLO 权重路径；为空时按项目默认候选查找。
        class_names: 当前模型类别顺序。新 Demo 由 ModelConfig 注入。
        yolo_model: 已加载的 Ultralytics YOLO 实例；避免重复加载模型。
        conf: YOLO 置信度阈值。
        iou: YOLO IoU 阈值。
        strict_class_check: 是否严格校验类别 profile。
    """

    # HSV回退参数
    HSV_WHITE_LOWER = np.array([0, 0, 155])
    HSV_WHITE_UPPER = np.array([180, 35, 255])

    def __init__(
        self,
        use_yolo: bool = True,
        model_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        yolo_model: Optional[object] = None,
        conf: float = 0.3,
        iou: float = 0.45,
        strict_class_check: bool = True,
        imgsz: int = 832,
        active_class_ids: Optional[List[int]] = None,
    ):
        self.use_yolo = use_yolo
        self.yolo_model = yolo_model
        self.model_path = model_path
        self.class_names = list(class_names) if class_names else None
        self.conf = conf
        self.iou = iou
        self.strict_class_check = strict_class_check
        self.imgsz = imgsz
        self.active_class_ids = (
            sorted({int(value) for value in active_class_ids})
            if active_class_ids is not None else None
        )

        if use_yolo and self.yolo_model is None:
            self._init_yolo()
        elif use_yolo and self.yolo_model is not None and self.class_names is None:
            self.class_names = self._names_from_model(self.yolo_model) or TEA13_CLASSES

        if self.class_names is None:
            self.class_names = TEA13_CLASSES

        # HSV模式的形态学核
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def _init_yolo(self):
        """初始化 YOLO 模型。"""
        try:
            if self.model_path is not None or self.strict_class_check:
                from .model_config import load_yolo_with_profile
                loaded = load_yolo_with_profile(
                    self.model_path,
                    requested_profile="auto",
                    strict=self.strict_class_check,
                )
                self.yolo_model = loaded.yolo_model
                self.model_path = loaded.model_path
                self.class_names = loaded.profile.class_names
                print(f"[TeaDetector] 模型已加载: {self.model_path}")
                print(f"[TeaDetector] 类别Profile: {loaded.profile.name} ({len(self.class_names)}类)")
                return

            from ultralytics import YOLO
            custom_model = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models", "tea_ware_best.pt"
            )
            if os.path.exists(custom_model):
                self.yolo_model = YOLO(custom_model)
                self.model_path = custom_model
                self.class_names = self._names_from_model(self.yolo_model) or TEA13_CLASSES
                print("[TeaDetector] Custom tea ware model loaded")
            else:
                self.yolo_model = YOLO("yolov8n.pt")
                self.class_names = self._names_from_model(self.yolo_model) or []
                print("[TeaDetector] Fallback to YOLOv8n pretrained")
        except Exception as e:
            if self.strict_class_check:
                raise
            print(f"[TeaDetector] YOLO init failed: {e} — falling back to HSV")
            self.use_yolo = False

    # ─── 主检测接口 ────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[DetectedItem]:
        """统一检测接口。"""
        if self.use_yolo and self.yolo_model is not None:
            items = self._detect_yolo(frame)
            # YOLO没检测到东西时回退HSV
            if len(items) < 2:
                items += self._detect_hsv(frame)
            return self._nms(items, iou_threshold=0.35)
        return self._detect_hsv(frame)

    # ─── YOLO检测 ──────────────────────────────────────

    def _detect_yolo(self, frame: np.ndarray) -> List[DetectedItem]:
        """使用 YOLO 模型检测茶具。"""
        h, w = frame.shape[:2]
        predict_args = {
            "verbose": False,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
        }
        if self.active_class_ids is not None:
            predict_args["classes"] = self.active_class_ids
        results = self.yolo_model(frame, **predict_args)
        items = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                if (
                    self.active_class_ids is not None
                    and cls_id not in self.active_class_ids
                ):
                    continue
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].cpu().numpy()

                x1, y1, x2, y2 = map(int, xyxy[:4])
                bw, bh = x2 - x1, y2 - y1
                area = bw * bh

                if area < 200 or area > w * h * 0.7:
                    continue

                at_edge = (x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2)
                # A fixed 50k-pixel threshold removed valid kettles placed at
                # the left edge after resizing the live frame to 1280x720.
                # Only reject implausibly large edge detections here.
                if at_edge and area > w * h * 0.35:
                    continue

                tea_name = self._class_name(cls_id)

                items.append(DetectedItem(
                    bbox=(x1, y1, bw, bh),
                    contour_area=area,
                    centroid=((x1 + x2) / 2, (y1 + y2) / 2),
                    aspect_ratio=bw / max(bh, 1),
                    confidence=conf,
                    item_name=tea_name,
                    class_id=cls_id,
                    source="yolo",
                ))

        items.sort(key=lambda it: it.contour_area, reverse=True)
        return items

    def _class_name(self, cls_id: int) -> str:
        if 0 <= cls_id < len(self.class_names):
            return self.class_names[cls_id]
        return "未知物品"

    @staticmethod
    def _names_from_model(model: object) -> List[str]:
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names.keys(), key=lambda x: int(x))]
        if isinstance(names, (list, tuple)):
            return [str(n) for n in names]
        return []

    # ─── HSV回退检测 ───────────────────────────────────

    def _detect_hsv(self, frame: np.ndarray) -> List[DetectedItem]:
        """HSV颜色分割（YOLO不可用或检出不足时的补充）。"""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.HSV_WHITE_LOWER, self.HSV_WHITE_UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        items = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200 or area > 250000:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            aspect_ratio = bw / max(bh, 1)
            solidity = area / max(bw * bh, 1)

            if solidity < 0.35:
                continue

            at_edge = (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1)
            if at_edge and area > 80000:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / max(perimeter ** 2, 1)

            M = cv2.moments(contour)
            cx = M["m10"] / M["m00"] if M["m00"] > 0 else x + bw / 2
            cy = M["m01"] / M["m00"] if M["m00"] > 0 else y + bh / 2

            items.append(DetectedItem(
                bbox=(x, y, bw, bh),
                contour_area=area,
                centroid=(cx, cy),
                aspect_ratio=aspect_ratio,
                circularity=circularity,
                solidity=solidity,
                item_name="未知物品",  # HSV模式下不猜测，交给ItemMatcher
                source="hsv",
            ))

        return items

    # ─── NMS去重 ───────────────────────────────────────

    def _nms(self, items: List[DetectedItem], iou_threshold: float = 0.35) -> List[DetectedItem]:
        if len(items) < 2:
            return items

        items_sorted = sorted(items, key=lambda it: it.contour_area, reverse=True)
        keep = []

        for item_a in items_sorted:
            suppressed = False
            for item_b in keep:
                iou = self._iou(item_a.bbox, item_b.bbox)
                if iou > iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                keep.append(item_a)

        keep.sort(key=lambda it: it.contour_area, reverse=True)
        return keep

    @staticmethod
    def _iou(bbox_a, bbox_b) -> float:
        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / max(union, 1)

    # ─── 调试 ──────────────────────────────────────────

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.HSV_WHITE_LOWER, self.HSV_WHITE_UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=2)
        return mask
