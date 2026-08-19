"""
跨帧检测记忆模块 — DetectionMemory

解决"品茗杯移出画面导致误扣分"的问题：
- 画面 overlay → 用当前帧实时检测（看到什么画什么）
- 打分依据     → 用跨帧累积记忆（见过就记住，不冤枉）

核心策略："只升不降"
  新检出 → 置信度高于记忆中 → 更新
  新检出 → 置信度低于记忆中 → 保留旧值（不降级）
  未检出 → 保留记忆中的记录   → 不更新
"""

from typing import Dict, Optional
from .tea_detector import DetectedItem


class DetectionMemory:
    """
    跨帧检测记忆。

    维护每个物品的历史最佳检测结果，
    只在置信度提升时更新，绝不因暂时丢失而降级。
    """

    def __init__(self):
        self._memory: Dict[str, dict] = {}
        self._occluded_items: set = set()  # 已确认被遮挡的物品名（只进不出，每件最多扣一次分）
        # { item_name: {"detected": bool, "confidence": float, "count": int, "first_seen": int, "last_seen": int} }

    # ─── 累积合并 ──────────────────────────────────────

    def accumulate(
        self,
        matched_items: list,  # List[DetectedItem]
        frame_idx: int = 0,
        hand_bboxes: list = None,  # List[(x, y, w, h)] — HandDetector 输出
        arm_bboxes: list = None,   # List[(x, y, w, h)] — PoseDetector 上肢输出
    ) -> None:
        """
        将当前帧的检出结果合并到记忆中。

        规则（只升不降）：
        - 已知物品且当前帧有检出 → 取最高置信度、最大数量，更新位置
        - 已知物品但当前帧未检出 → 保留记忆（不降级）
        - 新物品                → 写入记忆
        - 有手/胳膊且物品消失    → 检测遮挡
        """
        # 统计当前帧各物品的检出
        current: Dict[str, dict] = {}
        for item in matched_items:
            name = item.item_name
            if name == "未知物品":
                continue
            if name not in current:
                current[name] = {
                    "detected": True,
                    "confidence": item.confidence,
                    "count": 0,
                    "bbox": item.bbox,           # (x, y, w, h)
                    "centroid": item.centroid,    # (cx, cy)
                }
            current[name]["confidence"] = max(current[name]["confidence"], item.confidence)
            current[name]["count"] += 1
            # 保留最新出现的那个实例的位置
            current[name]["bbox"] = item.bbox
            current[name]["centroid"] = item.centroid

        # 合并到记忆
        for name, cur in current.items():
            if name not in self._memory:
                # 首次检出
                self._memory[name] = {
                    "detected": True,
                    "confidence": cur["confidence"],
                    "count": cur["count"],
                    "first_seen": frame_idx,
                    "last_seen": frame_idx,
                    "last_bbox": cur["bbox"],
                    "last_centroid": cur["centroid"],
                    "seen_frames": 1,
                }
            else:
                mem = self._memory[name]
                # 置信度：取最高
                if cur["confidence"] > mem["confidence"]:
                    mem["confidence"] = cur["confidence"]
                # 数量：取最大
                if cur["count"] > mem["count"]:
                    mem["count"] = cur["count"]
                mem["last_seen"] = frame_idx
                mem["detected"] = True  # 确认仍存在
                # 更新最后位置
                mem["last_bbox"] = cur["bbox"]
                mem["last_centroid"] = cur["centroid"]
                mem["seen_frames"] = mem.get("seen_frames", 0) + 1

        # 注：未检出的物品不更新，保留记忆中的旧值（含位置）

        # ── 遮挡检测（MediaPipe Hands + Pose 上肢） ──
        # 逻辑：物品在记忆中但当前帧消失 + 手/胳膊框与物品最后位置重叠 → 判定为遮挡
        current_names = {
            item.item_name for item in matched_items
            if item.item_name != "未知物品"
        }

        # 合并手部 + 上肢 bbox
        occluder_bboxes = list(hand_bboxes or []) + list(arm_bboxes or [])

        if occluder_bboxes:
            for name, mem in self._memory.items():
                if mem.get("detected") and name not in current_names:
                    if name not in self._occluded_items:
                        if _hand_overlaps_item(
                            occluder_bboxes,
                            mem.get("last_bbox"),
                            mem.get("last_centroid"),
                        ):
                            self._occluded_items.add(name)

    # ─── 查询 ──────────────────────────────────────────

    def get_checklist(
        self,
        items_config: dict,  # tea_items.json 中的 items 配置
        current_frame: int = None,
        max_age_frames: int = None,
    ) -> Dict[str, dict]:
        """
        基于累积记忆生成物品清单。

        Returns:
            {item_name: {name_cn, essential, detected, count, confidence, ...}}
        """
        checklist = {}
        for name, cfg in items_config.items():
            mem = self._memory.get(name, {})
            count = mem.get("count", 0)
            qty_range = cfg.get("quantity_range", [1, 1])

            is_recent = True
            if current_frame is not None and max_age_frames is not None:
                last_seen = mem.get("last_seen", -1)
                is_recent = last_seen >= 0 and current_frame - last_seen <= max_age_frames
            present = bool(mem.get("detected", False) and is_recent)

            if present:
                detected = qty_range[0] <= count <= qty_range[1] if qty_range else True
            else:
                detected = False

            checklist[name] = {
                "name_cn": cfg["name_cn"],
                "essential": cfg["essential"],
                "present": present,
                "detected": detected,
                "count": count,
                "expected_range": qty_range,
                "confidence": round(mem.get("confidence", 0.0), 3),
                "first_seen": mem.get("first_seen", -1),
                "last_seen": mem.get("last_seen", -1),
                "seen_frames": mem.get("seen_frames", 0),
                "age_frames": (
                    current_frame - mem.get("last_seen", current_frame)
                    if current_frame is not None and mem else None
                ),
            }

        return checklist

    # ─── 重置 ──────────────────────────────────────────

    def reset(self) -> None:
        """清空记忆（新视频开始时调用）"""
        self._memory.clear()
        self._occluded_items.clear()

    # ─── 属性 ──────────────────────────────────────────

    @property
    def remembered_count(self) -> int:
        """已记住的物品类别数"""
        return sum(1 for v in self._memory.values() if v.get("detected"))

    @property
    def occluded_count(self) -> int:
        """已确认被遮挡的物品数量（不重复）"""
        return len(self._occluded_items)

    @property
    def occluded_items(self) -> set:
        """已确认被遮挡的物品名集合（只读副本）"""
        return set(self._occluded_items)

    @property
    def memory(self) -> Dict[str, dict]:
        """只读访问内部记忆"""
        return dict(self._memory)

    def __repr__(self) -> str:
        items = [f"{k}({v['confidence']:.0%})" for k, v in self._memory.items() if v.get("detected")]
        occ = f", {len(self._occluded_items)} occluded" if self._occluded_items else ""
        return f"<DetectionMemory: {len(items)} items{occ} — {', '.join(items[:5])}{'...' if len(items) > 5 else ''}>"


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════

def _hand_overlaps_item(
    hand_bboxes: list,
    item_bbox: tuple,
    item_centroid: tuple,
    iou_threshold: float = 0.05,
    distance_threshold: float = 150.0,
) -> bool:
    """
    判断手部是否与物品区域重叠。

    采用宽松判定（IoU 或 中心距）：
    - 任一 hand_bbox 与 item_bbox 的 IoU > iou_threshold
    - 或 hand_bbox 中心到 item_centroid 的距离 < distance_threshold 像素

    Args:
        hand_bboxes: 手部框列表 [(x, y, w, h), ...]
        item_bbox: 物品框 (x, y, w, h) 或 None
        item_centroid: 物品中心 (cx, cy) 或 None
        iou_threshold: IoU 阈值（宽松，因为手和茶具不会精确重合）
        distance_threshold: 距离阈值（像素）

    Returns:
        True 如果任一手上存在重叠
    """
    if item_bbox is None and item_centroid is None:
        return False

    # 物品区域
    if item_bbox is not None:
        ix, iy, iw, ih = item_bbox
        ix2, iy2 = ix + iw, iy + ih
    else:
        ix = iy = iw = ih = ix2 = iy2 = 0

    if item_centroid is not None:
        icx, icy = item_centroid

    for hx, hy, hw, hh in hand_bboxes:
        hx2, hy2 = hx + hw, hy + hh

        # 检查1：IoU 重叠
        if item_bbox is not None:
            inter_x1 = max(ix, hx)
            inter_y1 = max(iy, hy)
            inter_x2 = min(ix2, hx2)
            inter_y2 = min(iy2, hy2)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            if inter_area > 0:
                item_area = max(iw * ih, 1)
                hand_area = max(hw * hh, 1)
                union = item_area + hand_area - inter_area
                iou = inter_area / max(union, 1)
                if iou > iou_threshold:
                    return True

        # 检查2：手部中心与物品中心距离
        if item_centroid is not None:
            hcx = hx + hw / 2
            hcy = hy + hh / 2
            dist = ((hcx - icx) ** 2 + (hcy - icy) ** 2) ** 0.5
            if dist < distance_threshold:
                return True

    return False
