"""
物品匹配模块 — 将检测到的候选区域与茶具模板库进行匹配

基于几何特征（面积、宽高比、圆度、位置）进行加权评分匹配。
"""

import json
import os
from typing import List, Dict, Optional, Tuple
import numpy as np

from .tea_detector import DetectedItem


class ItemMatcher:
    """
    物品匹配器。

    将 TeaDetector 检测到的候选区域与预设茶具模板库进行匹配，
    输出每个候选区域最可能的物品分类。
    """

    # 匹配权重
    WEIGHT_AREA = 0.30
    WEIGHT_ASPECT = 0.25
    WEIGHT_CIRCULARITY = 0.20
    WEIGHT_POSITION = 0.15
    WEIGHT_COUNT = 0.10

    def __init__(
        self,
        config_path: Optional[str] = None,
        supported_items: Optional[List[str]] = None,
        item_aliases: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            config_path: tea_items.json 路径，默认使用 config/tea_items.json
            supported_items: 当前模型可检测类别；为空时使用完整配置
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config", "tea_items.json",
            )

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        all_items = self.config["items"]
        if supported_items is not None:
            supported = set(supported_items)
            self.items_config = {
                name: cfg for name, cfg in all_items.items()
                if name in supported
            }
        else:
            self.items_config = dict(all_items)

        self.essential_items = [
            name for name, cfg in self.items_config.items() if cfg["essential"]
        ]
        self.optional_items = [
            name for name, cfg in self.items_config.items() if not cfg["essential"]
        ]
        self.all_item_names = list(self.items_config.keys())
        self.item_aliases = dict(item_aliases or {})

    # ─── 主匹配逻辑 ────────────────────────────────────

    def match(
        self,
        detected_items: List[DetectedItem],
        frame_shape: Tuple[int, int],
    ) -> List[DetectedItem]:
        """
        将检测到的物品与模板库匹配。

        Args:
            detected_items: TeaDetector检测到的候选物品
            frame_shape: 推理帧尺寸 (h, w)

        Returns:
            已匹配的物品列表（item_name已填充）
        """
        if not detected_items:
            return []

        h, w = frame_shape
        matched_items = []

        for item in detected_items:
            item.item_name = self.item_aliases.get(item.item_name, item.item_name)

            # 新ontology中的碗盖、显示屏等不参与步骤一评分，但必须保留给后续状态模块。
            if item.source in {"yolo", "track"} and item.item_name not in self.items_config:
                matched_items.append(item)
                continue

            # 如果YOLO已给出明确类别且置信度>0.5，直接信任YOLO
            if (item.item_name and item.item_name != "未知物品"
                    and item.item_name in self.items_config
                    and item.confidence > 0.5):
                matched_items.append(item)
                continue

            # 否则用几何模板匹配（HSV回退或YOLO低置信度时）
            scores = {}
            for name, cfg in self.items_config.items():
                score = self._compute_match_score(item, cfg, w, h)
                scores[name] = score

            best_name = max(scores, key=scores.get)
            best_score = scores[best_name]

            if best_score > 0.2:
                item.item_name = best_name
                item.confidence = best_score
            else:
                item.item_name = "未知物品"
                item.confidence = 0.1

            matched_items.append(item)

        # 去重：同一类别只保留置信度最高的
        matched_items = self._deduplicate_by_category(matched_items)

        return matched_items

    # ─── 单项得分计算 ──────────────────────────────────

    def _compute_match_score(
        self,
        item: DetectedItem,
        cfg: dict,
        frame_w: int,
        frame_h: int,
    ) -> float:
        """计算候选物品与某类模板的匹配得分"""

        # 1. 面积得分
        area_score = self._range_score(
            item.contour_area, cfg["area_range"]
        )

        # 2. 宽高比得分
        aspect_score = self._range_score(
            item.aspect_ratio, cfg["aspect_ratio_range"]
        )

        # 3. 圆度得分
        circ_min = cfg["circularity_min"]
        if item.circularity >= circ_min:
            circ_score = 1.0
        else:
            circ_score = item.circularity / max(circ_min, 0.001)

        # 4. 位置得分
        pos_score = self._position_score(
            item.centroid, cfg.get("position_hint", ""), frame_w, frame_h
        )

        # 5. 加权汇总
        total = (
            self.WEIGHT_AREA * area_score
            + self.WEIGHT_ASPECT * aspect_score
            + self.WEIGHT_CIRCULARITY * min(circ_score, 1.0)
            + self.WEIGHT_POSITION * pos_score
        )

        return total

    # ─── 辅助函数 ──────────────────────────────────────

    @staticmethod
    def _range_score(value: float, valid_range: List[float]) -> float:
        """值越接近范围中心，得分越高"""
        lo, hi = valid_range
        if lo <= value <= hi:
            center = (lo + hi) / 2
            half_range = (hi - lo) / 2
            dist = abs(value - center)
            return 1.0 - 0.5 * (dist / max(half_range, 0.001))
        elif value < lo:
            return max(0.0, 0.5 * value / max(lo, 0.001))
        else:
            return max(0.0, 0.5 * hi / max(value, 0.001))

    @staticmethod
    def _position_score(
        centroid: Tuple[float, float],
        hint: str,
        frame_w: int,
        frame_h: int,
    ) -> float:
        """位置合理性评分"""
        cx, cy = centroid
        nx, ny = cx / frame_w, cy / frame_h  # 归一化坐标

        hint_scores = {
            "center": lambda: 1.0 - abs(nx - 0.5) * 1.5 - abs(ny - 0.5) * 1.5,
            "right_of_gaiwan": lambda: 1.0 - abs(nx - 0.6) * 2.0 - abs(ny - 0.5) * 1.5,
            "multiple_grouped": lambda: 1.0 - abs(ny - 0.6) * 2.0,
            "near_gaiwan": lambda: 1.0 - abs(nx - 0.5) * 2.0 - abs(ny - 0.5) * 1.5,
            "edge": lambda: max(abs(nx - 0.5) * 1.5, abs(ny - 0.5) * 1.5),
            "edge_area": lambda: max(abs(nx - 0.5) * 1.5, abs(ny - 0.5) * 1.5),
            "side_area": lambda: abs(nx - 0.5) * 2.0,
            "center_bottom": lambda: 1.0 - abs(nx - 0.5) * 1.0 - abs(ny - 0.7) * 2.0,
        }

        scorer = hint_scores.get(hint)
        if scorer:
            return max(0.0, min(1.0, scorer()))
        return 0.5  # 未知提示，给中性分

    @staticmethod
    def _deduplicate_by_category(items: List[DetectedItem]) -> List[DetectedItem]:
        """
        按类别去重：同一类别只保留置信度最高的。
        特殊处理「品茗杯」（多个聚集）和「未知物品」（保留）。
        """
        category_best: Dict[str, DetectedItem] = {}
        multi_items = {"品茗杯", "未知物品"}

        for item in items:
            name = item.item_name
            if name in multi_items:
                continue  # 保留多个
            if name not in category_best or item.confidence > category_best[name].confidence:
                category_best[name] = item

        # 保留多实例类别 + 唯一类别的胜出者
        result = []
        for item in items:
            if item.item_name in multi_items:
                result.append(item)
            elif category_best.get(item.item_name) is item:
                result.append(item)

        return result

    # ─── 汇总统计 ──────────────────────────────────────

    def get_checklist(
        self, matched_items: List[DetectedItem],
    ) -> Dict[str, dict]:
        """
        生成物品清单检查结果。

        Returns:
            {item_name: {essential, detected, confidence, quantity}}
        """
        # 统计检出物品
        detected_counts: Dict[str, int] = {}
        detected_confs: Dict[str, float] = {}
        for item in matched_items:
            name = item.item_name
            if name != "未知物品":
                detected_counts[name] = detected_counts.get(name, 0) + 1
                detected_confs[name] = max(detected_confs.get(name, 0.0), item.confidence)

        checklist = {}
        for name, cfg in self.items_config.items():
            count = detected_counts.get(name, 0)
            qty_range = cfg.get("quantity_range", [1, 1])
            detected = qty_range[0] <= count <= qty_range[1] if qty_range else count > 0

            checklist[name] = {
                "name_cn": cfg["name_cn"],
                "essential": cfg["essential"],
                "present": count > 0,
                "detected": detected,
                "count": count,
                "expected_range": qty_range,
                "confidence": round(detected_confs.get(name, 0.0), 3),
            }

        return checklist

    # ─── 摆放合理性评分 ────────────────────────────────

    def get_placement_score(
        self,
        matched_items: List[DetectedItem],
        frame_shape: Tuple[int, int],
    ) -> float:
        """
        评估物品摆放合理性（0-1）。

        对每个已匹配的非"未知物品"，使用其对应模板的 position_hint
        计算位置得分，按 confidence 加权平均。

        Args:
            matched_items: ItemMatcher.match() 的输出
            frame_shape: 推理帧尺寸 (h, w)

        Returns:
            0-1 分值（1 = 所有物品都在预期位置）
        """
        h, w = frame_shape
        total_weight = 0.0
        weighted_sum = 0.0

        for item in matched_items:
            name = item.item_name
            if name == "未知物品" or name not in self.items_config:
                continue

            cfg = self.items_config[name]
            hint = cfg.get("position_hint", "")
            if not hint:
                continue

            pos = self._position_score(item.centroid, hint, w, h)
            weight = max(item.confidence, 0.1)
            weighted_sum += pos * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.5  # 无法评估时返回中性分

    # ─── 操作规范性评分 ────────────────────────────────

    def get_area_normality_score(
        self,
        matched_items: List[DetectedItem],
        occluded_count: int = 0,
        occlusion_penalty_per_item: float = 0.10,
    ) -> float:
        """
        评估操作区域规范性（0-1）。

        1. 基于画面整洁度：未知物品越少越好
        2. 遮挡扣分：被手/身体遮挡的物品，每件扣固定比例

        无未知物品且无遮挡 → 1.0

        Args:
            matched_items: ItemMatcher.match() 的输出
            occluded_count: 已确认被遮挡的物品数量
            occlusion_penalty_per_item: 每件遮挡物品的扣分比例 (默认 0.10 = 10%)

        Returns:
            0-1 分值
        """
        if not matched_items:
            return 1.0 - occluded_count * occlusion_penalty_per_item

        unknown_count = sum(1 for it in matched_items if it.item_name == "未知物品")
        total = len(matched_items)

        base = max(0.0, 1.0 - unknown_count / total)
        penalty = occluded_count * occlusion_penalty_per_item
        return max(0.0, base - penalty)

    # ─── 汇总统计 ──────────────────────────────────────

    def compute_score(self, checklist: Dict[str, dict]) -> Tuple[int, int, float]:
        """
        计算得分。

        Returns:
            (detected_count, total_essential, score_percentage)
        """
        essential_found = sum(
            1 for name in self.essential_items
            if checklist[name]["detected"]
        )
        total_essential = len(self.essential_items)
        score = (essential_found / total_essential) * 100 if total_essential > 0 else 0
        return essential_found, total_essential, score

    def get_verdict(self, essential_found: int, total_essential: int) -> Tuple[str, str]:
        """
        根据检出数量判定等级。

        Returns:
            (verdict_label, verdict_color)
        """
        ratio = essential_found / max(total_essential, 1)

        if ratio >= 1.0:
            return "A · 优秀", "#27ae60"
        elif ratio >= 0.8:
            return "B · 良好", "#2ecc71"
        elif ratio >= 0.6:
            return "C · 合格", "#f39c12"
        elif ratio >= 0.4:
            return "D · 不合格", "#e74c3c"
        else:
            return "E · 严重不足", "#c0392b"
