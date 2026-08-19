"""
标注绘制模块 — 在视频帧上叠加检测框、标签、信息面板
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .tea_detector import DetectedItem


# ─── 颜色常量 ─────────────────────────────────────────

COLOR_GREEN = (0, 220, 80)
COLOR_ORANGE = (0, 180, 255)
COLOR_RED = (60, 60, 255)
COLOR_BLUE = (220, 140, 60)
COLOR_WHITE = (240, 240, 240)
COLOR_GRAY = (160, 160, 160)
COLOR_DARK = (35, 35, 45)
COLOR_BG_TRANSPARENT = (30, 30, 40)


def draw_detections(
    frame: np.ndarray,
    items: List[DetectedItem],
    show_label: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    在帧上绘制所有检测框和标签。

    Args:
        frame: 原始帧（会被原地修改或返回新帧）
        items: 检测到的物品列表
        show_label: 是否显示物品名称
        show_confidence: 是否显示置信度

    Returns:
        标注后的帧
    """
    output = frame.copy()

    text_entries = []
    for item in items:
        x, y, w, h = item.bbox

        # 根据物品类型选择颜色
        name = item.item_name
        if name == "未知物品":
            color = COLOR_GRAY
        elif item.confidence > 0.6:
            color = COLOR_GREEN
        elif item.confidence > 0.35:
            color = COLOR_ORANGE
        else:
            color = COLOR_BLUE

        # 绘制矩形框
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        # 绘制标签
        if show_label:
            label_parts = []
            if getattr(item, "track_id", None) is not None:
                label_parts.append(f"#{item.track_id}")
            if name != "未知物品":
                label_parts.append(name)
            if show_confidence:
                label_parts.append(f"{item.confidence:.0%}")

            label = " ".join(label_parts)
            if label:
                tw, th = _measure_text(label, 0.45)
                text_y = y - 4 if y >= th + 6 else min(output.shape[0] - 3, y + th + 4)
                cv2.rectangle(
                    output,
                    (x, max(0, text_y - th - 2)),
                    (min(output.shape[1] - 1, x + tw + 4), min(output.shape[0] - 1, text_y + 3)),
                    color, -1,
                )
                text_entries.append((label, (x + 2, text_y), 0.45, (0, 0, 0), 1))

    _draw_text_batch(output, text_entries)
    return output


def draw_info_panel(
    frame: np.ndarray,
    panel_data: dict,
    position: str = "top_right",
) -> np.ndarray:
    """
    在帧上绘制信息面板。

    Args:
        frame: 输入帧
        panel_data: {
            "step_name": str,
            "detected_count": int,
            "total_count": int,
            "score": float,
            "grade": str,
            "fps": float,
            "frame_idx": int,
        }
        position: 面板位置 "top_right" | "top_left" | "bottom_right"

    Returns:
        带面板的帧
    """
    output = frame.copy()
    h, w = output.shape[:2]

    panel_w, panel_h = 340, 224

    if position == "top_right":
        px, py = w - panel_w - 15, 15
    elif position == "top_left":
        px, py = 15, 15
    else:
        px, py = w - panel_w - 15, h - panel_h - 15

    # 半透明背景
    overlay = output.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), COLOR_DARK, -1)
    cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)

    text_entries = []
    y = py + 25
    lh = 24

    # 标题
    text_entries.append(("茶艺实时检测", (px + 12, y), 0.5, (180, 200, 255), 1))
    y += lh + 4

    # 分隔线
    cv2.line(output, (px + 12, y), (px + panel_w - 12, y), (80, 80, 100), 1)
    y += 8

    # 物品检出统计
    detected = panel_data.get("detected_count", 0)
    total = panel_data.get("total_count", 10)
    score = panel_data.get("score", 0)
    grade = panel_data.get("grade", "-")

    text_entries.append((f"Items Detected: {detected}/{total}", (px + 12, y), 0.48, COLOR_WHITE, 1))
    y += lh

    # 得分
    text_entries.append((f"Score: {score:.0f}/100", (px + 12, y), 0.48, (120, 255, 120), 1))
    y += lh

    coverage = panel_data.get("requirement_coverage")
    reliability = panel_data.get("evidence_reliability")
    if coverage is not None:
        evidence_text = f" | Evidence:{reliability:.0f}%" if reliability is not None else ""
        text_entries.append((
            f"Coverage: {coverage * 100:.0f}%{evidence_text}",
            (px + 12, y), 0.42, COLOR_GRAY, 1,
        ))
        y += lh

    # 等级
    grade_color_hex = panel_data.get("grade_color", "#f39c12")
    bgr = _hex_to_bgr(grade_color_hex)
    text_entries.append((f"Grade: {grade}", (px + 12, y), 0.55, bgr, 2))
    y += lh

    # FPS
    fps = panel_data.get("fps", 0)
    text_entries.append((
        f"FPS: {fps:.1f}  |  Frame: {panel_data.get('frame_idx', 0)}",
        (px + 12, y), 0.42, COLOR_GRAY, 1,
    ))
    y += lh

    mode = panel_data.get("mode")
    profile = panel_data.get("profile")
    if mode or profile:
        mode_text = mode or "DETECT"
        profile_text = f" | {profile}" if profile else ""
        text_entries.append((
            f"Mode: {mode_text}{profile_text}", (px + 12, y), 0.42, COLOR_GRAY, 1,
        ))

    _draw_text_batch(output, text_entries)
    return output


def draw_tracks(frame: np.ndarray, trajectories: dict) -> np.ndarray:
    """绘制 ByteTrack 轨迹线。"""
    output = frame.copy()
    for _track_id, trail in trajectories.items():
        if len(trail) < 2:
            continue
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            thickness = max(1, int(alpha * 3))
            cv2.line(output, trail[i - 1], trail[i], (150, 150, 150), thickness, cv2.LINE_AA)
    return output


def draw_accessory_detections(frame: np.ndarray, detections: List[dict]) -> np.ndarray:
    """Draw dedicated hand-ROI accessory detections."""
    output = frame.copy()
    for item in detections:
        x, y, w, h = [int(value) for value in item.get("bbox", (0, 0, 0, 0))]
        cv2.rectangle(output, (x, y), (x + w, y + h), COLOR_RED, 2)
        label = f"Accessory:{item.get('class_id', '?')} {float(item.get('confidence', 0)):.0%}"
        cv2.putText(output, label, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_RED, 1)
    return output


def draw_observation_panel(
    frame: np.ndarray,
    observations: dict,
    camera_role: str,
    position: str = "top_left",
) -> np.ndarray:
    """Display temporal observation states without treating them as scores."""
    output = frame.copy()
    if not observations:
        return output
    h, w = output.shape[:2]
    rows = list(observations.values())[:8]
    panel_w = min(480, max(320, w - 30))
    row_height = 46
    panel_h = 56 + len(rows) * row_height
    px = 15 if position == "top_left" else max(15, w - panel_w - 15)
    py = 15
    overlay = output.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), COLOR_DARK, -1)
    cv2.addWeighted(overlay, 0.78, output, 0.22, 0, output)
    role_name = {
        "tabletop": "桌面",
        "side": "侧面",
        "single": "单摄像头",
    }.get(camera_role, camera_role)
    text_entries = [(
        f"动作观测 | 机位:{role_name}",
        (px + 12, py + 22), 0.48, COLOR_WHITE, 1,
    )]
    y = py + 50
    state_colors = {
        "completed": COLOR_GREEN,
        "active": COLOR_ORANGE,
        "candidate": COLOR_BLUE,
        "uncertain": COLOR_GRAY,
        "failed": COLOR_RED,
        "idle": COLOR_GRAY,
    }
    for snapshot in rows:
        state = getattr(getattr(snapshot, "state", None), "value", "unknown")
        text = _format_observation_snapshot(snapshot)
        text_entries.append((
            text, (px + 12, y), 0.4, state_colors.get(state, COLOR_GRAY), 1,
        ))
        reason = str(getattr(snapshot, "reason", "") or "")
        if reason:
            text_entries.append((
                reason[:32], (px + 24, y + 18), 0.34, COLOR_GRAY, 1,
            ))
        y += row_height
    _draw_text_batch(output, text_entries)
    return output


def draw_vessel_pose(
    frame: np.ndarray,
    pose_results: List[dict],
    minimum_keypoint_confidence: float = 0.25,
) -> np.ndarray:
    """Draw vessel outlet/center/rear keypoints produced by YOLO pose."""
    output = frame.copy()
    point_colors = (COLOR_ORANGE, COLOR_GREEN, COLOR_BLUE)
    text_entries = []
    for row in pose_results:
        point_names = (
            ("端点A", "中心", "端点B")
            if row.get("class_name") == "茶荷"
            else ("出水端", "中心", "后端")
        )
        points = np.asarray(row.get("keypoints", []), dtype=float)
        confidences = np.asarray(
            row.get("keypoint_confidences", []), dtype=float
        )
        if points.shape != (3, 2) or confidences.shape != (3,):
            continue
        valid_points = []
        for index, ((x, y), confidence) in enumerate(zip(points, confidences)):
            if confidence < minimum_keypoint_confidence or x <= 0 or y <= 0:
                valid_points.append(None)
                continue
            point = (int(round(x)), int(round(y)))
            valid_points.append(point)
            cv2.circle(output, point, 6, point_colors[index], -1)
            cv2.circle(output, point, 8, COLOR_WHITE, 1)
            text_entries.append((
                f"{point_names[index]} {confidence:.0%}",
                (point[0] + 7, max(14, point[1] - 7)),
                0.32,
                point_colors[index],
                1,
            ))
        for left, right in ((0, 1), (1, 2)):
            if valid_points[left] is not None and valid_points[right] is not None:
                cv2.line(
                    output, valid_points[left], valid_points[right],
                    COLOR_WHITE, 2, cv2.LINE_AA,
                )
    _draw_text_batch(output, text_entries)
    return output


def draw_sop_panel(
    frame: np.ndarray,
    sop_state: Optional[dict],
    observations: Optional[dict] = None,
    position: str = "left",
) -> np.ndarray:
    """Show compact SOP order and progress in the upper-left corner."""
    if not sop_state:
        return frame.copy()
    output = frame.copy()
    h, w = output.shape[:2]
    steps = list(sop_state.get("steps", []))
    runtime = dict(sop_state.get("runtime", {}))
    current_id = sop_state.get("current_step_id")
    current_index = next(
        (index for index, step in enumerate(steps)
         if step.get("step_id") == current_id),
        None,
    )
    business_groups = []
    for step in steps:
        group_id = step.get("business_step") or step.get("step_id")
        if not any(item["id"] == group_id for item in business_groups):
            business_groups.append({
                "id": group_id,
                "name": step.get("business_step_name") or step.get("name") or group_id,
                "steps": [],
            })
        next(item for item in business_groups if item["id"] == group_id)["steps"].append(step)
    current_group_index = next((
        index for index, group in enumerate(business_groups)
        if any(step.get("step_id") == current_id for step in group["steps"])
    ), None)
    completed_groups = sum(
        all(runtime.get(step.get("step_id"), {}).get("status") in {"completed", "skipped"}
            for step in group["steps"])
        for group in business_groups
    )
    panel_w = min(400, max(400, w - 30))
    panel_h = 160
    px = max(15, w - panel_w - 15) if position == "right" else 15
    py = 15
    overlay = output.copy()
    cv2.rectangle(
        overlay, (px, py), (px + panel_w, py + panel_h), COLOR_DARK, -1
    )
    cv2.addWeighted(overlay, 0.82, output, 0.18, 0, output)

    mode = sop_state.get("mode", "free_observation")
    status = sop_state.get("status", "running")
    mode_label = "顺序模式" if mode == "strict" else "自由观测"
    status_label = {
        "running": "运行中",
        "completed": "全部完成",
        "failed": "已中止",
        "needs_review": "需要复核",
    }.get(status, str(status))
    text_entries = [
        (f"SOP {mode_label} | {status_label}", (px + 12, py + 23),
         0.48, COLOR_WHITE, 1),
        (f"步骤 {completed_groups}/{len(business_groups)}", (px + panel_w - 130, py + 23),
         0.42, COLOR_GREEN, 1),
    ]
    cv2.line(
        output, (px + 12, py + 34), (px + panel_w - 12, py + 34),
        (80, 80, 100), 1,
    )

    if current_index is None:
        prompt = "全部动作已完成" if status == "completed" else "自由观测中"
        text_entries.append((prompt, (px + 12, py + 62), 0.52, COLOR_GREEN, 1))
    else:
        current = steps[current_index]
        current_group = business_groups[current_group_index] if current_group_index is not None else None
        current_name = current.get("name") or current.get("step_id")
        text_entries.append((
            f"第{(current_group_index or 0) + 1}步：{current_group['name'] if current_group else current_name}",
            (px + 12, py + 62),
            0.52, COLOR_ORANGE, 1,
        ))
        text_entries.append((
            f"当前动作：{current_name}",
            (px + 12, py + 88), 0.43, COLOR_WHITE, 1,
        ))
        flow = " → ".join(str(value) for value in current.get("action_flow", []))
        flow_lines = _wrap_panel_text(f"流程：{flow}", 34, 2) if flow else []
        for line_index, line in enumerate(flow_lines):
            text_entries.append((
                line, (px + 12, py + 115 + line_index * 21),
                0.36, COLOR_GRAY, 1,
            ))
    _draw_text_batch(output, text_entries)
    return output


def draw_step_details_panel(
    frame: np.ndarray,
    sop_state: Optional[dict],
    observations: Optional[dict] = None,
) -> np.ndarray:
    """Show action requirements and live judgments in the upper-right corner."""
    if not sop_state or not sop_state.get("current_step_id"):
        return frame.copy()
    output = frame.copy()
    h, w = output.shape[:2]
    steps = list(sop_state.get("steps", []))
    runtime = dict(sop_state.get("runtime", {}))
    current_id = sop_state.get("current_step_id")
    current = next(
        (step for step in steps if step.get("step_id") == current_id), None
    )
    if current is None:
        return output
    group_id = current.get("business_step") or current_id
    group_steps = [
        step for step in steps
        if (step.get("business_step") or step.get("step_id")) == group_id
    ]
    panel_w = min(610, max(430, w - 30))
    row_h = 102
    panel_h = min(h - 80, 54 + row_h * len(group_steps))
    px, py = max(15, w - panel_w - 15), 15
    overlay = output.copy()
    cv2.rectangle(
        overlay, (px, py), (px + panel_w, py + panel_h), COLOR_DARK, -1
    )
    cv2.addWeighted(overlay, 0.82, output, 0.18, 0, output)
    group_name = current.get("business_step_name") or current.get("name") or group_id
    text_entries = [(
        f"{group_name} | 动作检测与要求",
        (px + 12, py + 24), 0.5, COLOR_WHITE, 1,
    )]
    cv2.line(
        output, (px + 12, py + 36), (px + panel_w - 12, py + 36),
        (80, 80, 100), 1,
    )
    state_labels = {
        "pending": "待检测",
        "active": "当前",
        "completed": "完成",
        "failed": "失败",
        "skipped": "跳过",
        "needs_review": "复核",
    }
    state_colors = {
        "active": COLOR_ORANGE,
        "completed": COLOR_GREEN,
        "failed": COLOR_RED,
        "needs_review": COLOR_RED,
    }
    observation_state_labels = {
        "idle": "等待动作",
        "candidate": "正在确认",
        "active": "识别中",
        "completed": "已完成",
        "failed": "未通过",
        "uncertain": "暂时无法判断",
    }
    y = py + 58
    for step in group_steps:
        step_id = step.get("step_id")
        state = runtime.get(step_id, {}).get("status", "pending")
        color = state_colors.get(state, COLOR_GRAY)
        label = (
            "不合格/继续"
            if state == "skipped" and runtime.get(step_id, {}).get("skip_reason")
            else state_labels.get(state, state)
        )
        text_entries.append((
            f"[{label}] {step.get('name') or step_id}"[:44],
            (px + 14, y), 0.42, color, 1,
        ))
        requirements = "；".join(str(value) for value in step.get("requirements", []))
        snapshot = (observations or {}).get(step.get("observation_id"))
        detail = requirements or "等待进入该动作"
        for line_index, line in enumerate(
            _wrap_panel_text(f"要求：{detail}", 39, 2)
        ):
            text_entries.append((
                line, (px + 28, y + 22 + line_index * 20),
                0.34, COLOR_WHITE if step_id == current_id else COLOR_GRAY, 1,
            ))
        if step_id == current_id:
            observation_state = getattr(
                getattr(snapshot, "state", None), "value", "idle"
            )
            judgment_label = observation_state_labels.get(
                observation_state, "等待动作"
            )
            live_reason = str(
                getattr(snapshot, "reason", "") or "正在等待检测结果"
            )
            for line_index, line in enumerate(
                _wrap_panel_text(
                    f"判断：{judgment_label} | {live_reason}", 39, 2
                )
            ):
                text_entries.append((
                    line, (px + 28, y + 65 + line_index * 20),
                    0.35, COLOR_ORANGE, 1,
                ))
        y += row_h
        if y > py + panel_h - 12:
            break
    _draw_text_batch(output, text_entries)
    return output


def _wrap_panel_text(text: str, max_chars: int, max_lines: int) -> List[str]:
    """Wrap compact Chinese UI text and mark any remaining truncation."""
    value = str(text).strip()
    if not value:
        return []
    lines = [value[index:index + max_chars] for index in range(0, len(value), max_chars)]
    if len(lines) <= max_lines:
        return lines
    visible = lines[:max_lines]
    visible[-1] = visible[-1][:-1] + "…"
    return visible


def _format_observation_snapshot(snapshot: object) -> str:
    state = getattr(getattr(snapshot, "state", None), "value", "unknown")
    value = getattr(snapshot, "value", None)
    state_label = {
        "idle": "等待",
        "candidate": "检测中",
        "active": "进行中",
        "completed": "已完成",
        "failed": "未通过",
        "uncertain": "无法判断",
    }.get(state, state.upper())
    if value == "其他布局":
        state_label = "不规则"
    name = getattr(snapshot, "name", None) or getattr(snapshot, "observation_id", "未知观测")
    value_text = (
        f" | {value}" if value not in (None, "", False, "其他布局") else ""
    )
    confidence = float(getattr(snapshot, "confidence", 0))
    experimental = " | 实验" if getattr(snapshot, "experimental", False) else ""
    return f"{name}: {state_label}{value_text} {confidence:.0%}{experimental}"


def draw_controls(
    frame: np.ndarray,
    paused: bool = False,
    tracking_enabled: bool = True,
    conf: Optional[float] = None,
    show_replay: bool = False,
) -> np.ndarray:
    """在画面底部绘制统一控制栏。"""
    output = frame.copy()
    h, w = output.shape[:2]
    bar_h = 34
    y0 = h - bar_h
    cv2.rectangle(output, (0, y0), (w, h), (25, 25, 30), -1)

    tips = [
        ("Q 退出", (80, 200, 80)),
        ("Space 暂停" if not paused else "Space 继续", (200, 180, 60)),
        ("S 截图", (120, 140, 220)),
        ("E 报告", (100, 190, 210)),
        ("T 追踪开关", (255, 160, 60) if tracking_enabled else (150, 150, 150)),
        ("N 跳过步骤", (90, 170, 220)),
        ("X 重置SOP", (90, 190, 150)),
        ("+/- 阈值", (180, 140, 220)),
    ]
    if show_replay:
        tips.append(("R 重播", (200, 140, 255)))

    text_entries = []
    x_pos = 8
    for tip, color in tips:
        tw, _ = _measure_text(tip, 0.4)
        cv2.rectangle(output, (x_pos, y0 + 4), (x_pos + tw + 10, h - 6), color, -1)
        text_entries.append((tip, (x_pos + 5, h - 10), 0.4, (0, 0, 0), 1))
        x_pos += tw + 16

    if conf is not None:
        text_entries.append((
            f"CONF:{conf:.2f}", (x_pos + 10, h - 10), 0.4, COLOR_GRAY, 1,
        ))
    _draw_text_batch(output, text_entries)
    return output


def draw_mask_overlay(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    在帧上叠加半透明颜色掩码（调试用，显示白色分割区域）。

    Args:
        frame: 原始帧
        mask: 二值掩码
        alpha: 透明度

    Returns:
        叠加后的帧
    """
    colored_mask = np.zeros_like(frame)
    colored_mask[mask > 0] = (0, 255, 100)
    output = cv2.addWeighted(frame, 1.0, colored_mask, alpha, 0)
    return output


# ─── 辅助函数 ─────────────────────────────────────────

def _put_text(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    scale: float = 0.5,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
):
    """Draw Unicode text using a Windows CJK font when available."""
    _draw_text_batch(img, [(text, pos, scale, color, thickness)])


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WINDOWS_DIR = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
_font_candidates = [
    _PROJECT_ROOT / "assets" / "fonts" / "NotoSansCJK-Regular.ttc",
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
]
if _WINDOWS_DIR:
    windows_fonts = Path(_WINDOWS_DIR) / "Fonts"
    _font_candidates[1:1] = [
        windows_fonts / "msyh.ttc",
        windows_fonts / "simhei.ttf",
        windows_fonts / "simsun.ttc",
    ]
_FONT_CANDIDATES = tuple(_font_candidates)


@lru_cache(maxsize=1)
def _resolve_font_path() -> Optional[str]:
    configured = os.environ.get("TEA_UI_FONT")
    if configured and Path(configured).is_file():
        return configured
    for candidate in _FONT_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return None


@lru_cache(maxsize=16)
def _load_font(pixel_size: int):
    font_path = _resolve_font_path()
    if font_path:
        return ImageFont.truetype(font_path, pixel_size)
    return ImageFont.load_default(size=pixel_size)


def _font_for_scale(scale: float):
    return _load_font(max(12, int(round(scale * 32))))


def _measure_text(text: str, scale: float) -> Tuple[int, int]:
    bbox = _font_for_scale(scale).getbbox(text)
    return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])


def _draw_text_batch(img: np.ndarray, entries: list) -> None:
    if not entries:
        return
    canvas = Image.fromarray(img)
    draw = ImageDraw.Draw(canvas)
    for text, pos, scale, color, thickness in entries:
        draw.text(
            (int(pos[0]), int(pos[1])),
            str(text),
            font=_font_for_scale(float(scale)),
            fill=tuple(int(channel) for channel in color),
            anchor="ls",
            stroke_width=max(0, int(thickness) - 1),
        )
    img[:] = np.asarray(canvas)


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """16进制颜色转BGR"""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


# ─── 手部骨架绘制 ─────────────────────────────────────

# MediaPipe Hands 21 个关键点之间的连接（手指骨架 + 手掌）
_HAND_CONNECTIONS = [
    # 拇指 (Thumb)
    (0, 1), (1, 2), (2, 3), (3, 4),
    # 食指 (Index)
    (0, 5), (5, 6), (6, 7), (7, 8),
    # 中指 (Middle)
    (0, 9), (9, 10), (10, 11), (11, 12),
    # 无名指 (Ring)
    (0, 13), (13, 14), (14, 15), (15, 16),
    # 小指 (Pinky)
    (0, 17), (17, 18), (18, 19), (19, 20),
    # 手掌 (Palm arch)
    (5, 9), (9, 13), (13, 17),
]

# 指尖关键点索引（用于高亮绘制）
_FINGERTIP_IDS = {4, 8, 12, 16, 20}


def draw_hand_skeleton(
    frame: np.ndarray,
    hand_results: list,
) -> np.ndarray:
    """
    在帧上绘制手部骨架。

    Args:
        frame: 输入帧（会被修改）
        hand_results: HandDetector.detect() 返回的手部列表

    Returns:
        标注后的帧（原地修改 + 返回）
    """
    for hand in hand_results:
        landmarks = hand["landmarks"]  # (21, 3) 像素坐标
        handedness = hand.get("handedness", "Unknown")

        # 左右手不同颜色
        if handedness == "Left":
            line_color = (255, 140, 60)     # 蓝色
            point_color = (255, 100, 30)
        else:
            line_color = (60, 220, 100)     # 绿色
            point_color = (40, 200, 80)

        # ── 绘制骨架连线 ──
        for i, j in _HAND_CONNECTIONS:
            pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
            pt2 = (int(landmarks[j][0]), int(landmarks[j][1]))
            cv2.line(frame, pt1, pt2, line_color, 2, cv2.LINE_AA)

        # ── 绘制关键点 ──
        for idx, (x, y, _z) in enumerate(landmarks):
            px, py = int(x), int(y)
            is_tip = idx in _FINGERTIP_IDS

            # 指尖用大圆，其余用小圆
            radius = 5 if is_tip else 3
            color = (60, 160, 255) if is_tip else point_color  # 指尖用黄色
            cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)
            if is_tip:
                cv2.circle(frame, (px, py), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# ─── 人体姿态骨架绘制 ─────────────────────────────────

# 用于显示的关键连接：双肩 + 双臂
_DRAW_POSE_CONNECTIONS = [
    # 躯干连线
    (11, 12),                     # 左肩 ↔ 右肩
    # 左臂
    (11, 13), (13, 15),           # 肩 → 肘 → 腕
    # 右臂
    (12, 14), (14, 16),           # 肩 → 肘 → 腕
]

# 关节大小
_POSE_JOINT_RADII = {
    11: 5, 12: 5,    # 肩部 — 大圆
    13: 3, 14: 3,    # 肘部 — 中圆
    15: 2, 16: 2,    # 腕部 — 小圆（与 HandDetector 手腕重叠）
}


def draw_pose_skeleton(
    frame: np.ndarray,
    pose_results: list,
) -> np.ndarray:
    """
    在帧上绘制人体上肢骨架（双肩 + 双臂）。

    Args:
        frame: 输入帧（会被修改）
        pose_results: PoseDetector.detect() 返回的列表

    Returns:
        标注后的帧
    """
    color_line = (200, 180, 100)     # 浅蓝灰色骨架线
    color_joint = (120, 200, 255)    # 金色关节点

    for pose in pose_results:
        landmarks = pose.get("landmarks")
        if landmarks is None:
            continue

        # 绘制连线
        for i, j in _DRAW_POSE_CONNECTIONS:
            if i < len(landmarks) and j < len(landmarks):
                pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
                pt2 = (int(landmarks[j][0]), int(landmarks[j][1]))
                # 跳过不可见点（坐标为零）
                if pt1 == (0, 0) or pt2 == (0, 0):
                    continue
                cv2.line(frame, pt1, pt2, color_line, 2, cv2.LINE_AA)

        # 绘制关节点
        for idx, radius in _POSE_JOINT_RADII.items():
            if idx < len(landmarks):
                px, py = int(landmarks[idx][0]), int(landmarks[idx][1])
                if px == 0 and py == 0:
                    continue
                cv2.circle(frame, (px, py), radius, color_joint, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), radius, (255, 255, 255), 1, cv2.LINE_AA)

    return frame
