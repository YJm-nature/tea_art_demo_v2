"""
茶艺红茶操作流程 — AI视觉实时检测 Demo V2

观测点：备具布席 — 物品准备检测
界面：Streamlit Web
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 添加项目路径
PROJECT = Path(__file__).resolve().parents[2]
VIDEO_INBOX = PROJECT / "dataset" / "tea_sop_modular_v1" / "raw_videos" / "00_inbox"
sys.path.insert(0, str(PROJECT))

from src.video_reader import VideoReader
from src.tea_detector import TeaDetector
from src.item_matcher import ItemMatcher
from src.draw_utils import draw_detections, draw_info_panel, draw_hand_skeleton, draw_pose_skeleton
from src.scoring import ScoringEngine
from src.detection_memory import DetectionMemory
from src.observation_point import OBSERVATION_REGISTRY, ObservationResult
from src.hand_detector import HandDetector
from src.pose_detector import PoseDetector

# ─── 页面配置 ─────────────────────────────────────────

st.set_page_config(
    page_title="茶艺红茶 · AI视觉检测",
    page_icon="🍵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2c3e50;
        padding: 0.5rem 0;
        border-bottom: 2px solid #e74c3c;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        text-align: center;
        margin: 0.3rem 0;
    }
    .stat-card.green { background: linear-gradient(135deg, #2ecc71, #27ae60); }
    .stat-card.orange { background: linear-gradient(135deg, #f39c12, #e67e22); }
    .stat-card.red { background: linear-gradient(135deg, #e74c3c, #c0392b); }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
    }
    .stat-label {
        font-size: 0.8rem;
        opacity: 0.9;
    }
    .check-item {
        padding: 0.3rem 0.5rem;
        border-radius: 6px;
        margin: 0.15rem 0;
        font-size: 0.85rem;
    }
    .check-item.found { background: #d5f5e3; color: #1e8449; }
    .check-item.missing { background: #fadbd8; color: #c0392b; }
    .grade-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 1.2rem;
        font-weight: 700;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ─── 初始化（缓存）─────────────────────────────────────

@st.cache_resource
def get_detector() -> TeaDetector:
    return TeaDetector()


@st.cache_resource
def get_matcher() -> ItemMatcher:
    return ItemMatcher()


@st.cache_resource
def get_hand_detector() -> HandDetector:
    return HandDetector(detect_every_n_frames=2)


@st.cache_resource
def get_pose_detector() -> PoseDetector:
    return PoseDetector(detect_every_n_frames=2)


# ─── 主函数 ──────────────────────────────────────────

def main():
    st.markdown(
        '<div class="main-header">🍵 茶艺红茶操作流程 — AI视觉实时检测</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"观测点：备具布席 · 物品准备检测  |  SOP Step 1/6  |  上传限制：{st.config.get_option('server.maxUploadSize')}MB")

    # 初始化 session state
    defaults = {
        "video_loaded": False,
        "processing": False,
        "current_frame": None,
        "frame_idx": 0,
        "checklist": {},
        "score": 0.0,
        "grade": "-",
        "grade_color": "#95a5a6",
        "detected_count": 0,
        "total_essential": 10,
        "fps": 0.0,
        "detection_history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ── 工具栏 ──
    col_t1, col_t2, col_t3, col_t4 = st.columns([2, 1, 1, 1])
    with col_t1:
        video_source = st.radio(
            "视频来源",
            ["使用默认茶艺视频", "上传视频文件"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with col_t3:
        sample_rate = st.selectbox("帧采样率", [1, 2, 3, 5], index=1,
                                   help="每N帧处理1帧，数值越大越快但可能漏检")
    with col_t4:
        if st.button("🔄 重新检测", use_container_width=True):
            st.session_state.video_loaded = False
            st.rerun()

    # ── 视频路径 ──
    video_path = None
    if video_source == "使用默认茶艺视频":
        default_videos = [
            VIDEO_INBOX / "VID_20260612_093947.mp4",
            VIDEO_INBOX / "VID_20260612_094215.mp4",
            VIDEO_INBOX / "VID_20260612_094722.mp4",
        ]
        existing = [v for v in default_videos if os.path.exists(v)]
        if existing:
            default_video = str(existing[0])  # 默认用最小的
            st.success(f"📁 {os.path.basename(default_video)}")
            if st.button("▶ 开始检测", type="primary", use_container_width=True):
                video_path = default_video
                st.session_state.video_loaded = True
        else:
            st.warning("未找到默认视频，请上传")
    else:
        uploaded = st.file_uploader("上传茶艺操作视频", type=["mp4", "mov", "avi", "mkv"])
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.read())
                video_path = tmp.name
            st.success(f"✅ {uploaded.name}")
            if st.button("▶ 开始检测", type="primary", use_container_width=True):
                st.session_state.video_loaded = True

    st.divider()

    # ── 主区域：左右分栏 ──
    col_left, col_right = st.columns([5, 3])

    # ── 左侧：视频画面 ──
    with col_left:
        video_placeholder = st.empty()

    # ── 右侧：检测结果面板 ──
    with col_right:
        # 统计卡片
        st.markdown("##### 📊 检测统计")
        metric_cols = st.columns(4)
        metrics = [
            ("总数", "total_essential", "blue"),
            ("检出", "detected_count", "green"),
            ("缺失", "missing_count", "orange"),
            ("得分", "score_display", "red"),
        ]

        metric_placeholders = {}
        for i, (label, key, color) in enumerate(metrics):
            with metric_cols[i]:
                metric_placeholders[key] = st.empty()

        # 评分等级
        st.markdown("##### 🏆 评分等级")
        grade_placeholder = st.empty()

        # 评分维度明细
        st.markdown("##### 📐 评分维度")
        dimension_placeholder = st.empty()

        # 物品清单
        st.markdown("##### 📋 物品清单")
        checklist_placeholder = st.empty()

        # 图例
        with st.expander("图例说明", expanded=False):
            st.markdown(
                "🟢 **绿色框** = 高置信度匹配  |  "
                "🟠 **橙色框** = 中等置信度  |  "
                "🔵 **蓝色框** = 低置信度  |  "
                "⚪ **灰色框** = 未能识别"
            )
            st.markdown(
                "✅ **绿色勾** = 已检出  |  "
                "❌ **红色叉** = 未检出  |  "
                "🟡 **黄色勾** = 可选品已检出"
            )

    # ── 处理视频 ──
    if video_path and st.session_state.video_loaded:
        run_detection(
            video_path, sample_rate,
            video_placeholder,
            metric_placeholders,
            grade_placeholder,
            checklist_placeholder,
            dimension_placeholder,
        )


def run_detection(
    video_path: str,
    sample_rate: int,
    video_placeholder,
    metric_placeholders: dict,
    grade_placeholder,
    checklist_placeholder,
    dimension_placeholder,
):
    """执行视频检测主循环"""
    detector = get_detector()
    matcher = get_matcher()
    hand_detector = get_hand_detector()
    pose_detector = get_pose_detector()

    try:
        reader = VideoReader(video_path, sample_every_n=sample_rate)
    except Exception as e:
        st.error(f"无法打开视频: {e}")
        return

    # 显示视频信息
    info_text = (
        f"📹 {reader.original_width}x{reader.original_height} | "
        f"{reader.fps:.0f}fps | "
        f"{reader.total_frames}帧 | "
        f"{reader.duration:.1f}秒"
    )
    st.caption(info_text)

    # ── 跨帧累积记忆（解决"品茗杯移出画面导致误扣分"） ──
    memory = DetectionMemory()

    # 主循环
    progress_bar = st.progress(0)
    status_text = st.empty()

    for original_frame, inference_frame in reader.read_all_frames():
        t_start = time.time()

        # ── 检测 ──
        items = detector.detect(inference_frame)

        # ── 匹配 ──
        h_inf, w_inf = inference_frame.shape[:2]
        matched_items = matcher.match(items, (h_inf, w_inf))

        # ── 摆放合理性评分 ──
        placement_score = matcher.get_placement_score(matched_items, (h_inf, w_inf))

        # ── 逐帧清单（用于画面 overlay，看到什么画什么） ──
        checklist_realtime = matcher.get_checklist(matched_items)

        # ── 手部检测（MediaPipe Hands） ──
        hand_results = hand_detector.detect(inference_frame)
        hand_bboxes = [h["bbox"] for h in hand_results]
        hand_count = len(hand_results)

        # ── 姿态检测（MediaPipe Pose → 上肢 bbox） ──
        pose_results = pose_detector.detect(inference_frame)
        arm_bboxes = pose_detector.get_arm_bboxes(inference_frame)

        # ── 累积记忆（用于打分，见过就记住，只升不降） ──
        memory.accumulate(
            matched_items, reader.frame_idx,
            hand_bboxes=hand_bboxes,
            arm_bboxes=arm_bboxes,
        )
        checklist_scoring = memory.get_checklist(matcher.items_config)
        essential_found, total_essential, score = matcher.compute_score(checklist_scoring)
        grade, grade_color = matcher.get_verdict(essential_found, total_essential)

        # ── 操作规范性（需在 accumulate 之后，occlusion_count 依赖手部检测结果） ──
        normality_score = matcher.get_area_normality_score(
            matched_items, occluded_count=memory.occluded_count
        )

        detected_optional = sum(
            1 for name in matcher.optional_items
            if checklist_scoring.get(name, {}).get("detected", False)
        )

        # ── 路由到观测点框架 ──
        obs = OBSERVATION_REGISTRY.get("obj_utensils_s1")
        if obs is not None:
            obs_result = obs.detect(inference_frame, context={
                "checklist": checklist_scoring,
                "essential_found": essential_found,
                "total_essential": total_essential,
                "score": score,
                "grade": grade,
                "grade_color": grade_color,
                "placement_score": placement_score,
                "normality_score": normality_score,
                "hand_count": hand_count,
            })
        else:
            obs_result = None

        # ── 更新session state ──
        st.session_state.checklist = checklist_scoring
        st.session_state.score = score
        st.session_state.grade = grade
        st.session_state.grade_color = grade_color
        st.session_state.detected_count = essential_found
        st.session_state.total_essential = total_essential
        st.session_state.placement_score = placement_score
        st.session_state.normality_score = normality_score
        st.session_state.occluded_count = memory.occluded_count
        st.session_state.occluded_items = memory.occluded_items
        st.session_state.hand_count = hand_count
        st.session_state.frame_idx = reader.frame_idx

        # ── 绘制标注（用实时检测结果，看到什么画什么） ──
        annotated = draw_detections(inference_frame, matched_items)
        # 叠加手部骨架
        if hand_results:
            annotated = draw_hand_skeleton(annotated, hand_results)
        # 叠加上肢骨架
        if pose_results:
            annotated = draw_pose_skeleton(annotated, pose_results)

        # 信息面板（用累积记忆打分）
        panel_data = {
            "step_name": "备具布席",
            "detected_count": essential_found,
            "total_count": total_essential,
            "score": score,
            "grade": grade,
            "grade_color": grade_color,
            "fps": st.session_state.fps,
            "frame_idx": reader.frame_idx,
        }
        annotated = draw_info_panel(annotated, panel_data)

        # 显示
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)

        # ── 更新右侧面板 ──
        # 统计卡片
        with metric_placeholders["total_essential"]:
            st.markdown(
                f'<div class="stat-card blue">'
                f'<div class="stat-number">{total_essential}</div>'
                f'<div class="stat-label">必备品</div></div>',
                unsafe_allow_html=True,
            )
        with metric_placeholders["detected_count"]:
            st.markdown(
                f'<div class="stat-card green">'
                f'<div class="stat-number">{essential_found}</div>'
                f'<div class="stat-label">已检出</div></div>',
                unsafe_allow_html=True,
            )
        with metric_placeholders["missing_count"]:
            missing = total_essential - essential_found
            card_class = "green" if missing == 0 else "orange"
            st.markdown(
                f'<div class="stat-card {card_class}">'
                f'<div class="stat-number">{missing}</div>'
                f'<div class="stat-label">未检出</div></div>',
                unsafe_allow_html=True,
            )
        with metric_placeholders["score_display"]:
            # 三维加权得分（含遮挡扣分）
            weighted_display = (
                score * 0.40
                + placement_score * 100 * 0.30
                + normality_score * 100 * 0.30
            )
            st.markdown(
                f'<div class="stat-card red">'
                f'<div class="stat-number">{weighted_display:.0f}</div>'
                f'<div class="stat-label">得分/100</div></div>',
                unsafe_allow_html=True,
            )

        # 等级
        with grade_placeholder:
            st.markdown(
                f'<div style="text-align:center;padding:0.5rem;">'
                f'<span class="grade-badge" style="background:{grade_color};">'
                f'{grade}</span></div>',
                unsafe_allow_html=True,
            )

        # 评分维度明细
        with dimension_placeholder:
            # 三维加权得分
            item_pct = score  # 检出率得分
            placement_pct = placement_score * 100
            normality_pct = normality_score * 100
            weighted = (
                item_pct * 0.40 + placement_pct * 0.30 + normality_pct * 0.30
            )
            _render_dimension_scores(
                item_pct, placement_pct, normality_pct, weighted,
                occluded_count=memory.occluded_count,
                occluded_items=memory.occluded_items,
                hand_bboxes=hand_bboxes,
            )

        # 物品清单（用累积记忆——见过就显示检出，🧠=历史记忆）
        with checklist_placeholder:
            _render_checklist(checklist_scoring, matcher, reader.frame_idx)

        # 进度条
        progress_bar.progress(reader.progress)
        # 观测点判定信息
        obs_info = ""
        if obs_result is not None:
            obs_info = f" | 观测点: {obs_result.verdict.value}"

        status_text.text(
            f"帧: {reader.frame_idx}/{reader.total_frames} | "
            f"已处理: {reader.processed_idx}帧"
            f"{obs_info}"
        )

        # FPS
        t_end = time.time()
        st.session_state.fps = 0.7 * st.session_state.fps + 0.3 / max(t_end - t_start, 0.001)

    reader.close()
    progress_bar.progress(1.0)
    status_text.text("✅ 检测完成！")
    st.balloons()


def _render_checklist(checklist: dict, matcher: ItemMatcher, current_frame: int = 0):
    """
    渲染物品清单。

    checklist: 基于累积记忆的清单（DetectionMemory.get_checklist()）
    每个物品会标注来源：当前帧检出 | 历史记忆
    """
    # 必备品
    st.markdown("**必备品 (10项)**")
    for name in matcher.essential_items:
        info = checklist.get(name, {})
        detected = info.get("detected", False)
        conf = info.get("confidence", 0.0)
        count = info.get("count", 0)
        name_cn = info.get("name_cn", name)
        last_seen = info.get("last_seen", -1)

        if detected:
            # 判断是当前帧检出还是历史记忆
            in_sight = (last_seen >= 0 and current_frame - last_seen <= 3)
            if in_sight:
                icon = "✅"
                tip = ""
            else:
                icon = "🧠"  # 记忆中的物品（当前帧未看到但历史检出过）
                tip = f" <small>(记忆, 最后帧#{last_seen})</small>"

            st.markdown(
                f'<div class="check-item found">'
                f'{icon} {name_cn} ×{count} <small>({conf:.0%})</small>{tip}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="check-item missing">'
                f'❌ {name_cn} <small>未检出</small></div>',
                unsafe_allow_html=True,
            )

    # 可选品
    st.markdown("**可选品 (3项)**")
    for name in matcher.optional_items:
        info = checklist.get(name, {})
        detected = info.get("detected", False)
        conf = info.get("confidence", 0.0)
        name_cn = info.get("name_cn", name)
        last_seen = info.get("last_seen", -1)

        if detected:
            in_sight = (last_seen >= 0 and current_frame - last_seen <= 3)
            icon = "🟡" if in_sight else "🧠"
            st.markdown(
                f'<div class="check-item found">'
                f'{icon} {name_cn} <small>({conf:.0%})</small></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="check-item" style="background:#f0f0f0;color:#999;">'
                f'⬜ {name_cn} <small>未检出</small></div>',
                unsafe_allow_html=True,
            )


def _render_dimension_scores(
    item_pct: float, placement_pct: float, normality_pct: float, weighted: float,
    occluded_count: int = 0, occluded_items: set = None,
    hand_bboxes: list = None,
):
    """
    渲染三维评分明细。

    显示公式：物品检出率×0.40 + 摆放合理性×0.30 + 操作规范性×0.30 = 综合得分
    如有遮挡且检测到手部，显示遮挡扣分明细。
    """
    # 手部检测状态
    hand_info = ""
    if hand_bboxes is not None:
        n_hands = len(hand_bboxes)
        if n_hands > 0:
            hand_info = f"""
        <div style="margin:2px 0;color:#2980b9;font-size:0.72rem;">
            ✋ 检测到 {n_hands} 只手
        </div>"""

    occlusion_html = ""
    if occluded_count > 0 and occluded_items:
        names = "、".join(sorted(occluded_items))
        occlusion_html = f"""
        <div style="margin:2px 0;color:#c0392b;font-size:0.72rem;">
            ⚠ 手部遮挡扣分：{occluded_count}件（{names}）每件 -10% 规范性
        </div>"""

    bars_html = f"""
    <div style="font-size:0.78rem;line-height:1.6;color:#555;padding:0.2rem 0;">
        <div style="margin:2px 0;">
            <span style="display:inline-block;width:90px;">📦 物品检出率</span>
            <span style="color:#888;">× 40%</span>
            <span style="float:right;font-weight:600;">{item_pct:.0f}</span>
        </div>
        <div style="margin:2px 0;">
            <span style="display:inline-block;width:90px;">📍 摆放合理性</span>
            <span style="color:#888;">× 30%</span>
            <span style="float:right;font-weight:600;">{placement_pct:.0f}</span>
        </div>
        <div style="margin:2px 0;">
            <span style="display:inline-block;width:90px;">🧹 操作规范性</span>
            <span style="color:#888;">× 30%</span>
            <span style="float:right;font-weight:600;">{normality_pct:.0f}</span>
        </div>
        {hand_info}
        {occlusion_html}
        <div style="border-top:1px solid #ddd;margin-top:3px;padding-top:3px;">
            <span style="font-weight:700;">📐 综合得分</span>
            <span style="float:right;font-weight:700;font-size:0.9rem;">{weighted:.0f}</span>
        </div>
    </div>
    """
    st.markdown(bars_html, unsafe_allow_html=True)


# ─── 入口 ────────────────────────────────────────────

if __name__ == "__main__":
    main()
