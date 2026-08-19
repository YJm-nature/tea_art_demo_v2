"""
茶具实时检测 Demo — Streamlit Web 版

功能:
- 支持摄像头实时检测 / 上传视频检测
- 9类茶具识别: 盖碗、公道杯、品茗杯、茶荷、茶巾、茶夹、茶拨、茶盘、建水
- 实时显示检测框 + 类别标签 + 置信度
- 右侧统计面板: 检出/缺失清单 + FPS

启动: streamlit run demo_app.py
"""
import streamlit as st
import cv2, time, sys, os
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[2]
VIDEO_INBOX = PROJECT / "dataset" / "tea_sop_modular_v1" / "raw_videos" / "00_inbox"

# ===== 页面配置 =====
st.set_page_config(
    page_title="茶具实时检测 Demo",
    page_icon="🍵",
    layout="wide",
)

# ===== 加载模型 =====
@st.cache_resource
def load_model():
    model_paths = [
        PROJECT / "models" / "tea_ware_train" / "weights" / "best.pt",
        PROJECT / "models" / "tea_ware_best.pt",
    ]
    for mp in model_paths:
        if os.path.exists(mp):
            return YOLO(mp)
    st.error("未找到模型文件！请先运行 scripts/3_train.py 训练模型")
    return None

CLASS_NAMES = ["盖碗", "公道杯", "品茗杯", "茶荷", "茶巾", "茶夹", "茶拨", "茶盘", "建水"]
COLORS = [
    (0, 255, 128),    # 盖碗 - 绿色
    (255, 128, 0),    # 公道杯 - 橙色
    (0, 200, 255),    # 品茗杯 - 黄色
    (255, 0, 128),    # 茶荷 - 粉色
    (128, 0, 255),    # 茶巾 - 紫色
    (0, 255, 255),    # 茶夹 - 青色
    (255, 255, 0),    # 茶拨 - 天蓝
    (128, 255, 0),    # 茶盘 - 黄绿
    (255, 80, 80),    # 建水 - 红色
]

# ===== 标题 =====
st.title("🍵 茶具实时检测 Demo")
st.caption("YOLOv8 | 9类茶具 | 实时推理")

# ===== 侧边栏 =====
with st.sidebar:
    st.header("⚙️ 设置")
    source = st.radio("视频来源", ["📹 摄像头", "📁 上传视频", "🎬 默认茶艺视频"])

    conf_threshold = st.slider("置信度阈值", 0.1, 0.9, 0.35, 0.05)
    iou_threshold = st.slider("IoU 阈值", 0.1, 0.9, 0.45, 0.05)

    st.divider()
    st.markdown("**检测类别 (9类)**")
    for cn in CLASS_NAMES:
        st.markdown(f"- {cn}")

    st.divider()
    st.caption("模型: YOLOv8n | 输入: 1280×720")

# ===== 主区域 =====
col_left, col_right = st.columns([2, 1])

with col_left:
    video_placeholder = st.empty()

with col_right:
    st.subheader("📊 检测统计")
    stats_placeholder = st.empty()

    st.subheader("📋 物品清单")
    checklist_placeholder = st.empty()

# ===== 视频源 =====
uploaded_file = None
if source == "📁 上传视频":
    uploaded_file = st.sidebar.file_uploader("选择视频文件", type=["mp4", "mov", "avi", "mkv"])
elif source == "🎬 默认茶艺视频":
    default_video = str(VIDEO_INBOX / "茶具检测.MP4")
    if not os.path.exists(default_video):
        st.error(f"找不到默认视频: {default_video}")

# ===== 主循环 =====
if st.sidebar.button("▶ 开始检测", type="primary", use_container_width=True):
    model = load_model()
    if model is None:
        st.stop()

    # 初始化视频源
    if source == "📹 摄像头":
        cap = cv2.VideoCapture(0)
    elif source == "📁 上传视频" and uploaded_file:
        tmp_path = f"temp_{int(time.time())}.mp4"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(tmp_path)
    else:
        cap = cv2.VideoCapture(default_video)

    fps_avg = 0
    frame_count = 0
    stop_btn = st.sidebar.button("⏹ 停止检测")

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            if source == "🎬 默认茶艺视频":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        frame_count += 1
        frame = cv2.resize(frame, (1280, 720))

        t0 = time.perf_counter()
        results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
        t1 = time.perf_counter()

        fps = 1.0 / max(t1 - t0, 0.001)
        fps_avg = 0.9 * fps_avg + 0.1 * fps

        # 解析结果
        detected = {}
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()

                    if cls_id < len(CLASS_NAMES):
                        cls_name = CLASS_NAMES[cls_id]
                        color = COLORS[cls_id]

                        # 只保留最高置信度
                        if cls_name not in detected or conf > detected[cls_name]["conf"]:
                            detected[cls_name] = {"conf": conf, "xyxy": xyxy, "color": color}

        # 画检测框
        annotated = frame.copy()
        for name, info in detected.items():
            x1, y1, x2, y2 = map(int, info["xyxy"])
            color = info["color"]
            conf = info["conf"]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # FPS 叠加
        cv2.putText(annotated, f"FPS: {fps_avg:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示
        video_placeholder.image(annotated, channels="BGR", use_container_width=True)

        # 统计面板
        detected_count = len(detected)
        total_count = len(CLASS_NAMES)

        with col_right:
            stats_placeholder.markdown(f"""
            | 指标 | 值 |
            |------|-----|
            | 检出 | **{detected_count}** / {total_count} |
            | 检出率 | **{detected_count/total_count*100:.0f}%** |
            | FPS | **{fps_avg:.1f}** |
            | 帧数 | {frame_count} |
            """)

            # 物品清单
            lines = []
            for cn in CLASS_NAMES:
                if cn in detected:
                    lines.append(f"✅ **{cn}** `{detected[cn]['conf']:.2f}`")
                else:
                    lines.append(f"❌ {cn}")
            checklist_placeholder.markdown("\n".join(lines))

    cap.release()
    st.sidebar.success("检测已停止")

else:
    # 未开始时的占位
    video_placeholder.info("👈 点击左侧「开始检测」按钮启动")

    st.markdown("---")
    st.markdown("""
    ### 📖 使用说明

    1. **选择视频来源**: 摄像头 / 上传视频 / 默认茶艺视频
    2. **调整参数**: 置信度阈值越低检出越多但误检也越多
    3. **点击「开始检测」**: 实时显示检测结果

    ### 🎯 检测目标 (9类)

    | 类别 | 说明 |
    |------|------|
    | 盖碗 | 白瓷三才碗，主泡器 |
    | 公道杯 | 玻璃材质，分茶用 |
    | 品茗杯 | 白瓷小杯，带杯托 |
    | 茶荷 | 长条形/椭圆形浅盘 |
    | 茶巾 | 方形布巾，折叠放置 |
    | 茶夹 | 竹制长夹 |
    | 茶拨 | 竹制细长拨杆 |
    | 茶盘 | 大面积矩形托盘 |
    | 建水 | 白色大碗，盛废水 |
    """)
