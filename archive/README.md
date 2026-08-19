# Archive

这里保存项目整理前的旧入口、实验脚本和本地缓存，方便回滚或对照。

## legacy_demos

旧 Demo 入口：

- `streamlit_app.py`：原 `app.py`，Streamlit Web 版，仅支持默认视频/上传视频。
- `camera_demo.py`：原摄像头检测烟雾测试脚本，无目标跟踪。
- `demo_app.py`：原简化 Streamlit Demo。
- `realtime_demo.py`：原完整 OpenCV 视频检测/评分脚本。
- `realtime_demo_fast.py`：原快速检测脚本。
- `realtime_demo_track.py`：原 ByteTrack 跟踪实验脚本。

当前推荐主入口为项目根目录的 `realtime_tracking_demo.py`。

## experiments

开发期检查或旧数据处理脚本：

- `inspect_model.py`
- `extract_frames.py`

## cache

IDE 和 Python 缓存归档。此目录内容不参与运行。
