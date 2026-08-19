# 茶艺红茶 — 摄像头实时检测与目标跟踪 Demo

基于《0611修订茶艺红茶操作流程需求确认表》，构建 SOP 步骤一「备具布席」茶具实时检测 Demo。

当前主入口已统一为 **OpenCV 摄像头实时检测 + ByteTrack 目标跟踪**：

- 茶具检测：YOLOv8 自定义模型 + HSV 回退
- 目标跟踪：Ultralytics ByteTrack，显示目标 ID 与运动轨迹
- 跨帧记忆：`DetectionMemory`，检出过就记住，不因短暂遮挡/移出画面误扣分
- 手部/上肢：MediaPipe Hands / Pose，用于遮挡检测
- 可审计评分：可观测得分、完整需求覆盖率、跨帧证据可靠度
- 报告导出：按 `E` 保存包含模型、阈值和逐项证据的 JSON

---

## 快速开始

> 仓库保留源码、配置、测试和主入口使用的精选检测权重。原始视频、完整数据集、
> 训练中间产物及其他实验权重体积较大，不纳入 Git 版本控制。

完整的数据集创建、MakeSense 审核、YOLOv8n 训练、验证、导出和预标注教程：

`docs/通用YOLO模型训练教程_茶艺项目版.md`

```bash
cd tea_art_demo_v2
pip install -r requirements.txt
python realtime_tracking_demo.py --source camera --camera-id 0 --track
```

Windows 用户可直接双击：

```bat
run.bat
```

首次运行如缺少 MediaPipe 模型，会自动下载到 `models/` 目录。

---

## 运行方式

### 摄像头实时检测（默认推荐）

```bash
python realtime_tracking_demo.py --source camera --camera-id 0 --track
```

### 指定视频文件回放

```bash
python realtime_tracking_demo.py --source video --video "E:\path\demo.mp4" --track
```

### 指定模型权重

```bash
python realtime_tracking_demo.py --source camera --profile auto
```

未传 `--model` 时，程序默认加载当前检测权重：

```text
models/low_vram/front_detect_selected_holdout_stage1-2/weights/best.pt
```

该权重会自动绑定 `tea18_warm_clean` profile，启用已训练的茶具类别和烧水壶。
命令行传入 `--model` 和 `--profile` 时仍可覆盖默认设置。器具 Pose 模型是独立
权重，只有显式传入 `--pose-model` 时才运行。

动作自由观测模式按固定机位运行：

```powershell
python realtime_tracking_demo.py --source video --video <桌面视频> --camera-role tabletop
python realtime_tracking_demo.py --source video --video <正侧面视频> --camera-role side
```

桌面与正侧面双视频串行运行（GTX 1060 6GB，不并发执行 YOLO）：

```powershell
python multiview_observation_demo.py --table-video <桌面视频> --side-video <正侧面视频>
```

离线批处理可增加 `--headless`，并用 `--max-frames` 限制试运行帧数。

按 `E` 保存观测事件报告。未提供 `--accessory-model` 时饰品观测明确显示为 `uncertain`；旧9类模型不能区分碗身和碗盖，闻香观测同样不会强制判错。

### 性能调试选项

```bash
python realtime_tracking_demo.py --source camera --no-hand --no-pose
python realtime_tracking_demo.py --source camera --width 960 --height 540
python realtime_tracking_demo.py --source camera --no-track
```

GTX 1060 6GB 推荐运行参数：

```bash
python realtime_tracking_demo.py --source camera --imgsz 832 --process-every 2 --hand-every 3 --pose-every 3
```

完整18类模型训练完成后会自动匹配 `tea18` profile；盖碗碗身会映射到步骤一的盖碗评分，碗盖和显示屏仍保留给后续状态/OCR模块。

---

## 按键控制

| 按键 | 功能 |
|---|---|
| `Q` / `Esc` | 退出 |
| `Space` | 暂停 / 继续 |
| `S` | 截图到 `output/screenshots/` |
| `E` | 导出当前步骤一评分报告到 `output/reports/` |
| `T` | 开关 ByteTrack 目标跟踪 |
| `+` / `-` | 调整检测置信度阈值 |
| `R` | 视频模式下重播 |

---

## 模型类别 Profile

项目里曾同时存在 9 类训练模型和 13 类业务配置。为避免类别 ID 错映射，主入口会在启动时读取 YOLO 权重自带的 `model.names`，并自动匹配 profile：

### `tea9`：9 类模型

```text
0 盖碗
1 公道杯
2 品茗杯
3 茶荷
4 茶巾
5 茶夹
6 茶拨
7 茶叶罐
8 建水
```

9 类模式只对模型支持的 9 类进行 checklist 和评分，不会把模型无法检测的烧水壶、电子秤、温度计、计时器计入缺失。

### `tea13`：13 类模型

```text
0 盖碗
1 公道杯
2 品茗杯
3 茶荷
4 茶巾
5 茶夹
6 茶拨
7 茶盘
8 烧水壶
9 建水
10 电子秤
11 温度计
12 计时器
```

如果模型类别数量或顺序与 `config/model_profiles.json` 不一致，程序会直接报错退出，避免静默错判。

---

## 项目结构

```text
tea_art_demo_v2/
├── realtime_tracking_demo.py      # 当前唯一推荐主入口：摄像头/视频 + 检测 + 跟踪 + 评分
├── run.bat                        # Windows 一键启动主入口
├── requirements.txt
│
├── src/
│   ├── model_config.py            # 模型路径、tea9/tea13 profile 校验
│   ├── tea_detector.py            # YOLOv8 + HSV 双模茶具检测
│   ├── object_tracker.py          # ByteTrack 封装 + 轨迹管理
│   ├── capture_source.py          # 摄像头/视频源统一打开
│   ├── realtime_pipeline.py       # 单帧检测、匹配、记忆、评分管线
│   ├── item_matcher.py            # 几何模板匹配 + 摆放/规范评分
│   ├── detection_memory.py        # 跨帧累积记忆 + 遮挡检测
│   ├── hand_detector.py           # MediaPipe Hands 封装
│   ├── pose_detector.py           # MediaPipe Pose 上肢提取
│   ├── draw_utils.py              # 检测框、目标轨迹、骨架、信息面板绘制
│   └── observation_point.py       # 观测点框架
│
├── config/
│   ├── tea_items.json             # 茶具几何模板库
│   └── model_profiles.json        # tea9 / tea13 类别 profile
│
├── models/                        # 模型权重与 MediaPipe 模型
├── dataset/                       # YOLO 数据集
├── scripts/                       # 数据抽帧、拆分、训练、离线可视化辅助脚本
├── output/                        # 运行输出
└── archive/                       # 旧 Demo、实验脚本和缓存归档
```

---

## Demo 评分口径

```text
当前模型可观测得分 = 必备品存在完整度 × 80% + 数量正确性 × 10% + 布局启发式 × 10%
```

步骤一完整需求有 10 个必备品，当前 tea9 模型只覆盖其中 8 个，因而报告会同时显示 80% 需求覆盖率和保守的覆盖率调整分。茶叶罐属于后续步骤物品，不计入步骤一必备品。

证据可靠度由检测置信度和至少 15 帧的稳定检出计算，只用于说明机器证据质量，不给学员加分。布局目前仍是单机位位置启发式，尚不能代替“1.5 米操作半径”和教师审美判断。该公式为 demo 暂定口径，不是已确认的完整 SOP 验收规则。

---

## 数据集状态提醒

当前推荐数据集 `dataset/final_tea9_dataset_20260723` 有 848 张图片、8229 个实例，来自 3 个场景。现有 train/val 是按帧随机拆分，同一场景同时存在于两边，并检测到 545 对相邻帧泄漏，因此训练日志的验证指标不能作为跨场景验收结果。

运行只读审计：

```bash
python scripts/audit_dataset.py dataset/final_tea9_dataset_20260723
```

按完整来源场景重建到新目录：

```bash
python scripts/group_split_dataset.py dataset/final_tea9_dataset_20260723 dataset/tea9_scene_holdout --val-sources office
```

注意：旧的 `scripts/2_split_dataset.py` 会原地清理和拆分，不要用于正式版本数据。新的分组脚本只写全新输出目录，并拒绝覆盖非空目录。

完整架构、六步实现拆解、采集标注规范和验收路线见 [docs/完整项目设计与迭代方案.md](docs/完整项目设计与迭代方案.md)。

GTX 1060 6GB 数据重建与训练工具已经放入 `scripts/`：

- `prepare_review_dataset.py`：无损合并旧848张图片并迁移到18类审核工作区；
- `review_dataset.py`：逐图查看、状态记录和接受门禁；
- `extract_active_learning.py`：从补采视频抽取去重、高价值候选帧；
- `publish_reviewed_dataset.py`：按session隔离发布train/val/test；
- `validate_release_dataset.py`：正式训练前质量门禁；
- `train_6gb.py`：smoke、640基线、960微调、分割、测试、基准和ONNX导出。

详细命令见 [docs/数据集重建与6GB训练操作手册.md](docs/数据集重建与6GB训练操作手册.md)。

---

## 旧入口归档

旧的 Streamlit Demo、快速检测 Demo、摄像头烟雾测试和跟踪实验脚本会保留在 `archive/legacy_demos/`，用于回滚和对照，不再作为推荐入口。
