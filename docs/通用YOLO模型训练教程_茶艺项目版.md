# 通用 YOLO 模型训练教程（茶艺项目版）

本文用于以后独立完成：创建数据集、抽取视频帧、标注、审核、划分数据集、训练 YOLO、验证模型和重新预标注。

适用环境：Windows、Python 3.11、GTX 1060 6GB、Ultralytics YOLOv8n。

## 一、先准备环境

所有命令都在项目根目录执行：

```powershell
Set-Location E:\tea_culture\tea_art_demo_v2
$PY = "D:\anaconda3\envs\tea-ai-6gb\python.exe"
```

检查 Python、CUDA 和显卡：

```powershell
& $PY --version
& $PY -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

正式训练必须看到：

```text
Python 3.11.x
CUDA: True
NVIDIA GeForce GTX 1060
```

训练前关闭浏览器硬件加速、播放器和其他 GPU 程序。不要同时训练两个模型。

## 二、选择数据集类型

### 1. 普通器具检测

适合识别盖碗、公道杯、品茗杯、茶荷等静态目标。标签格式为 YOLO Detection。

### 2. 动作观测数据

温杯、托举、闻香、注水、出汤等动作不直接作为 YOLO 类别。它们使用器具检测框、手部关键点、人体姿态、跟踪和 SOP 状态机判断。

因此，动作视频通常需要抽取两类素材：

- 器具检测帧：标注茶具框。
- 动作事件 JSONL：记录动作的起止时间、正例、错误例和困难负例。

### 3. 关键点模型

只有在需要判断壶嘴、公道杯出口、盖碗沿等方向时，才建立 YOLO Pose 数据集。不要把“倾斜 10 度、20 度”作为检测类别。

## 三、创建可复用的数据集目录

本项目推荐一个数据集根目录对应一个版本，不覆盖旧版本：

```text
dataset/
└── tea_sop_front_v1/
    ├── raw_videos/              # 原始视频，只读保存
    │   ├── front_s01/           # 一个连续拍摄 session
    │   ├── front_s02/
    │   └── front_s08/
    ├── derived/                 # 抽帧后的候选图片
    │   ├── detection/images/
    │   └── pose/images/
    ├── annotations/             # 审核后的标签
    │   ├── detection_yolo18/labels/
    │   ├── pose_yolo/labels/
    │   └── actions/events.jsonl
    ├── annotation_batches/      # MakeSense 审核批次
    ├── releases/                # 可训练发布版本
    │   └── detection/front_detect_reviewed_v1/
    ├── manifests/               # 视频、session、帧索引记录
    └── reports/                 # 审核和数据质量报告
```

首次创建正面机位目录：

```powershell
& $PY -B scripts\prepare_front_dataset.py
```

该命令会创建 `front_s01` 到 `front_s08`。同一段连续视频只能属于一个 session，不能把相邻帧拆到 train 和 val/test。

## 四、建立类别文件

YOLO 类别编号从 0 开始，训练过程中不能随意调整已有编号。当前茶艺检测固定 18 类：

```text
0  盖碗（碗身）
1  盖碗（碗盖）
2  公道杯
3  品茗杯
4  茶荷
5  茶巾
6  茶夹
7  茶拨
8  茶盘
9  烧水壶
10 建水
11 电子秤
12 温度计
13 计时器
14 茶叶罐
15 茶则
16 水壶显示屏
17 电子秤显示屏
```

类别文件位置：

```text
dataset\tea_dataset_v1_reviewed\classes.txt
```

增加新类别时：

1. 只能追加到文件末尾。
2. 所有旧标签编号保持不变。
3. 新类别至少准备多个 session 的正例、遮挡例和负例。
4. 修改 `data.yaml` 的 `nc` 和 `names`。
5. 重新训练并重新验证，不能直接混用旧模型。

## 五、视频拍摄和抽帧

建议每个 session 固定机位、分辨率和光照。动作视频前后各保留约 2 秒静止画面；同一个视频尽量只改变一种因素，例如遮挡、距离或动作速度。

把视频放入对应目录，例如：

```text
dataset\tea_sop_front_v1\raw_videos\front_s01\
```

抽取首轮检测候选帧：

```powershell
& $PY -B scripts\extract_front_frames.py front_s01 `
  --detection-limit 100 `
  --pose-limit 50
```

候选帧必须人工筛选：

- 删除模糊帧、重复帧和完全相同的静止帧。
- 保留拿起、放下、遮挡、交互开始和结束等状态变化。
- 每个类别不能只来自一个视频。
- 空桌、只有手、相似物体和错误动作要保留一部分作为负样本。

旧版通用抽帧脚本也可以处理单个视频：

```powershell
& $PY -B scripts\1_extract_frames.py `
  --video dataset\tea_sop_modular_v1\raw_videos\00_inbox\茶具检测.MP4 `
  --output dataset\images `
  --count 150
```

## 六、标注 YOLO Detection

每张图片对应一个同名 `.txt` 文件：

```text
图片：frame_0001.jpg
标签：frame_0001.txt
```

每行格式：

```text
class_id center_x center_y width height
```

全部坐标必须归一化到 `0~1`，例如：

```text
0 0.512500 0.480000 0.260000 0.310000
3 0.720000 0.650000 0.100000 0.120000
```

标注原则：

- 框贴合目标可见区域，不要把整张桌面框进去。
- 盖碗碗身和碗盖必须分开标注。
- 被手部分遮挡时，框仍按目标可见及可推断的完整边界标注，所有图片保持同一规则。
- 图片中没有目标时，保留空的 `.txt` 文件。
- 不要把动作名称写入器具类别文件。

MakeSense 批量审核建议每批 100 张或 200 张。导出的修订 ZIP、图片和 `batch_manifest.csv` 一起保存，不要只保留截图。

## 七、把审核批次发布成正式数据集

当前项目全部批次审核完成后，运行：

```powershell
& $PY -B scripts\build_reviewed_front_release.py
```

默认输出：

```text
dataset\tea_sop_front_v1\releases\detection\front_detect_reviewed_v1\
```

标准发布目录必须包含：

```text
front_detect_reviewed_v1/
├── data.yaml
├── train/images/       train/labels/
├── val/images/         val/labels/
├── test/images/        test/labels/
├── classes.txt
├── manifest.csv
└── release_summary.yaml
```

重建已存在的发布目录时，必须明确使用：

```powershell
& $PY -B scripts\build_reviewed_front_release.py --force
```

不要手动删除原数据。发布脚本会优先使用 MakeSense 修订标签；空背景图片若未出现在 ZIP 中，会生成空标签。

## 八、检查数据集后再训练

先查看 `data.yaml`：

```powershell
Get-Content dataset\tea_sop_front_v1\releases\detection\front_detect_reviewed_v1\data.yaml
```

检查以下项目：

- `nc: 18`。
- `train`、`val`、`test` 路径存在。
- 类别编号与 `classes.txt` 一致。
- 同一连续视频没有跨 split。
- 每张图片有同名标签文件。
- 空背景图片确实没有目标。

正式训练脚本会自动检查 Python、CUDA、数据路径和类别数量。先做 dry-run：

```powershell
& $PY -B scripts\train_6gb.py --dry-run `
  stage1 `
  dataset\tea_sop_front_v1\releases\detection\front_detect_reviewed_v1\data.yaml `
  --model yolov8n.pt `
  --name front_detect_reviewed_stage1_640
```

## 九、训练 YOLOv8n

### 1. 冒烟训练

先用 100 张图片训练 3 个 epoch，确认标签和显存：

```powershell
& $PY -B scripts\train_6gb.py `
  smoke `
  dataset\tea_sop_front_v1\releases\detection\front_detect_reviewed_v1\data.yaml `
  --model yolov8n.pt `
  --name front_detect_reviewed_smoke
```

### 2. 640 基线训练

```powershell
& $PY -B scripts\train_6gb.py `
  stage1 `
  dataset\tea_sop_front_v1\releases\detection\front_detect_reviewed_v1\data.yaml `
  --model yolov8n.pt `
  --name front_detect_reviewed_stage1_640
```

默认配置：

```text
imgsz=640
batch=4
epochs=100
patience=20
workers=0
cache=false
amp=true
optimizer=AdamW
```

如果显存不足，脚本自动尝试 batch 2 和 batch 1。模型输出目录：

```text
models\low_vram\front_detect_reviewed_stage1_640\weights\best.pt
```

### 3. 960 小目标微调

茶夹、茶拨、碗盖等小目标需要高分辨率时，从 stage1 的 `best.pt` 继续：

```powershell
& $PY -B scripts\train_6gb.py `
  stage2 `
  dataset\tea_sop_front_v1\releases\detection\front_detect_reviewed_v1\data.yaml `
  --model models\low_vram\front_detect_reviewed_stage1_640\weights\best.pt `
  --name front_detect_reviewed_stage2_960
```

stage2 默认使用 `960/batch 2`，显存不足时依次尝试 `960/batch 1`、`832/batch 2`、`640/batch 4`。

不要一开始就用 960 训练全部 epoch。先完成 640 基线，再根据逐类 Recall 判断是否值得微调。

## 十、验证和速度测试

使用独立 test 集验证：

```powershell
& $PY -B scripts\train_6gb.py `
  validate `
  dataset\tea_sop_front_v1\releases\detection\front_detect_reviewed_v1\data.yaml `
  models\low_vram\front_detect_reviewed_stage1_640\weights\best.pt `
  --split test `
  --imgsz 960
```

报告位置：

```text
output\validation_best_test.json
```

GTX 1060 推理速度测试：

```powershell
& $PY -B scripts\train_6gb.py `
  benchmark `
  dataset\tea_sop_front_v1\releases\detection\front_detect_reviewed_v1\data.yaml `
  models\low_vram\front_detect_reviewed_stage1_640\weights\best.pt `
  --imgsz 832 `
  --limit 100
```

验收重点：

- 普通器具 Recall 不低于 0.90。
- 茶夹、茶拨、碗盖 Recall 不低于 0.85。
- 独立 test 的 mAP50-95 作为最终参考。
- 新场景指标明显下降时，补充新场景数据，不盲目增加 epoch。

## 十一、导出 ONNX

部署前导出 ONNX：

```powershell
& $PY -B scripts\train_6gb.py `
  export `
  models\low_vram\front_detect_reviewed_stage1_640\weights\best.pt `
  --imgsz 832
```

导出文件通常位于权重文件附近。先用 PyTorch 版本确认精度，再比较 ONNX Runtime 的速度和显存。

## 十二、用新模型重新预标注下一批图片

不要覆盖原始审核结果。可使用独立脚本对未审核批次生成预标注：

```powershell
& $PY -B scripts\relabel_remaining_batches.py `
  --model models\low_vram\front_detect_reviewed_stage1_640\weights\best.pt `
  --first-batch 14 `
  --last-batch 20 `
  --imgsz 832 `
  --conf 0.20 `
  --device 0
```

脚本会：

- 读取各批次 `01_images`。
- 生成同名 YOLO 标签到 `02_auto_labels`。
- 旧的 `02_auto_labels` 自动备份为带时间戳的目录。
- 不修改 `01_images`、修订 ZIP 和历史审核结果。

生成的标签仍然必须人工检查，不能把预标注直接当作训练真值。

## 十三、常见问题

### 显存不足

按顺序处理：

1. 关闭浏览器、播放器和其他 GPU 程序。
2. 使用 `stage1`，不要直接使用 960。
3. 让脚本自动降低 batch。
4. 仍然不足时使用 832 或 640，不要偷偷改成 CPU 长时间训练。

### 训练脚本提示没有 test

说明使用的是实验数据集或旧数据集。正式训练需要独立 `test/images`；只有冒烟训练可以没有独立 test。

### 训练后碗身和碗盖混成一个框

检查：

- 类别 ID 0、1 是否正确。
- 碗身和碗盖是否在同一张图片中分别有两个框。
- 训练是否使用了 18 类 `data.yaml`，而不是旧 9 类模型。
- 碗盖是否有足够的分离、遮挡、手持和开合样本。

### 标签数量和图片数量不一致

空背景图片也必须有同名空 `.txt`。MakeSense 导出时可能省略空标签，发布脚本会对明确的空背景补空文件；普通目标图片缺少修订标签时应停止检查，不能直接猜标签。

### 预标注很差

优先修正高价值错误：

- 错类别。
- 框过大或过小。
- 小目标漏检。
- 手持和遮挡漏检。
- 新机位或新光照下的错误。

每轮只加入经过人工审核的高价值图片，再重新训练。不要把上一轮伪标签全部当作真值。

## 十四、以后每轮训练的固定流程

```text
1. 新建或确认 session 目录
2. 放入原始视频并记录 session
3. 抽取候选帧、删除模糊和重复帧
4. 按固定类别编号在 MakeSense 标注
5. 导出 ZIP，保留图片、标签、manifest
6. 人工复核并补齐空标签
7. 按完整 session 划分 train/val/test
8. 运行 release 或生成 data.yaml
9. smoke 训练 3 epoch
10. stage1 训练 640 基线
11. 查看逐类 Precision、Recall、混淆矩阵
12. 必要时 stage2 做 960 小目标微调
13. 在锁定 test 上只评估一次
14. 用新模型预标注下一批，人工审核后再进入下一轮
```

## 十五、当前项目的可直接使用命令

当前已审核的 1,238 张正式数据集：

```powershell
$DATA = "dataset\tea_sop_front_v1\releases\detection\front_detect_reviewed_v1\data.yaml"

# 冒烟
& $PY -B scripts\train_6gb.py smoke $DATA --model yolov8n.pt --name front_detect_reviewed_smoke

# 正式 640
& $PY -B scripts\train_6gb.py stage1 $DATA --model yolov8n.pt --name front_detect_reviewed_stage1_640
```

训练时只需要修改 `$DATA`、`--name` 和 `--model`；其余参数由项目配置统一管理。
