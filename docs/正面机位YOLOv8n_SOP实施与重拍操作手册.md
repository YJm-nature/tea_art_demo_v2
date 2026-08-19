# 正面机位 YOLOv8n SOP 实施与重拍操作手册

## 当前已经完成的代码

- 默认相机角色改为 `front`，旧 `single/tabletop/side` 仍可回放历史视频。
- 主器具模型 profile 为 `tea18_front`，固定18类编号，首轮启用15类。
- 独立 `YOLOv8n-pose` 接口识别烧水壶、公道杯、盖碗和茶荷的3个关键点。
- 通用倾倒分析器要求手接触、器具离桌、角度变化、出水端对准目标同时成立。
- 温度和重量显示屏使用原图ROI；连续5次稳定后才发布OCR事件。
- 新SOP默认配置为 `config/sop_red_tea_front_v1.yaml`。
- 已实现温杯顺序、投茶姿态、旋转注水、8至12秒计时、出汤、分茶、布局和双手奉茶事件。
- 所有倾倒事件都记录 `liquid_verified: false`，只确认规范动作。
- 评分接口只保存逐项通过、失败、不确定、置信度和证据，不产生正式总分。

这些是可运行的算法和接口，不代表模型已训练完成。新正面检测、关键点和OCR数据达到验收量之前，正式验收仍为关闭状态。

## 第一天先做什么

1. 将相机固定为正面斜俯视，保证脸部鼻点、双手、桌面全部关键器具可见。
2. 用4K 30 FPS录制，锁定焦距、曝光和白平衡；不要手持相机。
3. 把视频放到 `dataset/tea_sop_front_v1/raw_videos/front_s01` 对应目录。
4. `front_s01` 只用于train，不要把同一连续拍摄改名伪装为多个session。
5. 先录器具单件、同框遮挡、手持和空桌负例，再录投茶及四种倾倒器具角度。
6. 每个动作视频只包含一种 positive、error 或 hard_negative，重复8至12次。
7. 动作开始前和结束后各静止2秒。

初始化目录和抽帧：

```powershell
cd E:\tea_culture\tea_art_demo_v2
& "D:\anaconda3\envs\tea-ai-6gb\python.exe" scripts\prepare_front_dataset.py
& "D:\anaconda3\envs\tea-ai-6gb\python.exe" scripts\extract_front_frames.py front_s01
```

第一批只标100张检测框和50张关键点，不要一开始抽几千张。

## 标注规则

检测标注使用固定18类，类别顺序见 `config/ontology_v1.yaml`。即使某张图只有一个类别，类别编号也不能重新从0排列。空桌和非茶具负例要保留一个空txt。

关键点统一为3点，定义见 `config/vessel_keypoints_v1.yaml`：

- 烧水壶：壶嘴尖端、壶身中心、手柄中心。
- 公道杯：出水口、杯身中心、手柄中心。
- 盖碗：左碗沿、右碗沿、碗底中心。
- 茶荷：前端出茶口、中心、后端。

不可见点标为不可见，不允许根据经验猜点。推荐用支持YOLO Pose导出的CVAT完成关键点标注。

OCR数据保留原始显示屏框和裁剪ROI。设备选择必须满足正向、大字体、高对比度；原4K帧单个数字高度至少24像素。边界值必须重点拍：

- 温度：89、90、95、96℃。
- 重量：2.9、3.0、5.0、5.1g。

## 发布和冒烟训练

标注文件分别放入：

- 检测：`annotations/detection_yolo18/labels`
- 关键点：`annotations/pose_yolo/labels`

只发布有同名txt的图片。`--allow-incomplete` 仅用于还没有val/test的 `front_s01` 冒烟训练：

```powershell
& "D:\anaconda3\envs\tea-ai-6gb\python.exe" scripts\publish_front_dataset.py detect front_s01_smoke --allow-incomplete
& "D:\anaconda3\envs\tea-ai-6gb\python.exe" scripts\publish_front_dataset.py pose front_s01_pose_smoke --allow-incomplete

& "D:\anaconda3\envs\tea-ai-6gb\python.exe" scripts\train_6gb.py smoke "dataset\tea_sop_front_v1\releases\detection\front_s01_smoke\data.yaml"
& "D:\anaconda3\envs\tea-ai-6gb\python.exe" scripts\train_6gb.py pose-smoke "dataset\tea_sop_front_v1\releases\pose\front_s01_pose_smoke\data.yaml"
```

确认3 epoch没有类别映射、关键点顺序和显存问题后，再继续完整session。

正式阶段一训练：

```powershell
& "D:\anaconda3\envs\tea-ai-6gb\python.exe" scripts\train_6gb.py stage1 <检测data.yaml> --name front_detect_stage1
& "D:\anaconda3\envs\tea-ai-6gb\python.exe" scripts\train_6gb.py pose-stage1 <关键点data.yaml> --name front_pose_stage1
```

960微调必须分别进行，不能两个模型并发：

```powershell
& "D:\anaconda3\envs\tea-ai-6gb\python.exe" scripts\train_6gb.py stage2 <检测data.yaml> --model <检测best.pt> --name front_detect_stage2
& "D:\anaconda3\envs\tea-ai-6gb\python.exe" scripts\train_6gb.py pose-stage2 <关键点data.yaml> --model <关键点best.pt> --name front_pose_stage2
```

## 实时启动

在主检测和关键点模型都训练完成后：

```powershell
cd E:\tea_culture\tea_art_demo_v2
& "D:\anaconda3\envs\tea-ai-6gb\python.exe" realtime_tracking_demo.py `
  --source camera `
  --camera-id 0 `
  --model "<正面检测best.pt>" `
  --profile tea18_front `
  --pose-model "<正面关键点best.pt>" `
  --camera-role front `
  --observation-mode free_observation `
  --imgsz 832 `
  --pose-imgsz 640 `
  --process-every 2 `
  --hand-every 1 `
  --pose-every 3
```

先用 `free_observation` 分别验收每个观测点。只有所有动作在锁定test上达标后，才改为 `--observation-mode strict` 运行完整SOP。

## 仍需数据才能启用的能力

- 新正面YOLOv8n器具模型尚未训练。
- YOLOv8n-pose关键点模型尚未训练。
- 水壶和电子秤显示屏OCR边界样本尚未建立。
- 温杯、投茶、注水、出汤、分茶和奉茶仍需计划中的正负独立session。
- 旧托举、闻香、杯位原视频审核为合格正面视角后可补入train，但不进入val/test。

当前不采集液柱、液珠、溅水、茶渣、液面和实际倒入量标注，避免无用工作。

