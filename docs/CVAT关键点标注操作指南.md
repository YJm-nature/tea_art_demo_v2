# CVAT 关键点标注操作指南

本项目使用 CVAT 标注器具关键点，使用 YOLOv8n-pose 训练。CVAT 不负责训练，也不把“温杯”“投茶”“注水”等动作名称加入器具类别。

## 一、三种标注分别做什么

| 类型 | 工具 | 输出 | 当前用途 |
|---|---|---|---|
| 器具框 | MakeSense 或 CVAT | YOLO 检测标签 5 列 | 训练 YOLOv8n |
| 器具关键点 | CVAT Skeleton | YOLO Pose 标签 14 列 | 训练 YOLOv8n-pose，计算倾斜和出水端 |
| 动作时序 | CSV/JSONL | action_id、variant、start_s、end_s | 规则和 SOP 状态机 |

动作图片已经按目录整理在：

```text
dataset/tea_sop_front_v1/derived/action_pool/new_front_full_202608/
```

但这些图片当前只是动作证据帧，不能直接当作 Pose 标签。动作时间段填写在：

```text
dataset/tea_sop_front_v1/manifests/action_segments_review.csv
```

## 二、启动 CVAT

推荐使用 CVAT 本地 Docker 版，数据不会离开电脑。需要先安装并启动 Docker Desktop。

启动后在浏览器打开：

```text
http://localhost:8080
```

如果使用 CVAT 在线版，操作界面相同，但上传前应确认视频和图片可以放到云端。

## 三、准备关键点图片

当前 Pose 目录为空。不要把 2,000 张检测图全部标关键点，第一批只选 50-100 张，优先选择倾斜变化、拿起、遮挡和目标器具清晰的帧。

目标图片放在：

```text
dataset/tea_sop_front_v1/derived/pose/images/
```

图片文件名必须保留 session 前缀，例如：

```text
front_s01__pose__water_injection__000001.jpg
```

当前 `scripts/publish_front_dataset.py` 识别 `front_s01` 到 `front_s08` 作为 session。来自旧 `new_front_full_202608` 的动作图片如果没有 `front_s01` 等前缀，先不要直接发布到正式 Pose 集；可以先存到临时审核目录，后续统一建立 session 映射。

## 四、创建 CVAT Task

1. 登录 CVAT，点击 `Tasks`，再点击 `Create new task`。
2. Task 名称示例：`front_s01_vessel_pose_batch01`。
3. 上传 `dataset/tea_sop_front_v1/derived/pose/images/` 中的图片。
4. 创建 4 个 `Skeleton` 类型的标签，顺序必须固定：

```text
0 kettle
1 pitcher
2 gaiwan_body
3 tea_lotus
```

不要创建 `gaiwan_lid`、`warm_clean`、`water_injection` 等 Pose 类别。

5. 每个 Skeleton 恰好创建 3 个节点，节点顺序必须固定：

```text
kettle:      outlet_tip, body_center, handle_center
pitcher:     outlet_tip, body_center, handle_center
gaiwan_body: left_rim, right_rim, base_center
tea_lotus:   front_outlet, center, rear
```

建议使用 ASCII 名称，避免 Windows 中文编码导致类别顺序显示异常。类别和节点顺序以 `config/vessel_keypoints_v1.yaml` 为准。

## 五、每张图片怎么标

只标注这 4 类器具：烧水壶、公道杯、盖碗碗身、茶荷。其他茶具不创建 Pose 标注。

标注规则：

- 一个可见目标对应一个 Skeleton。
- 节点顺序不能交换。
- `outlet_tip` 是实际出水口/出茶口的尖端，不是器具框边缘。
- `body_center` 是器具主体几何中心，不是画面中心。
- 盖碗标左碗沿、右碗沿和碗底中心。
- 茶荷标前端出茶口、中心和后端。
- 被遮挡但能确定位置的点标为 `occluded`。
- 完全看不到且无法确定位置的点标为 `outside/not visible`，不要凭经验猜坐标。
- 物体严重遮挡到 3 个点都无法可靠判断时，整条 Skeleton 删除，保留该图片供检测模型使用。

可见性对应关系：

```text
visible       -> 2
occluded      -> 1
not visible   -> 0
```

## 六、导出

完成一批标注后，在 CVAT Task 页面点击 `Actions -> Export task dataset`，格式选择：

```text
Ultralytics YOLO Pose 1.0
```

下载 ZIP 并解压。只需要使用其中的图片和 Pose 标签，不要覆盖器具检测标签。

图片放到：

```text
dataset/tea_sop_front_v1/derived/pose/images/
```

对应 `.txt` 标签放到：

```text
dataset/tea_sop_front_v1/annotations/pose_yolo/labels/
```

图片和标签必须同名：

```text
front_s01__pose__water_injection__000001.jpg
front_s01__pose__water_injection__000001.txt
```

Pose 标签每行 14 列：

```text
class_id cx cy width height
kp1_x kp1_y kp1_visible
kp2_x kp2_y kp2_visible
kp3_x kp3_y kp3_visible
```

所有坐标归一化到 `0-1`，可见性只能是 `0/1/2`。

## 七、发布和冒烟训练

先确认图片和标签数量对应，再发布临时 Pose 集：

```powershell
cd E:\tea_culture\tea_art_demo_v2

& "D:\anaconda3\envs\tea-ai-6gb\python.exe" `
  scripts\publish_front_dataset.py `
  pose `
  front_pose_v1 `
  --root "dataset\tea_sop_front_v1" `
  --allow-incomplete
```

第一批只做 3 epoch 冒烟训练：

```powershell
& "D:\anaconda3\envs\tea-ai-6gb\python.exe" `
  -B scripts\train_6gb.py `
  pose-smoke `
  "dataset\tea_sop_front_v1\releases\pose\front_pose_v1\data.yaml" `
  --model "yolov8n-pose.pt" `
  --name "front_pose_smoke"
```

确认类别顺序、节点顺序和显存正常后，再继续批量标注和正式训练。

## 八、动作标注不要混入 Pose

动作标签使用时间段，不使用 Skeleton 类别：

```text
action_id,variant,start_s,end_s
water_injection,positive,12.4,18.1
gaiwan_to_pitcher,positive,24.0,29.2
```

动作正例、错误例和困难负例都应来自完整视频 session。同一连续视频只能进入 train、val、test 其中一个集合，不能按图片随机拆分。

## 九、完成检查

提交标注前检查：

- 4 个 Skeleton 类别顺序没有变化。
- 每个 Skeleton 恰好 3 个关键点。
- 节点顺序符合配置文件。
- 图片和 `.txt` 文件一一对应。
- 不把动作名写入 YOLO Pose 类别。
- 不修改 `dataset/tea_sop_front_v1/releases/detection/front_detect_merged_dedup_v2`。

