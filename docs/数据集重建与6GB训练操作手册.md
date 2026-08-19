# 数据集重建与 GTX 1060 6GB 训练操作手册

## 1. 当前状态

旧 `final_tea9_dataset_20260723` 保持不变。新的审核工作区已经创建在：

```text
dataset/tea_dataset_v1_reviewed
```

截至 2026-07-28，50 张人工修框已用于生成自动标注候选。当前工作区有 848 张图片，其中 844 张为 `needs_fix`，4 张为 `rejected`，另有 202 张近重复候选。近重复仅做标记，没有自动删除。

按固定随机规则抽出403张二审样本：自动标注和茶夹/茶拨/茶叶罐等高风险样本按50%抽检，其余按20%抽检。主审和二审不能是同一人员。

新 ontology 为18类，见 `config/ontology_v1.yaml`。当前有实物并进入本轮审核的类别 ID 为 `0,1,2,3,4,5,6,7,10,14`；类别 `8,9,11,12,13,15,16,17` 暂不训练，但编号永久保留，以后补齐器具时直接增量扩充，禁止重新编号。

工作区中的旧标签保存在只读目录 `pool/labels/legacy_tea9`。当前自动候选已写入 `pool/labels/detect`，应用前的旧框备份在：

```text
pool/labels/detect_before_auto_v2_current10_legacy_split
```

自动候选由两部分组成：轻量 YOLO 模型补充/微调当前器具框；盖碗碗身和碗盖在跨场景漏检时，使用50张人工样本学习到的“旧整框到双框”关系补初始框。它们只用于降低修框工作量，全部必须人工复核，不能直接作为正式训练真值。

## 2. 创建训练环境

当前全局 Python 3.13 出现过 OpenMP 冲突，正式训练使用独立 Python 3.11 环境：

```powershell
cd E:\tea_culture\tea_art_demo_v2
conda env create -f environment-6gb.yml
conda activate tea-ai-6gb
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
```

不要设置 `KMP_DUPLICATE_LIB_OK` 掩盖环境冲突。

## 3. 逐张审核

查看进度：

```powershell
python scripts\review_dataset.py dataset\tea_dataset_v1_reviewed summary
```

打开审核器：

```powershell
python scripts\review_dataset.py dataset\tea_dataset_v1_reviewed gui `
  --reviewer 姓名 --status needs_fix
```

Windows 下修改 YOLO 标注框使用项目内的 LabelImg 兼容启动器：

```powershell
python scripts\labelimg_compat.py `
  dataset\tea_dataset_v1_reviewed\pool\images `
  dataset\tea_dataset_v1_reviewed\classes.txt `
  dataset\tea_dataset_v1_reviewed\pool\labels\detect
```

该启动器解决 LabelImg 1.8.6 在新版 PyQt5 下绘制浮点坐标崩溃的问题。`pool/labels/detect/classes.txt` 是 LabelImg 专用的 GBK 兼容副本；项目根目录的 `classes.txt` 仍保持 UTF-8。

如果 LabelImg 在当前 Python/PyQt 环境下仍不稳定，优先使用 `E:\tea_culture\make-sense` 的本地 MakeSense。执行 `npm run dev` 后访问 `http://localhost:3000`。

MakeSense 默认每批处理 100 张：

```powershell
python scripts\prepare_makesense_batch.py `
  dataset\tea_dataset_v1_reviewed `
  output\makesense_review_batches\office_batch_001 `
  --source office --status needs_fix --limit 100
```

1. 将批次目录 `01_images` 中的全部图片载入 MakeSense。
2. 导入 YOLO 时进入 `02_yolo_import`，按 `Ctrl+A` 全选100个同名标注 TXT 和 `labels.txt` 后再点打开；只点一个 TXT 无法导入整批。
3. 修框后导出 YOLO ZIP。先保留 ZIP，再把其中100个同名 TXT 更新回 `pool/labels/detect`。
4. 使用审核器逐图确认，正确按 `A`，仍需修改按 `F`，无效/重复图按 `R`。

一批完成并更新审核状态后，把输出目录名改为 `office_batch_002` 再执行相同命令；脚本默认只选择仍为 `needs_fix` 的图片，因此已完成样本不会重复打包。

MakeSense 必须使用 UTF-8 的 `labels.txt`，不要使用 LabelImg 专用的 GBK `classes.txt`。

按键：`A` 接受、`F` 待修、`R` 废弃、`N/P` 前后切换、`Q` 退出。每次操作立即写回 manifest，可以中断续做。

审核器用于逐图查看和记录状态，不负责画框。需要修框时：

1. 将 `pool/images` 和当前 `pool/labels/detect` 导入 CVAT 或本仓库 MakeSense。
2. 使用工作区 `classes.txt` 的固定18类顺序。
3. 修正后将 YOLO 标签导回 `pool/labels/detect`，文件名必须和图片一致。
4. 再回到审核器按 `A`。

旧图只要含旧盖碗，就必须同时存在 class 0 碗身和 class 1 碗盖，否则工具会自动标成 `needs_fix`，不能进入训练集。自动生成的双框主要用于起点：碗身框不要把托碟算入，碗盖只框可见盖体；手遮挡时按可见区域修正。

动作帧中若碗身或碗盖完全离开画面/被完全遮挡，不要画猜测框。删除错误框后显式记录该类已经人工确认不可见：

```powershell
python scripts\review_dataset.py dataset\tea_dataset_v1_reviewed mark-absent SAMPLE_ID 1 `
  --reviewer 姓名 --note "碗盖已移出画面"
```

也可以单张更新：

```powershell
python scripts\review_dataset.py dataset\tea_dataset_v1_reviewed set SAMPLE_ID needs_fix --reviewer 姓名 --note "漏标碗盖"
```

所有图片必须最终为 `accepted` 或 `rejected`。重复候选仍需人工确认，再决定保留质量更好的图片还是拒绝重复项。

被抽中的图片完成二审：

```powershell
python scripts\review_dataset.py dataset\tea_dataset_v1_reviewed second-set SAMPLE_ID accepted --reviewer 二审姓名
```

二审选择 `rejected` 时，主状态会自动退回 `needs_fix`。未完成必要二审的数据不能发布。

## 4. 补采与候选帧

每次录制先填写 `config/collection_session_template.csv`。同一套顶部/侧面视频使用相同 `session_id`。

从新视频抽取清晰、去重且有状态变化的候选帧：

```powershell
python scripts\extract_active_learning.py VIDEO_TOP.mp4 VIDEO_SIDE.mp4 `
  --output dataset\candidates_v1 `
  --interval 1.0 `
  --max-per-video 500
```

有阶段一模型后加入不确定性筛选：

```powershell
python scripts\extract_active_learning.py VIDEO.mp4 `
  --output dataset\candidates_v2 `
  --model models\low_vram\stage1_xxx\weights\best.pt
```

候选 manifest 保存原视频、帧号、时间戳、清晰度、画面变化和模型不确定性。候选图片标注完成后，再作为新 session 合入下一数据版本；不要直接复制到旧 train 目录。

## 5. 发布 train/val/test

正式发布要求所有图片已完成审核。发布物仍保留18类固定编号，但当前门禁只要求10个active类有实例，并拒绝8个deferred类意外混入。当前只有三个旧来源，只能发布场景留一的原型集：

```powershell
python scripts\publish_reviewed_dataset.py `
  dataset\tea_dataset_v1_reviewed `
  dataset\tea_dataset_v1_release `
  --assign office=train original=val focus=test `
  --allow-prototype
```

补充至少5个独立 session/类别后，正式发布不要使用 `--allow-prototype`。未显式分配的 session 会按70/15/15目标整体分配，不会拆散视频。

发布后执行门禁：

```powershell
python scripts\validate_release_dataset.py dataset\tea_dataset_v1_release
python scripts\audit_dataset.py dataset\tea_dataset_v1_release
```

门禁检查文件/标签、18类实例、train/val/test、session泄漏、相同SHA图片跨集合和每类独立session数。

## 6. 6GB训练顺序

先打印参数，不训练：

```powershell
python scripts\train_6gb.py --dry-run smoke dataset\tea_dataset_v1_release\data.yaml
python scripts\train_6gb.py --dry-run stage1 dataset\tea_dataset_v1_release\data.yaml
```

正式执行：

```powershell
# 100张、3 epochs显存与标签冒烟
python scripts\train_6gb.py smoke dataset\tea_dataset_v1_release\data.yaml

# 640 / batch 4，OOM时自动降为batch 2、1
python scripts\train_6gb.py stage1 dataset\tea_dataset_v1_release\data.yaml

# 自动寻找最新stage1 best.pt，按960/2 -> 960/1 -> 832/2 -> 640/4降级
python scripts\train_6gb.py stage2 dataset\tea_dataset_v1_release\data.yaml

# 对照模型
python scripts\train_6gb.py stage1 dataset\tea_dataset_v1_release\data.yaml `
  --model yolov8n.pt --name yolov8n_stage1
```

液体分割使用独立4类数据集：

```powershell
python scripts\train_6gb.py segment dataset\tea_seg_v1\data.yaml `
  --model yolov8n-seg.pt
```

检测与分割不得同时训练。

## 7. 独立测试与导出

```powershell
python scripts\train_6gb.py validate dataset\tea_dataset_v1_release\data.yaml MODEL.pt --split test --imgsz 960
python scripts\train_6gb.py benchmark dataset\tea_dataset_v1_release\data.yaml MODEL.pt --imgsz 832 --limit 100
python scripts\train_6gb.py export MODEL.pt --imgsz 832
```

验证结果写入 `output/validation_*.json`。基准测试输出 FPS、单图延迟和峰值 CUDA 显存。

## 8. 必须遵守的门槛

- 不允许使用旧9类随机帧拆分数据进行正式stage1训练。
- 不允许 pending 或 needs_fix 图片进入发布集。
- 不允许同一 session 跨 train/val/test。
- 不允许用当前存在泄漏的 mAP50 0.98 作为验收成绩。
- 训练前先跑 smoke；出现 OOM 由脚本按固定配置降级，不手工随机改参数。
- 只有独立 test 达到逐类 Recall 门槛后，模型才进入实时 demo。
