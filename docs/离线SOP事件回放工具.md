# 离线 SOP 事件回放工具

该工具不打开摄像头，也不运行 YOLO。它读取已经生成或人工编写的观测事件，按
`config/sop_red_tea_v1.yaml` 驱动同一套 SOP 状态机，用于检查步骤顺序、超时、
低置信度复核、跳步和重试逻辑。

## 当前默认范围

默认只加载 `available`、`experimental` 和 `partial` 节点：

1. 从茶叶罐取茶至茶荷；
2. 双手托举茶荷赏茶；
3. 打开盖碗闻香；
4. 冲泡等待与出汤（当前不含注水）。

温杯洁具和茶汤品茗杯茶盘布局仍为 `deferred`，默认不会影响当前回放完成状态。
使用 `--include-deferred` 可以模拟未来完整节点，但不代表这些能力已经验收。
步骤一目前仍由独立评分结果提供，尚未发布时序事件，因此配置为
`runtime_enabled: false`。完整六步结构测试还需要同时使用 `--include-disabled`。

## 启动命令

在 `E:\tea_culture\tea_art_demo_v2` 下运行：

```powershell
& "D:\anaconda3\envs\tea-ai-6gb\python.exe" scripts\replay_sop_events.py `
  --events examples\sop_events_current_example.jsonl `
  --mode strict `
  --output output\reports\sop_replay.json `
  --require-complete
```

退出码为 `0` 表示工具正常执行且在启用 `--require-complete` 时全部当前节点完成；
`2` 表示配置或事件文件错误；`3` 表示回放有效但 SOP 没有完成。

完整六步结构模拟命令：

```powershell
& "D:\anaconda3\envs\tea-ai-6gb\python.exe" scripts\replay_sop_events.py `
  --events examples\sop_events_full_simulation.jsonl `
  --mode strict `
  --include-deferred `
  --include-disabled `
  --output output\reports\sop_replay_full_simulation.json `
  --require-complete
```

该结果只证明配置、事件契约和状态转换完整，不代表延期观测器已经具备真实视频能力。

## 事件格式

JSONL 每行一个对象，也可使用 JSON 数组或 `{ "events": [...] }`：

```json
{"observation_id":"action_hold_lotus","phase":"completed","end_time":10.0,"confidence":0.9}
```

支持 `phase` 或 `event_type`，支持 `end_time` 或 `timestamp`。输入默认严格保留原始
顺序；仅在确实需要按时间整理历史文件时使用 `--sort-events`。

控制记录格式：

```json
{"operation":"tick","timestamp":30}
{"operation":"skip","step_id":"hold_lotus","timestamp":31,"reason":"人工跳过","force":true}
{"operation":"review","step_id":"smell","timestamp":32,"approved":true}
{"operation":"retry","step_id":"smell","timestamp":33}
```

报告中的 `records` 保留每条输入导致的全部状态转换；`final_machine` 是完整可序列化
状态机快照。低于节点 `min_confidence` 的事件进入 `needs_review`，不会自动判错。

完整需求、现有实现、缺失数据和正式评分可用性见
`config/observation_capability_matrix.yaml`。
