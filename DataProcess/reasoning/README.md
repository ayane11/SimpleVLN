# R2R / RxR 全量推理标注

使用本地 Qwen3.5-27B、vLLM TP=2，默认 GPU **2、3**。当前目录的代码只负责离线标注，不启动 SFT。脚本已经准备好，尚未执行全量轨迹恢复或推理。

## 启动与续跑（Docker 内）

```bash
cd /workspace/VLN/SimpleVLN
bash DataProcess/reasoning/run_full.sh all
```

中断后执行**同一条命令**即可续跑。若希望后台运行：

```bash
mkdir -p outputs/reasoning/full_qwen35_27b/logs
nohup bash DataProcess/reasoning/run_full.sh all \
  > outputs/reasoning/full_qwen35_27b/logs/launcher.log 2>&1 &
```

宿主机进入已有容器：`docker exec -it dyj_vln bash`。脚本以自身位置定位源码，不依赖当前目录。可用 `--gpus 2,3 --workers 8 --port 22992 --output /workspace/VLN/SimpleVLN/outputs/reasoning/full_qwen35_27b` 覆盖默认设置。不要同时启动两个全量任务；输出目录有进程锁。

只做某一阶段：

```bash
bash DataProcess/reasoning/run_full.sh prepare
bash DataProcess/reasoning/run_full.sh generate
bash DataProcess/reasoning/run_full.sh export
```

`prepare` 会使用 GPU 进行 Habitat 重放，`generate` 会启动模型；`export` 只合并已有结果。不要把 `prepare` 当作无 GPU 的预览命令。

## 执行顺序

1. 遍历本地 R2R **10,819** 条、RxR **19,990** 条源轨迹。按与 SFT 相同的 `prep_actions_and_images` 规则，移除初始动作占位符并补终止动作，统一为 **第 t 帧是执行动作 t 之前的观测**。
2. 用原始 episode 的场景、起点恢复轨迹。固定相机高度 1.25 m、HFOV 79°、前进 0.25 m、转向 15°；从首帧拟合初始朝向，再核对整段轨迹 RGB。先 160×120，未通过者再以 640×480 重放并缩小比较。门槛：平均 MAE < 10、最大 MAE < 20。
3. 只有通过数值对齐检查的轨迹进入房间事件选取；没有节点的轨迹保留为动作样本，不强行增加推理。不能对齐或缺数据的轨迹写入 `excluded.json`，不会让教师自行补选节点。
4. 重放结束、释放模拟器后，启动 GPU 2、3 上的一个 TP=2 vLLM 服务。R2R、RxR 依次处理，各自最多 8 个并发请求。每条轨迹的推理节点仍按时间顺序生成，以保证历史记忆依赖正确。
5. 先按原文拆分指令，再生成三段文本：场景描述、进度判断、下一步计划。输入当前/历史图像、上一节点文本与结构化进度、已经执行的专家动作及紧邻动作提示。严格约束当前专家动作和 STOP；计划文本规范为下一步原子动作。未通过校验时最多修复重试两次，仍失败则跳过该节点并保留此前有效记忆。
6. 各数据集导出 `cot_new.json` 和 `cot_index.json`，并汇总为跨数据集的 `exports/cot_index.json`。

节点规则、文本结构借鉴 AwareVLN；本地房间类别确认、图像对齐门槛、专家动作约束是本实现的具体选择，**不声称完全重现其未公开的标注生成器**。数值对齐通过也不代表文本语义经过人工审核。

## 统一输出目录

`/workspace/VLN/SimpleVLN/outputs/reasoning/full_qwen35_27b/`

| 路径 | 内容 |
| --- | --- |
| `preparation/{r2r,rxr}/replays/` | 逐轨迹重放、位置、房间类别、RGB 误差；原子保存，用于续跑 |
| `preparation/{r2r,rxr}/events.json` | 提供给教师的推理节点 |
| `preparation/{r2r,rxr}/annotations.ready.json` | 对齐通过的源轨迹 |
| `preparation/{r2r,rxr}/status.json`, `excluded.json` | 覆盖率、节点数、排除原因 |
| `generation/{r2r,rxr}/episodes/` | 指令、专家动作、图像路径、节点文本、历史记忆、结构化进度、失败和排除节点 |
| `generation/{r2r,rxr}/requests/` | 精确输入、教师输出原文、解析结果、生成配置和耗时 |
| `generation/{r2r,rxr}/errors/` | 新任务运行失败信息，供续跑排错；不是旧审阅记录 |
| `generation/{r2r,rxr}/cot_new.json` | Aware 风格 segments，每个数据集独立保存 |
| `exports/cot_index.json` | 全量汇总；键是 `[dataset, episode_id, instruction_index]`，避免两数据集 ID 冲突 |
| `logs/` | 模拟器、模型服务和生成日志 |
| `seed_v3/{r2r,rxr}/` | 已迁移的最后一版各 50 条原始结果及 247 个成功请求；仅作为复用种子，不是新全量输出 |
| `seed_v3/provenance/` | 种子请求引用的上游原始响应，保证追溯链有效 |
| `seed_alignment/` | 原试跑轨迹和节点数据；全量流程仍使用新规则重放，不冒充新输入指纹缓存 |

输出 JSON 保存图像**路径**，不复制图片或模型权重。现存种子结果保留原貌；全量流程会重新校验精确匹配的响应，并应用 `excluded_nodes.json` 中已确认有问题的 7 个节点。排除节点不会更新历史记忆，后续请求若因此变化就重新生成。

## 续跑与异常行为

- 模拟器按 episode 原子落盘；输入图片/场景文件发生变化时拒绝使用旧缓存。数值不通过的轨迹默认不会重复计算，可单独给 `prepare_full.py` 加 `--retry-rejected` 重试。
- 教师按**请求**和**轨迹**分别落盘。中途退出后可复用已完成请求；存在失败节点的轨迹会再次处理，成功节点从缓存读取。恢复的节点会更新后续记忆，后续请求按新内容生成。
- 失败节点会导致生成阶段非零退出，但已成功结果照常保存、导出；另一数据集仍会处理。检查 `run_summary.json` 的 `failures` 和 `failed_nodes`。
- 数据、配置、节点排除规则或代码变化时，输出 manifest 会阻止混入不同版本；请指定新的 `--output`，不要删除 manifest 强行续跑。
- 重跑准备阶段不会改写内容未变的 ready 文件，保证生成 manifest 的文件指纹稳定。
- 脚本检查端口占用，只停止自己启动的服务和子进程。中断或失败不会清理已落盘结果。
- 原有检查只约束格式、证据帧、专家动作及部分进度一致性，不能保证每条自然语言都正确。未来人工抽检可直接从新 `episodes/` 和 `requests/` 获取图像路径与原文。

## 无模型验证

```bash
python3 -m unittest discover -s DataProcess/reasoning -p 'test_*.py' -q
bash -n DataProcess/reasoning/run_full.sh
```

测试用假模拟器/假教师验证动作索引、对齐失败排除、重放续跑、失败节点重试、请求缓存和跨数据集导出，不加载模型、不执行 Habitat。真实全量运行的速度、对齐覆盖率与输出质量需要实际启动后统计。
