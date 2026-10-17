---
name: anime-style-pipeline
description: 把一段视频逐帧做风格化/人物替换(调用 Seedream 图像模型)并合成成品的 6 工序流水线。决策 Agent 开始任务前必须读取;包含阶段顺序、参数决策规则、质检解读与失败处置。
---

# 视频逐帧风格化流水线(Agent 操作手册)

你通过工具驱动流水线,**每个工具对应一个阶段**。工具会真正提交到执行引擎并等待结果;引擎负责重试、租约、缓存和崩溃恢复,你负责**决策**:参数选什么、结果能不能接受、失败了怎么办。

## 1. 阶段与工具(必须按顺序)

| 顺序 | 工具 | 做什么 | 是否花钱 | 能否并行 |
|---|---|---|---|---|
| 1 | `ingest_video` | 校验素材,转 1080p/24fps/H.264,分离音轨 | 否 | — |
| 2 | `extract_keyframes(chunks, stride, max_gap)` | 选关键帧并分块(镜头切换点两侧强制选) | 否 | — |
| 3 | `stylize_chunk(chunk)` | **调用 Seedream** 对本块每个关键帧风格化 | **是** | **同一轮并行调用所有块** |
| 4a | `poststats` | 汇总各块色彩统计(需要 3 全部完成) | 否 | — |
| 4b | `postprocess_chunk(chunk)` | 色彩对齐、去噪、按源视频运动补帧、编码分段 | 否 | 同一轮并行所有块 |
| 5a | `quality_check` | 质检 | 否 | — |
| 5b | `mux_audio` | 拼接并合入源音轨 | 否 | — |
| 6 | `package_and_publish` | 打包成品+元数据并发布 | 否 | — |

辅助工具:`get_run_status`(各阶段状态与累计用量)、`inspect_frames(chunk)`(逐帧结果)、`retry_stage(stage_id)`、`escalate_to_human(reason)`。

阶段 id:`ingest`、`keyframes`、`stylize:<块>`、`poststats`、`post:<块>`、`qc`、`mux`、`package`。

## 2. 参数决策规则(`extract_keyframes`)

先看 `ingest_video` 返回的 `standard.duration_s` / `frames` 与预算:

- **stride**(关键帧间隔,帧):默认 6。素材 ≤ 10 s 用 6;10–30 s 用 8–12;更长用 12–24。stride 越小画面越连贯但越贵;stride 大于 24 补间会明显变差。
- **chunks**(并行块数):关键帧数 ≈ `frames / stride`。每块 6–10 个关键帧为宜;一般 `chunks = clamp(round(关键帧数 / 7), 1, 8)`。块数不是越多越好:每块是一个独立任务。
- **max_gap**:默认 24,不小于 stride。
- **预算**:`extract_keyframes` 的返回里有 `est_cost_cny` 与 `budget_cny`。**估算超过预算的 80% 就必须调大 stride 后重新调用 `extract_keyframes`**;不允许在预算不够的情况下继续风格化。
- 你不能修改预算、提示词、模型、供应商;这些是运行配置,固定。

## 3. 结果解读

- `stylize_chunk` 返回的 `seedream` 里有 `calls / billed / output_tokens / cost_cny_est`;`cached_frames` 大于 0 表示命中缓存,没有重复花钱;`maybe_duplicated` 不为空说明有帧可能被重复计费,汇报时必须写明。
- `degraded` 不为空:有帧被内容审核拒绝而退回原图。比例小可以接受(汇报时写明);比例大会让阶段失败。
- `quality_check`:
  - `ok=false`(硬检查失败)→ **不要发布**。
  - `warnings` 里的"结构保真度"低于关键帧的 85% → 补帧可能有重影,汇报时指出;不要为此重做风格化(花钱且不一定改善)。
  - 有降级帧或"可能重复计费"→ 汇报时如实写明。
- 不要在汇报里编造数字;所有数字来自工具返回。

## 4. 失败处置

失败的工具结果里有 `diagnosis`:`playbook`(场景)、`retry_allowed`、`human_steps`。

| 场景 | 动作 |
|---|---|
| `retry_allowed=true`(瞬时错误、部分帧失败、内存不足) | `retry_stage(stage_id)`,最多按 `max_retries` 次;再失败 → `escalate_to_human` |
| 鉴权失败 / 欠费 / 模型未开通 / 模型不存在 / 请求参数被拒 | **不要重试**,立刻 `escalate_to_human` |
| 触及预算保护线 | 不要重试;`escalate_to_human`,说明需要人提高预算 |
| 帧被审核拒绝且阶段失败 | 可以用 `inspect_frames` 查看哪些帧,再用 `stylize_chunk(chunk, frame_prompts={帧号: 更温和的提示词})` 只为这些帧换提示词(其余帧命中缓存不重复付费);仍失败则 `escalate_to_human` |
| 质检硬失败 | 不要发布;`escalate_to_human` |
| 坏素材 | `escalate_to_human` |
| 未知错误 | 不重试,`escalate_to_human` |

`retry_stage` 被拒绝时,返回信息里已经写明原因,照做,不要换个说法再试。

## 5. 禁止事项

- 不要跳过阶段,不要在上游未成功时调用下游工具。
- 不要对同一个失败阶段反复重试以"碰运气"。
- 质检未通过不得 `package_and_publish`。
- 不要承诺工具没有返回的结果(例如"成品已发布"必须以 `package_and_publish` 返回 SUCCEEDED 为准)。
