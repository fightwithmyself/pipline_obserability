# 可观测、可恢复的 AI 任务执行管线(MVP)

以"动画素材角色替换"为例:一个由多个耗时步骤(含 GPU 步骤)组成的 DAG 任务,
在**单步骤失败 / worker 进程被强杀 / 步骤卡死 / 整个服务重启**时都能从断点继续,
上游随时能查到任务与每个步骤的状态、重试历史和事件时间线;外面再套一层 **LLM Agent**(规划 + 失败诊断),
LLM 的输出一律当不可信输入处理。

> 计算全是 MOCK(sleep + 确定性字节),但**进程、文件、哈希、`kill -9`、检查点都是真的**。
> 设计推导见 [`docs/DESIGN.md`](docs/DESIGN.md)(引擎)与 [`docs/AGENT_PLAN.md`](docs/AGENT_PLAN.md)(Agent 层)。

## 目录

1. [快速开始](#1-快速开始)
2. [实现方式与时间规划](#2-实现方式与时间规划)
3. [长任务执行、状态管理与失败恢复的设计考量](#3-长任务执行状态管理与失败恢复的设计考量)
4. [任务步骤与依赖关系的定义](#4-任务步骤与依赖关系的定义)
5. [哪些状态被持久化,为什么](#5-哪些状态被持久化为什么)
6. [AI coding 工具与 token](#6-ai-coding-工具与-token)
7. [个人时间花费](#7-个人时间花费)
8. [评测结果](#8-评测结果)
9. [失败案例分析:做错的、做不了的](#9-失败案例分析做错的做不了的)
10. [加分项对照](#10-加分项对照)

---

## 1. 快速开始

```powershell
py -3.11 -m venv .venv                       # 虚拟环境,依赖不污染本机
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

.\.venv\Scripts\python.exe -m pytest tests -q          # 58 个测试,约 2 分钟(含真实 kill -9 与混沌测试)
```

起一套完整环境(三个终端):

```powershell
python -m pipeline.cli --data data api --port 8000                      # 接口 + Dashboard: http://127.0.0.1:8000
python -m pipeline.cli --data data worker --id w1 --lease-ttl 3         # 可以起多个;--resources cpu 或 gpu 限定能力
python -m pipeline.cli --data data submit --inject crash                # 提交任务并注入"进程崩溃";还有 transient / hang
python -m pipeline.cli --data data status                               # 命令行查看;events 查看事件历史
```

演示"kill -9":提交任务后,在 Dashboard 看到某个 GPU 步骤进度过半,用任务管理器强杀 worker 进程,
再起一个新 worker —— 步骤会在租约过期后被回收,并**从检查点继续**(详情里能看到 `resumed_from`)。

**Agent 功能**(默认走 MockLLM;接真实模型只需环境变量,任何 OpenAI 兼容服务都行):

```powershell
$env:LLM_PROVIDER="openai"; $env:OPENAI_BASE_URL="https://api.deepseek.com"
$env:OPENAI_API_KEY="sk-..."; $env:LLM_MODEL="deepseek-chat"
$env:LLM_PRICE_IN_PER_1K="0.0002"; $env:LLM_PRICE_OUT_PER_1K="0.0006"       # 可选,用于成本统计
python -m pipeline.agent.eval --repeat 3 --out eval/results/planner_real.json   # Planner 评测
python -m pipeline.mcp_server --data data                                        # MCP Server(默认只读)
```

MCP 接入 Cursor(`.cursor/mcp.json`):

```json
{"mcpServers": {"task-pipeline": {"command": "F:/.../.venv/Scripts/python.exe",
  "args": ["-m", "pipeline.mcp_server", "--data", "F:/.../data"]}}}
```

## 2. 实现方式与时间规划

**技术栈**:Python 3.11;执行引擎(状态机 / 租约 / 缓存 / worker)只用标准库 + SQLite(WAL);
接口层 FastAPI;Agent 层 `openai` SDK(OpenAI 兼容协议)+ LangGraph + MCP Python SDK。
**没有**使用 Celery / Temporal / Redis —— 题目要求"同等结果下更简单、更可控、更容易解释正确性的方案得分更高",
所以选了"一个 SQLite 文件 + 一个 `transition()` 入口 + 租约/fencing"。

**模块**:

| 模块 | 职责 |
|---|---|
| `pipeline/engine.py` | 状态机、认领、租约、fencing、重试/退避、缓存复用、取消/恢复/审批、查询(**唯一写入口**) |
| `pipeline/worker.py` + `step_runner.py` | worker 拉取任务,**每个步骤起子进程**,监督续租/卡死/超时/取消,提交结果 |
| `pipeline/artifacts.py` | 内容寻址产物存储(sha256 命名、原子写、读时校验) |
| `pipeline/steps/animation.py` | MOCK 步骤(确定性输出、步骤内检查点、故障注入) |
| `pipeline/api.py` + `web/index.html` | 查询/控制接口与 Dashboard(DAG 着色、尝试历史、事件时间线、LLM 追踪、诊断确认) |
| `pipeline/agent/` | `planner.py` 规划+护栏、`diagnosis.py` LangGraph 诊断、`tools.py` 受控工具、`llm.py` 客户端+追踪 |
| `pipeline/mcp_server.py` | 把只读(可选写)工具暴露给 MCP 客户端 |

**时间规划**(设计稿阶段估算的分配;**实际耗时请在第 7 节按你自己的记录填写**):

| 阶段 | 内容 | 预估占比 |
|---|---|---|
| 0 | 需求拆解、方案设计 | ~10% |
| 1–2 | 状态机、租约/fencing、worker 与子进程监督 | ~30% |
| 3 | 产物缓存、步骤内检查点、MOCK 步骤 | ~10% |
| 4 | API + Dashboard | ~8% |
| 5 | 故障矩阵、混沌测试、基准 | ~17% |
| 6 | Agent 层:Planner/护栏、诊断(LangGraph)、MCP、评测 | ~20% |
| 7 | README 与失败分析 | ~5% |

## 3. 长任务执行、状态管理与失败恢复的设计考量

**一句话**:至少一次执行 + 恰好一次提交;"谁在跑"由租约判定,"能不能提交"由 fencing token 判定;状态与历史在同一个事务里变化。

1. **租约 + fencing**。worker 在 `BEGIN IMMEDIATE` 事务里认领步骤并写入 `lease_expires_at` 与 `attempt_no`。
   `attempt_no` 同时是 fencing token:`complete / fail / heartbeat / save_checkpoint` 都必须带它且步骤仍是 `RUNNING`。
   租约被回收、别人重跑后,旧 worker(僵尸)的提交会被拒绝并留下 `COMMIT_REJECTED` 事件(测试 F6)。
2. **区分"慢"和"死"**。固定超时对长任务无解。子进程每次 `progress()` 都向 worker 上报,
   worker 用 `stall_timeout`(无进展)和 `timeout`(总时限)两个阈值判断,命中则**强杀子进程**并按 `STALL/TIMEOUT` 重试;
   "总耗时 3s 但每 0.3s 上报一次"的步骤不会被误杀(F7)。
3. **两层恢复粒度**。大粒度:把 `segment / swap` 按分块拆成独立步骤(缓存、重试、并行);
   小粒度:步骤内检查点。恢复时**先校验已落盘分片的 sha256,不信任检查点本身**。
4. **失败分类决定行为**:`TRANSIENT / RESOURCE` 指数退避重试;`PERMANENT` 不重试且下游 `BLOCKED`;
   `LEASE_LOST`(worker 失联)**不占失败预算**但有单独上限 `MAX_LOST=5`,防止"毒丸"步骤无限拖垮 worker。
5. **进程隔离**。子进程不碰数据库,只通过 stdout 的 JSON 行协议上报;worker 被 `kill -9` 时,
   子进程因 stdin EOF 自杀,不会成为继续写产物的孤儿(F4/F7b)。fencing 与状态正确性只在父进程一处保证。
6. **结果复用**。`cache_key = sha256(类型, 版本, 参数, 输入产物 hash)`,命中前校验产物文件 hash;
   篡改的产物视为未命中并被重算修复(F9)。生成式(非确定性)步骤靠"固化产物"而不是"重算"来保证前后一致(test_engine)。
7. **有外部副作用的 `publish`**:不进缓存,用幂等键去重 —— 重复执行只会发布一次(混沌测试 I5)。
8. **人工审批**是一个不占 worker、不计超时的状态 `AWAITING_APPROVAL`,可以挂几小时。
9. **可观测**:任务状态由步骤派生并在同一事务物化;`events` 追加式历史(状态变更、重试、缓存命中、被拒绝的提交、Agent 动作);
   `attempts` 记录每次尝试的 worker、子进程 pid、耗时、错误类别与诊断信息;`/api/metrics` 给出浪费耗时比例、缓存命中率等。

**Agent 层的设计考量**(面试要点):

- **LLM 只"想",引擎来"做"**。Planner 产出阶段级计划(JSON),系统确定性地展开为分块 DAG;LLM **不能**设置重试/超时/chaos(Pydantic `extra=forbid`)。
- **护栏**:结构 → 白名单 → 依赖/无环 → 业务策略(发布到生产库必须审批、发布必须质检)→ 预算(分块数/帧数/GPU 步骤数)→ source 路径白名单;
  所有违规一次性反馈给模型,最多 3 轮修复,仍不合规就 `REJECTED`,**不会勉强执行**。
- **LLM 输出也要持久化**:`plans` 表存每轮原始输出与错误,恢复任务不重新问 LLM。
- **诊断 Agent(LangGraph)**:`collect → diagnose →(建议重试)→ human_gate(interrupt)→ act`;
  图状态存 SqliteSaver,**进程重启后可继续且不会重复调用 LLM**(测试验证);写操作必须经人确认,
  执行时**以引擎当前状态为准**(确认前步骤已被别人重试 → 记录"未执行")。
- **权限**:Agent 工具只有读 + `retry_step/resume/cancel/submit`,**刻意没有 approve** —— Agent 不能批准自己负责的产物。MCP 写工具默认关闭。
- **LLM 追踪**:每次调用的 prompt / 输出 / token / 耗时 / 成本入 `llm_calls` 并关联任务,Dashboard 可看。
  `tokens_estimated` 明确区分"供应商真实用量"与"粗估"。

## 4. 任务步骤与依赖关系的定义

步骤声明(`StepSpec`):`key`、`type`、`deps`(DAG 边)、`params`、`version`(变化使缓存失效)、`resource`(cpu/gpu/none)、
`retry`(次数/退避)、`timeout_s`、`stall_timeout_s`、`cacheable`、`chaos`(故障注入,**不参与缓存 key**)。

```
                       ┌─► extract_audio ───────────────────────────────┐
probe ─► extract_frames ─┼─► segment_0..N-1 (gpu) ─┐                      ▼
   │                     │                         ├─► swap_0..N-1 ─► assemble ─► mux ─► quality_check ─► [human_approval] ─► publish
   └─► gen_reference (gpu,生成式,结果必须固化) ────┘
```

- **依赖规则**:提交时校验 DAG(key 唯一、依赖存在、无自依赖、Kahn 无环);`fan-out` 在提交时**静态展开**
  (`segment_i → swap_i` 按块对应,`swap_*` 汇入 `assemble`)。不做运行时动态 DAG,换取可解释性。
- **数据流**:步骤之间只通过**产物**(内容寻址文件)传递,不依赖内存/本地临时文件,所以任何步骤都能在任何 worker 重跑。
- **"可运行"是派生谓词**而非落库状态:`状态∈{PENDING,RETRY_WAIT} ∧ 到点 ∧ 所有依赖 SUCCEEDED ∧ worker 具备所需资源`。
- **上游失败**:失败步骤的所有 `PENDING` 下游标记 `BLOCKED`;人工重试后自动解除(固定点迭代,结果与调用顺序无关)。
- **为什么这样拆**(对应真实 AI pipeline):`probe` 轻量先行(坏输入尽早永久失败,不浪费 GPU);
  `segment/swap` 成本最高且 OOM 多发 → 按块独立重试/并行;`gen_reference` 非确定性 → 单独成步并固化;
  `quality_check` 失败语义不同(需人工判断)→ 不应触发上游重算;`publish` 有外部副作用 → 单独成步 + 幂等键。

## 5. 哪些状态被持久化,为什么

| 表 | 内容 | 为什么必须持久化 |
|---|---|---|
| `tasks` | 完整规格 JSON、状态、幂等键、请求原文 | 恢复的唯一依据;幂等提交 |
| `steps` | 状态、`attempt_no`(fencing)、失败/丢失计数、`next_run_at`、租约、`cache_key`、进度、`checkpoint`、产物引用、最近错误 | 崩溃后判定"该做什么、谁在做";检查点 |
| `step_deps` | DAG 边 | 派生"可运行"谓词 |
| `attempts` | 每次尝试:worker、子进程 pid、起止、结局、错误类别、诊断 notes | 失败史、审计、孤儿进程排查 |
| `artifacts` / `step_cache` | sha256、大小;`cache_key → 产物` | 结果复用;完整性校验 |
| `events` | 追加式状态变更与 Agent 动作 | 执行历史、时间线、增量订阅(`after=seq`) |
| `workers` / `approvals` | 能力与心跳;审批记录 | 可见性;审计 |
| `plans` / `llm_calls` / `diagnoses` | 规划原始输出与校验错误;每次 LLM 调用;诊断索引 | LLM 输出需固化(恢复不重问);可追溯成本;待确认列表 |
| `agent_checkpoints.db`(LangGraph) | 诊断图的推理进度 | 人工确认可能隔很久,期间进程会重启 |

**刻意不持久化**:内存调度队列(由数据库谓词派生,重启即恢复);步骤函数内**未存档**的局部状态(丢了就从检查点/开头重算,这是契约);
进度的每一次更新(随心跳节流写入,否则小时级任务会把库写爆);日志全文(只存错误摘要,`attempt_id` 可关联外部日志)。
**"以引擎为准"**:LangGraph 的 checkpoint 与引擎状态是两套持久化,冲突时(如确认前步骤已被他人重试)以引擎校验结果为准。

## 6. AI coding 工具与 token

- **工具**:Cursor(Agent 模式,对话式协作完成设计、实现、测试与调试)。
- **token 用量**:**【待你填写】** 我无法读取精确用量,请到 Cursor 的 Usage 页面查看本项目对应时间段的数字后填入。
  请不要用我的估算代替真实数据。
- **消耗最多的部分(定性判断,请与用量页对照后修正)**:
  1. 引擎核心的设计推导与实现(`engine.py` / `worker.py`,租约、fencing、状态转换)以及围绕它们的反复调试;
  2. 故障矩阵与混沌测试的编写与排障(`tests/`);
  3. Agent 层(Planner 护栏、LangGraph 诊断、MCP)——其中需要现查新版 `langgraph` / `mcp 2.x` 的真实 API,而不是凭记忆写。

## 7. 个人时间花费

**【待你填写】** 题目要求"不算使用 LLM 直接做的部分"。建议如实记录,常见的大头:
读懂并验证引擎的正确性论证(为什么 fencing 足够)、亲手跑 `kill -9` 与观察 Dashboard、
判断哪些护栏规则符合业务、接真实 LLM 后重跑评测并分析差异。

## 8. 评测结果

> 所有数字来自本仓库的测试与脚本,可复现。没有真实 LLM Key,**Agent 相关结果均来自 MockLLM**,见 8.3 的限定。

### 8.1 故障矩阵(`tests/`,58 个测试全部通过,约 2 分钟)

| # | 场景 | 验证的行为 |
|---|---|---|
| F1 / F1b | 瞬时失败 / GPU OOM | 退避重试到成功;**上游只执行一次**;attempts 保留每次失败的类别 |
| F2 / F2b | 永久失败 | 不重试;下游 `BLOCKED`;无关分支继续完成;`resume` 后只重跑失败步骤及其下游 |
| F3 | 预算耗尽 → 人工重试 | 任务 `FAILED`(可恢复);重试后上游产物 hash 不变、未重跑 |
| F4/F5/F7b | **真实 `kill -9` worker** | 孤儿子进程自杀;租约过期后被回收;**从检查点继续**;已成功步骤不重跑、产物不变;丢失不占失败预算 |
| F6 | 僵尸提交 | 过期租约持有者的 heartbeat / complete / fail 全部被拒绝;重复提交也被拒;留下 `COMMIT_REJECTED` |
| F7 | 卡死 vs 慢 | 无进展步骤被判 `STALL` 并**强杀子进程**;"慢但有进展"的步骤不被误杀;总时限独立生效 |
| F8 | 缓存复用 | 完全相同任务:除 `publish` 外全部命中缓存(`attempt_no=0`);只改下游参数时上游复用、下游重算 |
| F9 | 产物被篡改 | 校验失败视为未命中、重算并修复存储;重算得到同一 hash(确定性) |
| F10 | 4 个 worker 并发抢 3 个任务 | 每个步骤恰好执行一次,无 `COMMIT_REJECTED` |
| F11 / F12 | 取消 / 幂等提交 | 子进程被杀、不再启动新步骤;同一 `Idempotency-Key` 返回同一任务 |
| **F13** | **混沌**:5 个任务并发,期间随机 `kill -9` worker 5 次并补位 | I1 全部成功;I2 **最终产物 hash 与无故障基线逐一相同**;I3 每步骤至多一次有效提交;I4 无遗留 RUNNING 尝试;I5 外部发布每任务恰好一次;I6 无孤儿子进程 |

另有:状态机非法转换回滚、毒丸步骤(丢失预算耗尽)、退避时间指数增长、人工审批(挂起不占 worker / 拒绝后重新审批)、资源路由(cpu worker 不跑 gpu 步骤)、
非确定性步骤复用、API 错误码映射(404/409/422)、DAG 规格校验(重复/缺依赖/成环)。

混沌测试一次运行的实测:`kills=5`,`attempts={'SUCCEEDED': 70, 'LOST': 5, 'FAILED': 2}`,
`errors={'LEASE_LOST': 5, 'TRANSIENT': 2}`,`waste_ratio=0.021`(浪费耗时 / 有效耗时)。

### 8.2 恢复基准(`scripts/bench_recovery.py`:步骤 12 帧×0.4s,处理到一半时 `kill -9`,每组重复 2 次)

| lease_ttl | 检查点 | 恢复延迟 | kill 时已完成帧 | 恢复后重算帧 | 白白重复的帧 |
|---|---|---|---|---|---|
| 1s | 有 | 1.02s | 6 | 6 | 0% |
| 1s | 无(人为清除) | 1.02s | 6 | 12 | 50% |
| 2s | 有 | 1.99s | 6 | 6 | 0% |
| 2s | 无 | 2.04s | 6 | 12 | 50% |
| 4s | 有 | 4.01s | 6 | 6 | 0% |
| 4s | 无 | 4.05s | 6 | 12 | 50% |

结论:**恢复延迟 ≈ `lease_ttl`**(故障发现时间由租约决定,与检查点无关);检查点把重复计算降到 0。
`lease_ttl` 越小恢复越快,但对 GC 停顿/抖动更敏感(误判失联会导致重复执行,fencing 保证不会重复提交)。

### 8.3 Planner 评测(`python -m pipeline.agent.eval`,23 个用例,**MockLLM**)

| 指标 | 结果 | 含义与限定 |
|---|---|---|
| 放行/拒绝判断正确率 | 100% | 脚本化 Mock 与护栏规则自洽;**不代表真实模型** |
| 预期被护栏拦截的用例中,确实被拦并命中预期错误类别 | 100% | 对应策略/预算/QC/畸形输出/越权字段/路径穿越 |
| `unsafe_submitted`(应拒绝却提交 / 违反策略不变量) | **0** | 评测内用独立的不变量检查器再验一遍,并有测试证明该检查器会报错 |
| 业务参数准确率 | 100% | 同上,Mock 用正则解析 |
| 第 1 轮即合规率 | 50% | **这是 Mock 的脚本行为(故意先犯错),不是模型质量** |
| 放行用例平均规划轮数 | 1.5 | 同上 |

**更有说服力的是与 Mock 无关的 `test_validator_catches_every_single_mutation`**:对合法计划做 50+ 种单点破坏
(删依赖、非白名单类型、成环、越界参数、越权字段、去掉审批……),校验器必须全部拦下,实测无遗漏。
真实模型的各项指标需拿到 Key 后用 `--repeat 3` 重跑,并与上表对比。

## 9. 失败案例分析:做错的、做不了的

### 9.1 开发中踩到的真实错误(都由测试暴露)

1. **子进程退出时死锁**。第一个冒烟测试就失败:子进程用守护线程 `sys.stdin.buffer.read()` 监听父进程是否存活,
   解释器退出时抢不到 BufferedReader 的锁,报 `Fatal Python error`,步骤被判崩溃、白白重试(单测耗时 46s)。
   修复:直接 `os.read(0, ...)` 读文件描述符,并用 `os._exit` 退出。教训:**"守护线程 + 带锁缓冲 IO + 解释器关闭"是经典陷阱**。
2. **`__main__` 双重导入导致 `except` 失效**。`step_runner` 以 `__main__` 运行,步骤模块再 `import pipeline.step_runner`
   会得到另一份同名 `StepError` 类,于是 `except StepError` 抓不到步骤抛的异常。修复:把 `StepContext/StepError` 拆到独立模块。
3. **我自己写错的测试断言**:F10 一开始用相同 `source` 的任务,上游步骤跨任务命中缓存,`attempt_no` 为 0 而不是 1。
   这不是引擎 bug,而是我忘了"复用"本身就是被设计的行为。
4. **测试里的"陈旧字节码"**:一次编辑后 pytest 报的是旧版本断言,曾被我误判为代码问题;之后统一禁用字节码缓存再复跑。

### 9.2 已知缺陷与"做不了"的事(按严重程度)

| 问题 | 原因 / 影响 | 如果继续做 |
|---|---|---|
| **没有 single-flight**:两个完全相同的任务**同时**提交,同一步骤会被各算一次 | 缓存只在提交后才可命中,进行中的结果不共享;浪费但不影响正确性 | 对 `cache_key` 加"进行中"登记,后来者等待 |
| **API 与 Dashboard 无鉴权**,任何人可以调用 `approve` | MVP 范围外。"Agent 没有 approve 工具"只约束了 Agent,不约束 HTTP 调用者 | 接入身份与角色,审批绑定到人 |
| **Dashboard 未在真实浏览器中验证** | 我只做了 JS 语法检查(`node --check`)和 API 层测试,**没有人打开页面点过** | 用 Playwright 做端到端截图测试 |
| **没有接过真实 LLM** | 没有 Key。`OpenAICompatClient` 只用 httpx MockTransport 验证了请求格式/用量解析/错误追踪;真实模型的 JSON 遵循率、被护栏退回的比例都是未知数 | 拿到 Key 后 `eval --repeat 3` 并公布差异 |
| 评测的"第 1 轮合规率"等指标来自脚本化 Mock | 见 8.3 | 同上 |
| **GPU 调度只做到"按能力认领"** | 没有显存装箱、优先级、抢占、公平份额;也没采集 GPU 利用率/显存 | 设计见 `docs/DESIGN.md` 第 11 节,未实现 |
| SQLite 单机、写串行 | 水平扩展与多机需换 PostgreSQL(`SKIP LOCKED`)并改用数据库时间,现在依赖本机时钟 | 见 `docs/DESIGN.md` 第 9 节 |
| 缓存校验在写事务内全量算 sha256 | 大产物会长时间持有写锁,拖慢其他 worker 的认领 | 事务外校验 + 缓存校验结果(路径+mtime+size) |
| 子进程隔离 ≠ 容器隔离 | 没有 cgroup/显存限额;Windows 进程树清理依赖 `taskkill`;父进程被硬杀到子进程自杀之间有短暂窗口,孤儿可能多写几个文件(在私有目录,不影响正确性,只留垃圾) | 容器/cgroup;产物 GC |
| 未分类异常一律按 `TRANSIENT` 重试 | 真正的代码 bug 会被白白重试到预算耗尽才报 FAILED | 对重复相同堆栈的失败提前熔断 |
| `events` / `llm_calls` 无限增长,无归档 | MVP 未做保留策略 | 分区/归档 |
| 动态 DAG 不支持 | 分块数在提交时静态展开 | — |
| 诊断建议的质量完全取决于模型 | 系统只保证"未经人确认不会执行",不保证建议正确;Mock 的建议是固定规则 | 用真实失败样本评测诊断建议 |
| 混沌测试断言 `LOST>=1` 依赖时序 | 极端情况下 5 次 kill 全部打在空闲 worker 上会误报失败 | 固定 kill 时机或提高 kill 次数 |
| F7、F4 等测试含真实 sleep 与时序 | 机器很慢时可能偶发超时;`TIME_SCALE`/`lease_ttl` 在 `tests/conftest.py` 可调 | — |

### 9.3 与设计稿的差异

- 设计稿写"心跳只在有进展时续租,卡死则租约自然过期";**实际实现不是这样**:worker 只要活着就按固定间隔续租(`engine.heartbeat`),
  卡死由 worker 自己用 `now - last_progress > stall_timeout` 判定,并**主动强杀子进程、登记 `STALL`**。
  好处是卡死的发现不必等租约过期、且能立即回收子进程;代价是卡死检测依赖 worker 本身存活 ——
  worker 整个挂掉时,只能靠租约过期兜底(此时记为 `LEASE_LOST` 而不是 `STALL`)。
- 设计稿的 GPU 槽位演示只做了资源能力匹配。
- `mcp` 2.x 把 `FastMCP` 改名为 `MCPServer`,按新 API 实现并在测试中直接调用 `list_tools/call_tool`;未用真实 MCP 客户端(如 Cursor)走通 stdio 联调。

## 10. 加分项对照

| 加分项 | 状态 |
|---|---|
| 步骤级重试、恢复、结果复用,并说明不同失败场景的行为 | ✅ F1–F13 |
| 任务/步骤状态、历史的完整设计,可视化或查询接口 | ✅ 状态机 + `events/attempts` + REST + Dashboard |
| 秒级 → 小时级的设计变化 | 📄 `docs/DESIGN.md` 第 9 节(+ 基准验证了"恢复延迟≈lease_ttl"、检查点收益) |
| 结合 AI pipeline 分析哪些步骤适合独立成可恢复任务 | ✅ 第 4 节 + `docs/DESIGN.md` 第 10 节 |
| GPU 资源调度与可观测 | ⚠️ 仅能力匹配 + 设计文档,未实现装箱/抢占/利用率采集 |
| Agent 层(Planner/护栏、诊断、MCP、LLM 追踪、评测) | ✅ 已实现;**真实模型未验证**(见 9.2) |
