# 设计方案:可观测、可恢复的 AI 任务执行管线(MVP)

> 状态:设计稿(待确认后实现)。最终交付的 README 将在此基础上补充实测结果与失败案例。

## 1. 需求拆解

题目的核心不是"做一个能跑动画处理的流程",而是回答一组**执行语义**问题:

| 题目要求 | 抽象成的工程问题 | 验收方式 |
|---|---|---|
| 单步骤失败能恢复 | 步骤级重试、失败分类、重试预算 | 注入故障,断言只重跑失败步骤 |
| 执行进程中断能继续 | 进程崩溃后状态不丢、"谁在跑"可判定、不重复提交 | `kill -9` worker / 整个服务,重启后继续 |
| 任务执行时间长 | 步骤内检查点、心跳续租、不能靠固定超时 | 长步骤中途被杀,从检查点而非从头继续 |
| 上游能看到状态 | 任务/步骤状态模型、历史、查询接口、可视化 | HTTP API + 简易 Dashboard |
| 结果复用 | 内容寻址缓存、产物完整性校验 | 重提交同一任务命中缓存;只改下游参数时上游不重算 |

**设计原则**(对应评分:"更简单、更可控、更容易解释正确性"):

1. **SQLite 作为唯一事实来源**:不引入队列、工作流引擎、Redis。
2. **所有状态变更走同一个入口 + 同一个事务**:状态、历史事件原子写入,便于证明不变量。
3. **至少一次执行 + 恰好一次提交**:用租约 + fencing token 实现,而不是追求"恰好一次执行"(分布式下做不到)。
4. **步骤间只通过"产物"传递数据**:不依赖内存,所以任何步骤都可以在任何进程重跑。
5. **MOCK 计算,但故障是真实的**:真的 `kill -9`、真的写文件、真的校验 hash。

## 2. 总体架构

```
            ┌────────────── HTTP API / Dashboard(只读为主) ──────────────┐
            │  POST /tasks  GET /tasks[/id]  GET /tasks/id/events  ...     │
            └───────────────────────────┬─────────────────────────────────┘
                                        │
                                ┌───────▼────────┐
                                │  SQLite (WAL)  │  tasks / steps / attempts
                                │  事实来源       │  artifacts / events / workers
                                └───▲────────▲───┘
                       claim/commit │        │ claim/commit
                    ┌───────────────┴─┐   ┌──┴───────────────┐
                    │ Worker(cpu)     │   │ Worker(gpu:1)    │   可多进程、可随时被杀
                    │ 执行线程+心跳线程 │   │ 执行线程+心跳线程  │
                    └────────┬────────┘   └────────┬─────────┘
                             └──── 产物目录 data/artifacts/<sha256> ────┘
```

- 进程角色:`api`(查询/提交)、`worker`(可启动 N 个)。**没有常驻调度器**:调度 = worker 拉取(pull),
  回收过期租约也由每个 worker 在轮询循环里顺带完成。少一个角色就少一个单点和一类故障。
- 技术栈:Python 3.11;核心(状态机/引擎/worker/存储)**仅用标准库**(`sqlite3`、`subprocess`、`hashlib`),
  FastAPI + uvicorn **只用于接口层**,核心逻辑不依赖它(测试可不经 HTTP 直接调用引擎)。
  Dashboard 为单个静态 HTML + 轮询 JSON。
- **步骤在独立子进程中执行**(见 5.2):worker 进程负责认领/续租/提交,子进程只负责计算,
  这样卡死、超时、取消都可以真正 `kill`。

## 3. 任务与步骤定义

### 3.1 示例任务:动画素材处理

```
                      ┌─► extract_audio(cpu) ───────────────────┐
probe ─► extract_frames ─┬─► upscale_0..N-1 (gpu, 分块 fan-out) ─► assemble ─► mux ─► quality_check ─► publish
 (cpu)     (cpu)         └─► thumbnail(cpu) ───────────────────────────────────────────────────────────┘
```

- `probe`:校验输入、读元数据(坏输入在此"永久失败")。
- `extract_frames`:拆帧,产物为帧清单 + 帧文件。
- `upscale_i`:超分,**按帧区间分块**,每块是独立步骤(GPU)。这是长任务的关键切分点。
- `assemble`:合并各块 → 无声视频;`extract_audio`:并行分支;`mux`:合流。
- `quality_check`:抽检,可注入"质检不通过"→永久失败。
- `publish`:**有外部副作用**的步骤,带幂等键,单独成步。

### 3.2 步骤声明(Step Spec)

```python
StepSpec(
  id="upscale_3", type="upscale", version="1",   # version 变化 → 缓存失效
  deps=["extract_frames"],                         # 依赖 = DAG 边
  params={"chunk": 3, "scale": 2},
  resource="gpu",                                  # cpu | gpu,worker 按能力认领
  retry=RetryPolicy(max_attempts=4, base_delay=2, max_delay=60, jitter=0.2),
  timeout_s=600, stall_timeout_s=30,               # 总时限 / 无进展时限
)
```

- **依赖关系**:显式 `deps`(DAG)。提交时做环检测、未知依赖检测。
  fan-out 在**提交时由任务模板展开**为静态步骤并整体落库(不做运行时动态 DAG,换取可解释性)。
- **数据流**:步骤通过 `ctx.input(dep_id, name)` 获取上游产物引用(artifact id + 路径 + sha256),
  通过返回值声明自己的产物;**不得依赖内存或本地临时文件**。
- **任务规格落库**:恢复只依赖数据库里的 spec,不依赖进程内的模板对象。

## 4. 状态模型

### 4.1 步骤状态机

```
PENDING ──(依赖全部 SUCCEEDED 且 next_run_at<=now,被 worker 认领)──► RUNNING
RUNNING ──成功提交──► SUCCEEDED
RUNNING ──可重试失败/租约过期,且预算未尽──► RETRY_WAIT ──(到点)──► 被认领 → RUNNING
RUNNING ──永久失败 / 预算耗尽──► FAILED
PENDING ──上游 FAILED──► BLOCKED
PENDING/RETRY_WAIT/RUNNING ──cancel──► CANCELLED
FAILED/BLOCKED ──人工 resume/retry──► PENDING(attempt 预算重置,历史保留)
```

- **不持久化 READY**:"可运行"是派生谓词(状态∈{PENDING,RETRY_WAIT} ∧ 依赖全成功 ∧ 到时间)。
  少一个需要与依赖保持一致的冗余状态,正确性更容易论证。
- `SUCCEEDED` 带 `cached` 标记:区分"真算了"和"命中缓存直接复用"。

### 4.2 任务状态(由步骤派生,并物化到 tasks 表便于查询)

`PENDING` → `RUNNING` → `SUCCEEDED` / `FAILED` / `CANCELLED`

- 仍有可运行或运行中的步骤 → `RUNNING`(此时有 FAILED 步骤也只算"降级中",在 API 里单独标注 `has_failed_steps`)。
- 无可运行步骤且存在 FAILED → `FAILED`(**可恢复终态**,允许 `resume`)。
- 全部 SUCCEEDED → `SUCCEEDED`。
- 同一事务内更新,避免"步骤已变而任务未变"。

### 4.3 失败分类(决定行为,而不是只看异常类型)

| 类别 | 例子 | 行为 |
|---|---|---|
| `TRANSIENT` | 网络抖动、IO 临时错误 | 指数退避 + 抖动重试 |
| `RESOURCE` | GPU OOM | 重试,并通过 `ctx.attempt` 让步骤降 batch size(MOCK 演示) |
| `PERMANENT` | 输入损坏、质检不通过 | 不重试,直接 FAILED,下游 BLOCKED |
| `LEASE_LOST` | worker 崩溃/卡死,租约过期 | 记录一次 attempt=LOST,按可重试处理(不一定是步骤的错,计入单独预算) |
| `TIMEOUT/STALL` | 总时限超出 / 长时间无进展 | 视为可重试,但标注原因 |
| `CANCELLED` | 用户取消 | 协作式停止 |

## 5. 核心机制(本方案的重点)

### 5.1 认领、租约、fencing

1. worker 在 `BEGIN IMMEDIATE` 事务内:选一个可运行步骤 → 置 `RUNNING`,写
   `lease_owner / lease_expires_at / attempt_no`,并新建一条 `attempts` 记录。**整个认领是一次原子操作**。
2. 执行期间,心跳线程定期续租(`lease_expires_at = now + lease_ttl`)。
3. 提交结果时,`UPDATE ... WHERE step_id=? AND attempt_no=? AND status='RUNNING'`:
   **attempt_no 即 fencing token**。若租约已被回收并由别的 worker 重跑,旧 worker(僵尸)的提交影响行数为 0 → 丢弃。
4. 任何 worker 的轮询循环会扫描 `RUNNING ∧ lease_expires_at < now` 的步骤并回收。

> 保证:**同一步骤同一时刻至多一个有效提交者**;执行可能重复(至少一次),提交不会重复。

### 5.2 "慢"与"卡死"的区分(长任务的关键)

固定超时对长任务无解:设短了误杀,设长了发现故障太晚。方案:

- **心跳不等于存活**:worker 的续租只说明"worker 进程活着",不说明"步骤在前进"。步骤是否前进由子进程上报的
  `ctx.progress()` / 检查点判定:worker 发现 `now - last_progress > stall_timeout` 就**主动强杀子进程并登记 `STALL`**。
  (初稿设想为"无进展则停止续租、等租约过期",实现时改为主动判定,以便立即回收子进程;worker 整体挂掉时仍由租约过期兜底。)
- 两个独立阈值:`stall_timeout`(无进展)与 `timeout`(总时限)。
- **子进程隔离**:worker 为每个步骤起一个子进程(`python -m pipeline.step_runner`),职责切分如下:

  | | worker(父) | step_runner(子) |
  |---|---|---|
  | 访问数据库 | 是(认领、续租、提交、失败登记) | **否** |
  | 与对方通信 | 读子进程 stdout 的 JSON 行(`progress` / `checkpoint`),收到才续租 | 只写 stdout;结果写 attempt 私有目录 |
  | 超时/卡死/取消 | `kill` 整个子进程(Windows 用 `taskkill /T` 或 Job Object),记录 TIMEOUT/STALL/CANCELLED | 无需配合 |
  | 父进程被 `kill -9` | —— | 监听 stdin EOF,父死则自杀,避免孤儿进程继续占用 GPU/写产物 |

  好处:子进程不碰 DB,**fencing 与状态正确性只在父进程一处保证**;子进程崩溃(含 OOM-kill)天然表现为非零退出码,
  可与"业务异常"区分(分类为 `RESOURCE`/`TRANSIENT`)。
- 残留风险(失败案例中写明):父进程被硬杀与子进程自杀之间有短暂窗口,孤儿子进程可能多写若干文件,
  但其产物写在 attempt 私有目录、且没有提交权,**不会影响正确性,只会留下待 GC 的垃圾**。

### 5.3 步骤内检查点(秒级 → 小时级的桥梁)

- 步骤可调用 `ctx.save_checkpoint(json)`,持久化到 `steps.checkpoint`;重试时 `ctx.checkpoint` 可读到。
- 两层粒度:
  - **大粒度**:把长步骤切成 N 个独立步骤(如 `upscale_i`),天然获得缓存、重试与并行。
  - **小粒度**:单个长步骤内按"已处理帧数"存检查点。
- 检查点只保存**可重新验证**的信息(如已完成的分块产物文件名+hash),恢复时先校验再信任。

### 5.4 结果复用(内容寻址缓存)

```
cache_key = sha256(step.type, step.version, canonical(step.params), [输入产物 sha256...])
```

- 步骤认领前先查 `artifacts`/缓存索引:命中且文件 hash 校验通过 → 直接标记 `SUCCEEDED(cached=1)`,不占用 worker。
- 产物写入流程:写入 attempt 私有临时目录 → 计算 sha256 → `rename` 到 `artifacts/<sha256>`(原子)→ 在提交事务中登记。
  - 崩溃发生在 rename 与提交之间:留下孤儿文件,**无害**(内容寻址,可 GC)。
  - 数据库登记了但文件缺失/损坏:读取时校验失败 → 视为未命中,重算。
- 输入的 hash 参与 key,所以上游重算出不同结果时,下游自动失效;**未改变的上游保持复用**。
- 非确定性步骤(如生成式模型)**不能靠"重算"复用**,必须靠"持久化已有产物"复用——这正是缓存的价值。

### 5.5 状态变更的单一入口

所有状态变化经 `transition(step_id, from_states, to_state, ...)`:
- 校验合法转换表;
- 在同一事务里追加 `events`(append-only:`seq, ts, task_id, step_id, kind, from, to, attempt, detail`);
- 同步更新任务派生状态。

这使得"状态表"与"历史"不会不一致,并且事件流可直接支撑 API 与 UI 的增量拉取(`after_seq`)。

## 6. 持久化什么、为什么

| 表 | 内容 | 为什么必须持久化 |
|---|---|---|
| `tasks` | spec、状态、幂等键、创建/完成时间 | 恢复的唯一依据;幂等提交 |
| `steps` | 状态、attempt 计数、`next_run_at`、租约、cache_key、最新进度、checkpoint、最近错误 | 崩溃后判定"该干什么";检查点 |
| `attempts` | 每次执行:worker、起止、结局、错误类别与信息 | 失败史、可观测性、回收后的审计 |
| `artifacts` | sha256、路径、大小、产出步骤、创建时间 | 复用;完整性校验 |
| `events` | 追加式状态变更历史 | 执行历史、时间线、增量订阅 |
| `workers` | id、能力(cpu/gpu)、最近心跳 | 可见性(哪些 worker 活着、在跑什么) |

**刻意不持久化**:
- 内存调度队列(由 DB 谓词派生,重启即恢复);
- 步骤函数内部未存档的局部状态(丢了就从检查点/开头重算,这是显式契约);
- 高频进度的每一次更新(节流写入,只存最新值;否则小时级任务会把 DB 写爆);
- 日志全文(MVP 只存错误摘要;生产应进日志系统,以 attempt_id 关联)。

## 7. 可观测性

**查询接口**(JSON):

| 接口 | 作用 |
|---|---|
| `POST /tasks` | 提交(支持 `Idempotency-Key`) |
| `GET /tasks`,`GET /tasks/{id}` | 列表 / 详情(含步骤 DAG、进度、当前步骤、耗时、ETA) |
| `GET /tasks/{id}/events?after=seq` | 执行历史(增量) |
| `GET /tasks/{id}/steps/{sid}/attempts` | 单步骤全部尝试记录与错误 |
| `POST /tasks/{id}/cancel`、`/resume`,`POST /steps/{id}/retry` | 控制面 |
| `GET /workers`,`GET /metrics` | worker 存活与负载;队列深度、重试率、缓存命中率、等待时间 |

**Dashboard**:任务列表 + 单任务 DAG(按状态着色、显示 attempt 次数、cached 标记、进度条)+ 事件时间线。

## 8. 失败场景矩阵与预期行为(也是测试用例)

| # | 场景 | 注入方式 | 预期行为 |
|---|---|---|---|
| F1 | 步骤瞬时失败 | 前 2 次抛 TRANSIENT | 退避重试第 3 次成功;上游步骤**不重跑**;attempts 有 2 条失败记录 |
| F2 | 步骤永久失败 | 抛 PERMANENT | 不重试;下游 BLOCKED;任务 FAILED(可 resume);无关分支继续完成 |
| F3 | 重试预算耗尽 | 始终 TRANSIENT | FAILED,错误历史完整;人工 `retry` 后可继续,上游复用 |
| F4 | worker 被 `kill -9` | 步骤中途杀进程 | 租约过期后被他人回收;从检查点继续;`attempt=LOST` 有记录 |
| F5 | 全部进程中断后重启 | 杀 api + 全部 worker | 重启后一致继续;已成功步骤不重跑 |
| F6 | 僵尸提交 | 暂停 worker 超过租约后恢复 | 旧提交被 fencing 拒绝,结果不被覆盖/重复登记 |
| F7 | 卡死(慢 vs 死) | 步骤停止上报进度 | stall 超时后**子进程被强杀**并重试;对比"慢但有进展"的步骤**不被误杀** |
| F7b | 孤儿子进程 | `kill -9` worker | 子进程随之自杀(stdin EOF),不再继续写产物 |
| F8 | 缓存命中 | 重提交相同任务 / 仅改 publish 参数 | 命中的步骤 `cached=1`,耗时≈0 |
| F9 | 缓存损坏 | 篡改产物文件 | hash 校验失败,视为未命中并重算 |
| F10 | 并发认领 | 多 worker 抢同一批步骤 | 每步骤执行次数=1(无故障时),无重复提交 |
| F11 | 取消 | 运行中 cancel | 协作式停止;不再启动新步骤;状态 CANCELLED |
| F12 | 重复提交 | 同一 Idempotency-Key | 返回同一任务,不创建新任务 |
| F13 | 混沌测试 | 随机 kill 多个 worker,跑 N 个任务 | 全部最终完成;**最终产物 hash 与无故障基线一致**;每步骤至多一次有效提交 |

**评测指标**(报告里用数据说话,而不是"感觉能恢复"):
- 恢复正确性:F1~F13 通过率;混沌测试的不变量违例数(目标 0)。
- 浪费率:`重复执行耗时 / 有效耗时`——对比"有/无检查点"、"有/无缓存"。
- 恢复延迟:故障发生 → 步骤重新开始运行的时间(受 `lease_ttl` 影响,给出取值权衡)。
- 缓存命中率与命中节省的时间。

## 9. 时间从秒级扩展到小时级:哪些必须改

| 方面 | MVP(秒级) | 小时级需要的变化 |
|---|---|---|
| 超时判定 | 固定超时 | 以"进展"驱动的心跳续租 + stall/total 双阈值(MVP 已实现) |
| 恢复粒度 | 步骤级 | 步骤内检查点 + 切分为分块子任务;避免"跑 3 小时失败 2 小时 59 分" |
| 状态写入 | 每次进度写库 | 进度节流/合并;事件采样;分区或归档历史 |
| 存储 | 本地目录 | 对象存储(分段上传、断点续传、校验)+ 产物生命周期/GC 与保留策略 |
| 数据库 | SQLite 单机 | PostgreSQL(`FOR UPDATE SKIP LOCKED` 认领);用 DB 时间而非 worker 时钟 |
| 租约 TTL | 秒级 | 与心跳间隔解耦;容忍网络抖动与 GC 停顿;避免"心跳风暴" |
| 重试 | 固定退避 | 按失败类别区分预算;避免重试风暴;熔断依赖不健康的下游 |
| 抢占 | 无 | 软取消 + 检查点;spot/抢占式 GPU 被回收视为 `LEASE_LOST` 且不计入失败预算 |
| 用户体验 | 轮询 | ETA 估计、阶段耗时分布、暂停/恢复、优先级、告警(长时间无进展) |
| 副作用 | publish 幂等键 | 所有外部写必须幂等或可补偿;必要时引入"提交意图"表(outbox) |

## 10. 结合真实 AI pipeline:哪些步骤应拆成独立可恢复任务

判定标准(满足越多越应独立):
1. **成本高**(GPU 时间/外部 API 费用)——失败后重算代价大;
2. **产物体积大或需长期保留**——值得落盘复用;
3. **资源画像不同**(CPU / 单卡 / 多卡 / 高内存)——便于独立调度与计费;
4. **失败模式不同**(OOM、限流、模型加载失败)——重试策略不同;
5. **非确定性**——必须把产物"固化"才算复用;
6. **有外部副作用**——必须单独隔离并带幂等键;
7. **有人工介入点**(审核/打标)——天然的暂停点。

| 步骤 | 建议 | 理由 |
|---|---|---|
| 输入校验/探测 | 独立,但极轻量 | 尽早永久失败,避免浪费 GPU |
| 拆帧/转码 | 独立 | CPU/IO 密集,产物大,可被多个下游共享 |
| 超分/补帧/抠图(GPU) | **按分块切成独立任务** | 成本最高、可并行、OOM 多发,分块失败只重算该块 |
| 生成式步骤(文生图/视频) | 独立,产物固化 | 非确定性,重算得不到同一结果 |
| 合成/封装 | 独立 | 资源画像不同(CPU),便于限速 |
| 质检 | 独立 | 失败语义不同(永久失败/需人工),不应触发上游重算 |
| 发布/回调 | 独立 + 幂等键 | 外部副作用,至少一次触发必须可去重 |
| 缩略图、元数据等小步骤 | 可与邻近步骤合并 | 太细会增加调度与状态开销 |

## 11. GPU 调度与可观测性(以设计说明为主,MVP 做轻量演示)

- **MVP 实现**:步骤声明 `resource=gpu`;worker 以 `--resources gpu:1` 启动,按能力认领;一个 GPU 槽位同时只跑一个 GPU 步骤。
- **设计扩展**(文档说明,不实现):
  - 调度:按显存/卡数声明资源需求(`gpu_mem_gb`、`gpus`),装箱调度;优先级 + 公平份额;
    OOM 触发"降 batch 重试"或"换更大显存的池"重试;抢占式 GPU 的优雅回收。
  - 指标:队列等待时间(与执行时间分开统计)、GPU 利用率/显存峰值(按 attempt 采样关联)、
    GPU 小时数与失败浪费的 GPU 小时(成本归因)、模型加载/冷启动耗时占比。
  - 关联:`attempt_id` 贯穿日志、指标、trace,支持从"某个慢任务"下钻到具体卡与具体进程。

## 12. 仓库结构(拟)

```
.
├── README.md                # 交付说明(实现、取舍、token、失败案例)
├── docs/DESIGN.md           # 本文
├── requirements-dev.txt
├── pipeline/
│   ├── db.py                # schema、连接、事务封装
│   ├── models.py            # 状态枚举、转换表、StepSpec/RetryPolicy
│   ├── engine.py            # submit/claim/commit/fail/reclaim/resume/cancel(核心,单入口 transition)
│   ├── artifacts.py         # 内容寻址存储、原子写、校验
│   ├── worker.py            # 轮询、认领、续租、监督子进程、故障分类、提交
│   ├── step_runner.py       # 子进程入口:加载步骤、stdout 上报进度/检查点、父死自杀
│   ├── steps/animation.py   # MOCK 步骤实现(确定性输出 + 故障注入)
│   ├── templates.py         # 动画素材任务模板(DAG 展开)
│   ├── api.py               # 查询/控制接口(FastAPI)
│   └── cli.py               # python -m pipeline {api,worker,submit,status}
├── web/index.html           # Dashboard
├── tests/                   # F1~F13 + 混沌测试
└── scripts/demo_*.ps1       # 一键演示(含 kill -9)
```

## 13. 时间规划(目标约 1 个工作日,按优先级可裁剪)

| 阶段 | 内容 | 预估 |
|---|---|---|
| 0 | 需求分析、设计稿、环境(本文) | 1h |
| 1 | schema + 状态机 + 单一 transition 入口 + 单测 | 1.5h |
| 2 | 认领/租约/fencing/回收 + worker(含子进程监督/强杀)+ 重试分类 | 2.5h |
| 3 | MOCK 动画步骤 + 产物存储 + 缓存复用 + 检查点 | 1.5h |
| 4 | API + Dashboard | 1h |
| 5 | 故障矩阵测试 + 混沌测试 + 指标统计 | 1.5h |
| 6 | README(含失败案例、token 统计、取舍)| 1h |

**范围决定**(已确认):FastAPI 接口层 + 子进程隔离 + 核心功能 + F1~F13 测试 + 简易 Dashboard。
GPU 槽位仅做能力匹配,混沌测试用中等规模。

**裁剪顺序**(时间不够时先砍):GPU 槽位演示 → Dashboard 美化 → 混沌测试规模 → 步骤内检查点。
**不可砍**:租约 + fencing、单一 transition 入口、缓存复用、F1~F8 测试。

## 14. 预先承认的局限(将写入失败案例章节)

- SQLite 单机:多机分布式需换 PostgreSQL;当前方案证明的是**语义**,不是水平扩展能力。
- 子进程隔离不等于容器隔离:无 cgroup/显存限制,Windows 下进程树清理依赖 `taskkill`;孤儿子进程有短暂存活窗口。
- 步骤与 worker 间通过 stdout 行协议通信,协议本身(半行、子进程输出噪声)需要健壮解析。
- 依赖本机时钟判断租约;多机需统一用数据库时间。
- 外部副作用(publish)的"恰好一次"依赖对端支持幂等键,本地无法单方面保证。
- 动态 DAG(运行时才知道分块数)未支持;采用提交时静态展开。
- GPU 调度仅做能力匹配,不含显存装箱、抢占与公平调度。
