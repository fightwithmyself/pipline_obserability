"""失败处置手册(runbook):把"步骤最终失败"的错误文本对应到 已知场景 + 能不能自动恢复 + 人工怎么恢复。

为什么不全交给 LLM:
- "能不能自动重试"是**安全边界**,不能由模型的一句话决定:认证/欠费/模型未开通这类错误,自动重试只会白烧额度、刷屏。
  手册只**限制**模型(模型说"重试"但手册说"不行"→ 降级为人工处理),从不放大它的权限。
- 人工恢复步骤是**确定性知识**(换哪个环境变量、重启 worker、哪个参数重新提交),模型没见过本项目,容易编造。
- 匹配的是本项目自己抛出的错误文本(`pipeline/steps/anime.py`、`pipeline/doubao.py`),文本改了要同步改这里
  (`tests/test_runbook.py` 用真实错误文本做回归)。

没有匹配到的错误:不自动恢复,走通用的人工排查步骤。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Playbook:
    code: str
    title: str                      # 一句话根因
    category: str                   # 与 Diagnosis.category 取值一致
    auto_retry: bool                # 允许 Agent 不经人确认直接重试(仅限低风险、无副作用放大的场景)
    max_auto: int = 0               # 同一步骤累计最多自动重试几次
    human_steps: tuple[str, ...] = field(default_factory=tuple)   # 自动恢复不了时,给人的操作清单


def _p(*a, **k) -> Playbook:
    return Playbook(*a, **k)


# 顺序即优先级(靠前的先匹配)。match 的依据都是真实抛出过的错误文本。
_RULES: list[tuple[callable, Playbook]] = [
    (lambda k, m: "auth:" in m or "ARK_API_KEY" in m,
     _p("auth", "豆包鉴权失败(密钥缺失/不对/与接口不匹配)", "permanent", False, 0, (
         "确认 worker 进程里的环境变量 ARK_API_KEY 正确:worker 的环境在启动时就固定了,改了变量必须重启 worker。",
         "普通按量密钥用 --provider ark;Agent Plan 专属密钥用 --provider ark_plan(专属路径 /api/plan/v3),两者不通用。",
         "如果密钥泄露过(例如贴到过聊天),先在控制台轮换再设置。",
         "修好后在看板点该节点的“重试此步骤”,已完成的帧会命中账本/缓存,不会重复计费。"))),
    (lambda k, m: "quota:" in m,
     _p("quota", "豆包额度不足或账户欠费", "resource", False, 0, (
         "到火山方舟控制台检查余额/套餐额度(Agent Plan 看套餐用量页)。",
         "充值或等额度刷新后,点“重试此步骤”;已完成的帧不会重复计费。"))),
    (lambda k, m: "model_not_open:" in m,
     _p("model_not_open", "账号未开通该模型", "permanent", False, 0, (
         "在方舟控制台开通该模型后点“重试此步骤”;或换一个已开通的模型 ID 重新提交任务(换模型后旧帧缓存不再命中,会重新计费)。"))),
    (lambda k, m: "not_found:" in m,
     _p("model_not_found", "模型 ID 不存在/已下线/当前密钥无权限", "permanent", False, 0, (
         "核对 --model 是否写对、是否已下线;确认该密钥所属的套餐支持这个模型。",
         "重试不会有帮助;请换模型 ID 后重新提交任务。"))),
    (lambda k, m: "bad_request:" in m,
     _p("bad_request", "豆包拒绝了请求参数(HTTP 4xx 未归类)", "permanent", False, 0, (
         "看错误详情里的 code/message,确认是哪个参数被拒(尺寸、模型不支持的字段等)。",
         "未知 4xx 一律按致命处理,是为了不对未知错误无限重试烧钱;确认是偶发问题后可手动“重试此步骤”。"))),
    (lambda k, m: "预算保护线" in m,
     _p("budget", "触及预算保护线,已主动停止", "resource", False, 0, (
         "这不是故障,是保护。确认花费合理后,用更高的 --budget 重新提交同样的任务:已完成的帧命中帧缓存,不会重复计费。"))),
    (lambda k, m: "内容审核拒绝" in m,
     _p("moderation", "部分帧被内容审核拒绝(on_reject=fail)", "quality", False, 0, (
         "看帧网格里红色的帧(点开看错误),对这些帧在 frame_overrides 里换提示词后重新提交;其余帧命中缓存,不重复计费。",
         "如果可以接受这些帧保持原图,改用 on_reject=passthrough 重新提交。"))),
    (lambda k, m: "降级(原图直通)" in m,
     _p("too_many_degraded", "原图直通的帧超过上限,风格化效果不可接受", "quality", False, 0, (
         "检查提示词是否容易触发审核、素材是否包含敏感内容;调整后重新提交。"))),
    (lambda k, m: "质检未通过" in m,
     _p("qc_failed", "质检未通过", "quality", False, 0, (
         "点 qc 节点看报告里的 hard_failures / warnings(帧数、尺寸、黑帧、结构保真度)。",
         "质检失败是业务判断,重试结果不会变;修改参数(提示词、--stride、--interp)后重新提交,"
         "或在确认可接受的情况下关闭 qc_strict。"))),
    (lambda k, m: any(s in m for s in ("素材文件不存在", "素材无法读取", "素材不满足", "素材文件在提交后被修改",
                                       "参考图不存在", "参考图无法读取", "参考图在提交后被修改")),
     _p("bad_input", "输入素材/参考图有问题", "bad_input", False, 0, (
         "按错误信息修复素材(路径、格式、时长、是否被改动),然后重新提交任务;重试同一个任务不会有帮助。"))),
    (lambda k, m: any(s in m for s in ("malloc", "Could not open encoder", "Cannot allocate", "out of memory")),
     _p("host_memory", "宿主机内存不足导致转码失败", "resource", True, 2, (
         "关闭占内存的程序(浏览器等)、检查虚拟内存/页面文件大小,再点“重试此步骤”。",
         "ffmpeg 的详细输出在该步骤最近一次尝试的 stderr.log 里(数据目录 attempts/<步骤id>-<尝试号>/)。"))),
    (lambda k, m: "帧失败" in m and "将重试" in m,
     _p("frames_failed", "部分帧在重试预算内仍失败(限流/网络/服务端错误)", "transient", True, 2, (
         "看帧网格里红色帧的错误类型:限流(429)就降低 --concurrency 或稍后再试;网络错误检查代理/网络。",
         "已完成的帧不会重做,点“重试此步骤”只会补做失败的帧。"))),
]

_KIND_RULES: dict[str, Playbook] = {
    "LEASE_LOST": _p("poison", "步骤反复让 worker 失联(疑似毒丸/被反复杀死)", "poison", False, 0, (
        "不要无脑重试:先看 worker 日志和该步骤 attempts 目录里的 stderr.log,确认是内存爆掉、被系统杀掉还是代码崩溃。",
        "修复原因后再点“重试此步骤”。")),
    "TRANSIENT": _p("transient", "重试预算耗尽的瞬时错误", "transient", True, 2, (
        "先看错误详情;如果是网络/服务端抖动,稍后点“重试此步骤”。上游产物都已缓存,重试代价低。")),
    "TIMEOUT": _p("timeout", "步骤超时", "transient", True, 2, (
        "确认是素材太大还是机器太慢;可调大 timeout/stall 或减小分块后重新提交。")),
    "STALL": _p("stall", "步骤长时间没有进展(疑似卡死)", "transient", True, 2, (
        "看该步骤最近一次尝试的 stderr.log 和心跳;确认外部服务是否卡住后再重试。")),
    "RESOURCE": _p("resource", "资源不足", "resource", True, 2, (
        "减小并发/分块或释放资源后再重试。")),
}

GENERIC = _p("unknown", "未知错误(没有对应的处置手册)", "unknown", False, 0, (
    "点该节点看“尝试历史”里的错误与 stderr.log,以及事件时间线里这个步骤的状态变化。",
    "确认原因后点“重试此步骤”;如果是输入或参数问题,修正后重新提交任务。",
    "这类错误的恢复方式我没有预先确认过,Agent 不会自动重试。"))


def match(error_kind: str | None, error_msg: str | None) -> Playbook:
    """按 具体错误文本 → 错误类别 的顺序匹配;都不匹配返回 GENERIC。"""
    m = error_msg or ""
    for pred, pb in _RULES:
        if pred(error_kind or "", m):
            return pb
    return _KIND_RULES.get(error_kind or "", GENERIC)
