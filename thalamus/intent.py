"""任务与难度判定。

不是"第一个正则命中就走"，而是：
  ① 显式指定（model 别名 / X-Thalamus-Task 头）
  ② 结构信号（带图片、超长上下文、工具调用循环中）—— 这些是确定的
  ③ 多信号打分（代码 / 推理 / 工具型任务），分数最高且过阈值者胜
  ④ 难度 1-3。衡量"要多聪明"，不是"请求多大" —— 规模词、推理深度、工具步数、
     reasoning_effort。上下文长度只在真正罕见的量级（>200K）上算一档。
  ⑤ 置信度低的短追问（"继续""好的"）沿用同会话上一轮的任务
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .features import Features

TASK_ALIASES = {"chat", "code", "agent", "reason", "vision", "long", "utility"}
# thalamus/<这些> 都是合法的 model 名（auto/fast/pro 是档位，其余是任务别名）
ALIAS_NAMES = TASK_ALIASES | {"auto", "fast", "pro"}


def _en(words: str) -> str:
    """英文词边界。配合 re.ASCII 使用：中文不算 \\w，"用python写" 两侧都有边界。
    文本会先整体转小写，所以这里不用 re.I（re.I 在长文本上明显更慢）。"""
    return rf"\b(?:{words})(?!\w)"


# 只分析最后一条用户消息的开头与结尾：要求通常写在这两处，中间大段粘贴的材料对判断任务帮助很小，
# 而且正则扫描耗时与长度成正比（2 万字曾需要 24ms）。代码块数量与长度仍按全文统计。
HEAD_CHARS, TAIL_CHARS = 2000, 4000
_A = re.ASCII

_LANG = re.compile(
    _en(r"python|javascript|typescript|golang|rust|c\+\+|c#|java|sql|php|ruby|kotlin|swift|lua|bash|powershell|"
        r"react|vue|nodejs|node\.js|django|flask|fastapi|spring|flutter|unity"),
    _A,
)
# 开发工具名：几乎只在技术语境出现，权重介于语言名与普通术语之间
_TOOLS = re.compile(
    _en(r"git|github|docker|docker-compose|kubernetes|k8s|kubectl|nginx|npm|pnpm|yarn|pip|conda|maven|gradle|"
        r"webpack|vite|terraform|ansible|systemd|redis|postgres(?:ql)?|mysql|mongodb|sqlite|linux|ubuntu|vscode"),
    _A,
)
_CODE_TERMS = re.compile(
    r"代码|函数|报错|异常|编译|部署|接口|脚本|正则|数据库|单元测试|重构|调试|变量|依赖|环境变量|命令行|仓库|"
    r"列表|数组|字典|字符串|哈希表|指针|递归|算法|时间复杂度|去重|排序|组件|分页|反向代理|负载均衡|端口|证书|容器|"
    r"镜像源|源码|前端|后端|数据结构|爬虫|快排|内存泄漏|死锁|并发|线程|进程|"
    + _en(r"sdk|regex|debug|refactor|compile|deploy|stack ?trace|exception|unit ?tests?|"
          r"runtime|segfault|shell|bug|api|json|yaml|sse|websocket|http|code|frontend|backend|"
          r"endpoint|panic|null ?pointer|linter|script|container|commit|merge conflict|rebase|"
          # 荷兰语
          r"programmeren|programmeer|foutmelding|functie|functies|variabele|implementeer|implementeren|"
          r"compileer|compileren|klasse|methode|lijst|databank|broncode|uitrollen"),
    _A,
)
_CODE_SYNTAX = re.compile(
    r"(?m)^\s*(?:def |class |import |from \S+ import|function |const |let |var |public |private |#include|"
    r"package |func |fn |SELECT |INSERT |UPDATE |CREATE TABLE|local function)|=>|\)\s*\{\s*$|;\s*$",
)
# 用有界的 ASCII 字符类而不是 \w+，避免在超长无空格文本上做无谓的长匹配尝试。
_STACK = re.compile(
    r"Traceback \(most recent call last\)|^[ \t]+at [A-Za-z0-9_.$<>]{1,200}\(|Exception in thread|\bpanic: |error\[E\d+\]|"
    r"segmentation fault|Uncaught [A-Za-z]{0,40}Error|File \"[^\"\n]{1,300}\", line \d+|"
    r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_.]{0,80}(?:Error|Exception): ",
    re.M | re.I,
)
_FILE_EXT = re.compile(
    r"(?<![a-z0-9_\-./\\])[a-z0-9_\-./\\]{1,160}\.(?:py|js|ts|tsx|jsx|go|rs|java|kt|cpp|cc|hpp|cs|php|rb|lua|sh|ps1|sql|toml|vue|swift)(?![a-z0-9_])",
)
_CODE_BIG = re.compile(
    r"重构|架构设计|整个项目|多个文件|从零(?:开始)?(?:实现|搭建|写)|完整实现|系统设计|迁移到|"
    + _en(r"implement (?:a|an|the)|from scratch|end-to-end|design (?:a|the) system|whole (?:codebase|project)|"
          r"vanaf nul|helemaal opnieuw|het hele project|de hele codebase"),
    _A,
)
_REASON_TERMS = re.compile(
    r"为什么|原因|分析|对比|比较|权衡|利弊|优缺点|策略|规划|评估|推理|证明|推导|假设|风险|决策|取舍|本质|"
    r"怎么看|如何看待|解释|"
    + _en(r"why|explain|trade-?offs?|pros and cons|analy[sz]e|compare|strategy|evaluate|assess|prove|derive|implications?|"
          r"risks?|recommend|"
          # 荷兰语
          r"waarom|analyseer|analyse|vergelijk|vergelijking|afweging|afwegen|voor- en nadelen|voordelen en nadelen|"
          r"strategie|evalueer|beoordeel|bewijs|risico|risico's|onderbouw|verklaar"),
    _A,
)
_REASON_DEEP = re.compile(
    r"证明|推导|深入|详细分析|全面分析|系统性|一步一步|"
    + _en(r"step by step|rigorous|in[- ]depth|thorough|stap voor stap|diepgaand|grondig"),
    _A,
)
_MATH = re.compile(
    r"[∑∫√∞≤≥≠±∂]|\\(?:frac|sum|int|sqrt)|\^\d|方程|概率|期望值|导数|积分|矩阵|定理|整除|素数|质数|多项式|不等式|"
    r"根号|无理数|有理数|数列|极限|"
    + _en(r"theorem|integral|derivative|probability|prime numbers?|irrational|priemgetal|stelling"),
    _A,
)
_NUMBERED = re.compile(r"(?m)^\s*(?:\d+[.、)]|[-*•])\s+\S")
_UTILITY_TERMS = re.compile(
    r"翻译|总结|摘要|概括|提取|抽取|分类|归类|改写|润色|纠错|校对|格式化|打标签|关键词|"
    + _en(r"translate|summari[sz]e|tl;?dr|extract|classify|categori[sz]e|rewrite|rephrase|proofread|paraphrase|"
          # 荷兰语
          r"vertaal|vertaling|samenvatting|vat samen|samenvatten|herschrijf|herformuleer|extraheer|classificeer|corrigeer"),
    _A,
)
_FOLLOWUP = re.compile(
    r"^\s*(?:继续|接着|然后呢|还有呢|好的?|嗯+|对|是的|不对|再来|改一下|换一种|重新来|再试试?|可以|行|没问题|"
    r"ok(?:ay)?|go on|continue|next|yes|no|thanks?|谢谢|"
    r"ga door|ga verder|verder|doorgaan|ja|nee|oké|dank je(?:wel)?|bedankt|prima|top)[\s。.!！?？,，~]*$",
    re.I,
)
# 用户在纠正上一轮回答。只看开头：后面常常跟着贴出来的报错。
_CORRECTION = re.compile(
    r"不对|错了|不是这样|不是我要的|不行|没用|还是(?:不行|报错|错|有问题)|没解决|答非所问|"
    + _en(r"wrong|incorrect|not right|doesn'?t work|didn'?t work|not working|still (?:broken|failing|wrong|an error|getting)|"
          r"that'?s not (?:it|what)|not what i (?:asked|wanted)|"
          r"klopt niet|werkt niet|nog steeds|verkeerd|dat is niet wat"),
    _A,
)


def is_correction(text: str) -> bool:
    return bool(_CORRECTION.search(text[:120].lower()))


@dataclass
class Intent:
    task: str
    difficulty: int
    confidence: float
    source: str  # explicit | structure | rule | session
    signals: list[str] = field(default_factory=list)
    latency_boost: float = 1.0
    regenerate: bool = False
    correction: bool = False

    def label(self) -> str:
        return f"{self.task}/{self.difficulty}"


def _clamp(d: int, hi: int = 3) -> int:
    return max(1, min(d, hi))


def _effort(f: Features, d: int) -> int:
    if f.reasoning_effort in ("high", "xhigh", "max"):
        return 3
    if f.reasoning_effort == "medium":
        return max(d, 2)
    if f.reasoning_effort in ("low", "minimal", "none"):
        return 1
    return d


def parse_alias(model: str | None, header_task: str | None) -> tuple[str | None, int | None, float]:
    """thalamus/code、thalamus/reason-3、thalamus/fast、thalamus/pro。返回 (task, difficulty, latency_boost)。"""
    raw = (header_task or "").strip().lower()
    if not raw and model:
        m = model.strip().lower()
        if m.startswith("thalamus/"):
            raw = m.split("/", 1)[1]
    if not raw or raw == "auto":
        return None, None, 1.0
    if raw == "fast":
        return "chat", 1, 3.0
    if raw == "pro":
        return None, 3, 1.0
    name, _, lvl = raw.partition("-")
    if name in TASK_ALIASES:
        return name, (_clamp(int(lvl)) if lvl.isdigit() else None), 1.0
    return None, None, 1.0


@dataclass
class _Scores:
    code: float
    reason: float
    utility: float
    utility_head: bool
    fences: int
    big: bool
    deep: bool
    math: bool
    signals: list[str]


def _window(text: str) -> str:
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return text
    return text[:HEAD_CHARS] + "\n" + text[-TAIL_CHARS:]


def _score(f: Features) -> _Scores:
    full = f.last_user
    win = _window(full)  # 保留大小写：语法/堆栈正则需要
    low = win.lower()  # 关键词正则用小写文本 + re.ASCII，比 re.I 快得多
    sig: list[str] = []
    fences = full.count("```") // 2
    code = 0.0
    if fences:
        code += 3.0 + (1.0 if fences >= 3 else 0.0)
        sig.append(f"fences:{fences}")
    if _STACK.search(win):
        code += 4.0
        sig.append("stacktrace")
    code += min(len(_LANG.findall(low)), 2) * 2.0
    code += min(len(_TOOLS.findall(low)), 2) * 1.5
    code += min(len(_FILE_EXT.findall(low)), 3) * 1.0
    code += min(len(_CODE_SYNTAX.findall(win)), 6) * 0.6
    code += min(len(_CODE_TERMS.findall(low)), 5) * 0.8
    big = bool(_CODE_BIG.search(low))
    if big:
        code += 1.5
    if f.prev_user:
        prev = f.prev_user[-2000:].lower()
        code += min(len(_CODE_TERMS.findall(prev)) + len(_LANG.findall(prev)), 3) * 0.3

    reason = min(len(_REASON_TERMS.findall(low)), 5) * 1.2
    math = bool(_MATH.search(low))
    deep = bool(_REASON_DEEP.search(low))
    if math:
        reason += 1.5
        sig.append("math")
    if deep:
        reason += 1.0
    if len(_NUMBERED.findall(win)) >= 3:
        reason += 0.8
    if len(full) > 300 and (win.count("?") + win.count("？")) >= 2:
        reason += 0.5

    utility = min(len(_UTILITY_TERMS.findall(low)), 3) * 2.2 + (2.0 if f.json_mode else 0.0)
    head = bool(_UTILITY_TERMS.search(low[:80]))
    if f.n_user_turns > 2:
        utility *= 0.4
        head = False
    return _Scores(code, reason, utility, head, fences, big, deep, math, sig)


def classify(
    f: Features,
    long_threshold: int,
    alias_task: str | None = None,
    alias_difficulty: int | None = None,
    latency_boost: float = 1.0,
) -> Intent:
    text = f.last_user
    s = _score(f)
    signals = s.signals

    # 难度衡量的是"这个任务需要多聪明的模型"，不是"这个请求有多大"。
    # 请求大小已经按 token 实打实计价了，再让它抬难度等于对同一件事收两次费 ——
    # 而难度 1→3 的每分价值跳 7.5 倍，于是"贴 58K 上下文问一句小事"会被定价成
    # 最难的任务，只有先验最高的模型能过够用线。2026-10-06 实测：同一句"跑一下测试"，
    # 空上下文走 deepseek-flash，带 58K 上下文走 kimi-k2.7-code。
    # 所以长度只保留一档、且阈值放到真正罕见的量级上。
    def base_difficulty() -> int:
        d = 1
        if len(text) > 1500 or f.est_tokens > 200_000:
            d += 1
        return d

    def code_difficulty() -> int:
        d = base_difficulty()
        if s.big:
            d += 1
        if s.fences >= 2:  # 多个代码块 = 多处改动；行数多只说明贴得多
            d += 1
        return _clamp(d)

    def reason_difficulty() -> int:
        d = base_difficulty()
        if s.reason >= 4:
            d += 1
        if s.deep or s.math:
            d += 1
        return _clamp(d)

    # ① 显式
    if alias_task or alias_difficulty:
        task = alias_task or ("code" if s.code >= 3 else "reason" if s.reason >= 2.4 else "chat")
        d = alias_difficulty or (
            code_difficulty() if task in ("code", "agent") else reason_difficulty() if task == "reason" else base_difficulty()
        )
        return Intent(task, _clamp(d), 1.0, "explicit", signals + ["alias"], latency_boost)

    # ② 结构
    if f.n_images:
        d = reason_difficulty() if s.reason >= 2.4 else code_difficulty() if s.code >= 3 else base_difficulty()
        return Intent("vision", _effort(f, _clamp(d)), 0.95, "structure", signals + [f"images:{f.n_images}"])
    if f.est_tokens >= long_threshold:
        d = 2 + (1 if s.reason >= 2.4 or s.code >= 3 or f.est_tokens > 400_000 else 0)
        return Intent("long", _effort(f, _clamp(d)), 0.9, "structure", signals + [f"tokens:{f.est_tokens}"])
    if f.has_tools and f.in_tool_loop:
        # 用本轮的工具步数，不用声明的工具数量：后者是客户端配置
        # （编码 agent 每次都声明 15-20 个），对"这次难不难"没有任何信息。
        d = 1
        if f.loop_depth >= 5 or f.n_user_turns >= 8:
            d += 1
        if f.loop_depth >= 15 or s.big:
            d += 1
        return Intent("agent", _effort(f, _clamp(d)), 0.9, "structure",
                      signals + [f"tool_loop:{f.loop_depth}"])

    # ③ 打分
    if _FOLLOWUP.match(text) or (len(text.strip()) <= 4 and s.code == 0 and s.reason == 0 and s.utility == 0):
        return Intent("chat", _effort(f, 1), 0.1, "rule", signals + ["followup"])
    if s.utility_head and s.utility >= 2.2 and s.code < 3:
        # "总结/翻译 + 一段材料"：材料里的词不应决定任务
        d = 1 + (1 if len(text) > 6000 else 0)
        return Intent("utility", _clamp(_effort(f, d), 2), min(1.0, s.utility / 5), "rule", signals + [f"utility:{s.utility:.1f}"])
    if s.code >= 3 and s.code >= s.reason * 0.8:
        return Intent("code", _effort(f, code_difficulty()), min(1.0, s.code / 7), "rule", signals + [f"code:{s.code:.1f}"])
    if s.reason >= 2.4:
        return Intent("reason", _effort(f, reason_difficulty()), min(1.0, s.reason / 6), "rule", signals + [f"reason:{s.reason:.1f}"])
    if s.utility >= 2.2 and s.code < 3:
        d = 1 + (1 if len(text) > 6000 else 0)
        return Intent("utility", _clamp(_effort(f, d), 2), min(1.0, s.utility / 5), "rule", signals + [f"utility:{s.utility:.1f}"])
    if s.code >= 2:
        return Intent("code", _effort(f, code_difficulty()), 0.4, "rule", signals + [f"code:{s.code:.1f}"])
    d = base_difficulty() + (1 if len(text) > 800 else 0)
    return Intent("chat", _effort(f, _clamp(d)), 0.4, "rule", signals)


def inherit(intent: Intent, prev_task: str | None, prev_difficulty: int | None) -> Intent:
    """低置信度的追问沿用上一轮任务，避免"继续"被当成闲聊而换到弱模型、丢缓存。"""
    if (
        intent.source == "rule"
        and intent.confidence < 0.35
        and prev_task in ("code", "agent", "reason", "long", "vision")
    ):
        task = "code" if prev_task == "vision" else prev_task
        return Intent(
            task,
            max(intent.difficulty, prev_difficulty or 1),
            0.5,
            "session",
            intent.signals + [f"inherit:{prev_task}"],
            intent.latency_boost,
            intent.regenerate,
        )
    return intent
