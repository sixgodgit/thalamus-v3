# Jev 作为 Thalamus 打分后模型仲裁器

**状态：设计提案，尚未实现。** 目标是在 `thalamus/auto` 的合格模型难以区分时，让 Jev 从少量候选中选择模型，同时保留 Thalamus 的任务分类、硬约束、渠道选择、额度控制和故障回退。Jev 不连接 Nyx；首版通过严格的字段白名单隔离保险箱数据、支付资料和工具调用内容。

## 接入边界

现有请求路径是 [`Engine.understand`](../thalamus/engine.py) 分类任务、[`Router.plan`](../thalamus/router.py) 过滤并打分、[`Call.prepare`](../thalamus/engine.py) 预留额度，然后连接上游。仲裁只插在 `plan = make(self.mode)` 之后、预留额度之前。它不修改 `Intent.task` 或 `difficulty`，也不生成路由器未批准的新候选。

```text
客户端请求 → 本地特征和意图 → Router.plan：硬过滤、死线、打分
                                  │
                                  ├─ 胜者稳定或无有效取舍 → 原排名
                                  └─ 不同模型且胜者不稳定 → Jev 在近分候选中选择
                                                               │
                            本地复核候选、必要时重排 → 额度预留 → 连接和现有容错
```

`RoutePlan.ranked` 是首版唯一可供仲裁的集合。`rejected` 同时包含硬拒绝和超出延迟死线的候选，不能从其文字原因中捞回模型。当前所有候选超死线时，路由器会把这些候选放回 `ranked`；这类计划应显式标记并交给本地延迟策略处理，首版不调用 Jev。

## 何时调用

先按 `model.key` 将 `ranked` 去重，每个模型保留本地得分最高的渠道 offer，再取至多三个接近首名的模型。以下情况直接使用原排名：客户端点名具体模型、去重后可选模型不足两个、纯价格或渠道延迟造成的取舍、当前用户轮次已经调用过 Jev。同一模型的不同渠道仅由本地选，不计作两个候选。

“不确定”不能用固定美元分差定义。[任务价值表](../config/policy.yaml)中每分质量的美元价值跨多个数量级；当前路由分数和健康统计只有点估计，没有经过校准的置信区间。初版触发器同时检查：

1. 不同模型的未舍入分数差，相对于该任务的质量价值足够小；
2. 在质量先验和延迟估计的合理扰动下，首选模型可能翻转；
3. 候选存在质量与交互速度之间有意义的取舍，足以值得一次额外决策调用。

扰动范围和阈值先在影子评测中校准。之后用实际任务结果与延迟分布估计评分误差，替代人为阈值。Jev 返回的置信度用于决定是否采纳其选择，不能反过来定义本地打分是否不确定。

## Jev 的输入与输出

首版只发送去标识的本地结构摘要，不发送原始请求体、完整对话、工具名称或参数、工具结果、密钥及 Nyx 内容。摘要包含任务类别与难度、结构标记（有无图片、工具、输入长度档位、交互优先级），以及候选的临时 ID、适用任务、质量是否达到够用线、预计延迟档位、成本档位和近期失败档位。美元运算与硬约束始终由本地代码完成。

`ArbitrationContext` 必须从允许的标量字段逐项构造，严禁直接序列化请求 `body`、`Features`、`Intent`、`Candidate` 或 `RoutePlan`。白名单同时覆盖 Jev 请求的 `state`、`questions.instructions`、`Choice.criteria` 与 `assets`：不传图片数据或 URL，不把动态工具说明、用户原文、模型与渠道内部名称或精确价格拼进选项描述。`shadow` 调用同样执行这些检查；没有 Nyx API 连接本身并不能阻止请求文本泄露 Nyx 内容。

区域受限的 key 在 `shadow` 和 `active` 模式下均默认跳过远程仲裁。TypeSafe [目前在美国处理请求](https://help.typesafe.ai/articles/5974925716-how-is-submitted-data-used-or-retained)；去标识摘要仍经过第三方服务处理，是否允许该数据出境不能由最终模型的区域过滤代替。

TypeSafe 的 [`POST /v1/systemone`](https://docs.typesafe.ai/api/system-one) 允许在一次请求内使用动态 `Choice.criteria`。候选以 `c0`、`c1`、`c2` 命名，并加入 `abstain`；返回的选择必须严格匹配本次发出的 ID。若只有数字分数与模型名，没有独立的任务或能力信息，Jev 只是重做现有打分，影子评测须把这种输入与本地确定性校准器比较。TypeSafe 也建议将数值计算留给代码，`Choice` 用于有明确选项的语义判断：[Choice 文档](https://docs.typesafe.ai/primitives/choice)、[模型局限](https://docs.typesafe.ai/model-jaggedness)。

一次请求的形状如下；候选 ID 与描述从本次 `RoutePlan` 生成，不接受 Jev 自行添加模型：

```json
{
  "model": "jev-1.14.0",
  "state": {
    "task": "agent",
    "difficulty": 1,
    "interaction": "live",
    "has_tools": true
  },
  "questions": {
    "best_model": {
      "type": "choice",
      "instructions": "Choose the eligible candidate most likely to complete this task within the interaction needs; abstain when the summary does not distinguish them.",
      "criteria": {
        "c0": "Meets the quality need; fast expected response; low cost",
        "c1": "Exceeds the quality need; slower expected response; higher cost",
        "abstain": "Insufficient evidence to prefer either candidate"
      }
    }
  }
}
```

`Choice` 的概率用于判断 Jev 是否有明确偏好，不等于模型完成任务的真实概率；采纳门槛需按候选数和影子数据校准。

首版的去标识摘要没有具体用户目标。两个同为 `reason/2`、输入长度相近的问题会产生近乎相同的 Jev 输入，所以首版只能验证“元数据仲裁”是否优于本地权重校准，不能声称能针对具体问题选模。若要做任务级语义仲裁，需要单独设计并验证只含当前目标的文本来源与脱敏规则，再确认允许发送的字段；这不要求 Jev 连接 Nyx，但不能从“Jev 不访问 Nyx”推断为已满足数据隔离。

接口建议：

```python
class RouteArbiter(Protocol):
    async def choose(self, context: ArbitrationContext) -> ChoiceResult | None: ...

# ChoiceResult 只含本次候选 ID、概率及提供者版本；不能返回任意模型或渠道。
```

Thalamus 已依赖 `httpx`，可用独立异步客户端实现 Jev 适配器。配置分 `off`、`shadow`、`active`。客户端只使用部署环境中的专用 `TYPESAFE_API_KEY`，绝不转发用户的 `Authorization`、上游密钥或管理口令，也不把密钥写入仓库。设置独立的短超时、并发上限与费用上限，关闭重试；超时、服务错误、低把握、`abstain` 或无效 ID 都立即回退原排名。TypeSafe 官方 Python SDK 也支持异步调用，但其默认超时和重试不适合直接用于首反馈路径：[SDK 客户端](https://docs.typesafe.ai/sdk/python/api/clients/async)、[重试配置](https://docs.typesafe.ai/sdk/python/api/retries)。

## 选中后的本地处理

Jev 选择的是**模型**；具体渠道仍由路由器在该模型的合格 offer 中确定。等待 Jev 期间渠道健康、额度或订阅压力可能变化，因此返回后重新规划，并验证所选模型仍属于当前允许仲裁的集合。验证通过后，只将其最佳 offer 移到 `ranked[0]`，其余候选按原顺序保留，用于首 token 前换家和对冲；验证失败则使用新计划的原始排名。额度预留必须基于最终排名重新计算。

若正常档预留失败并转入经济档，应重新规划并使用经济档的确定性排序，不沿用正常档的 Jev 选择。验证失败后的定向重试继续使用现有规则，不再调用 Jev。另用 `(key, session_key, n_user_turns, last_user_hash)` 在本地标记本轮已尝试，超时和弃权也计入，防止每个工具步骤重新调用。这些关联标识不进入 Jev 请求或持久日志。决策缓存只复用相同请求指纹、候选快照和策略版本下的选择；每次仍重新打分和执行硬过滤。配置或 Jev 模型版本变化时使缓存失效。

## 验证与启用条件

先使用模拟 Jev 响应验证超时、无效 ID、候选变化、额度降档、区域与套餐限制、同模型跨渠道、换家、对冲和重试。随后在 `shadow` 模式异步调用 Jev，实际请求仍走原排名，记录触发率、分歧率、决策耗时分布、弃权与故障率、额外费用及候选是否仍有效。候选顺序交换后也要检查选择是否稳定。

灰度试验按会话分组，比较原路由、简单的本地权重校准和 Jev 仲裁。评估首个下游 SSE 与 Hermes 可见进度的延迟、任务完成率、纠正与重试次数、总耗时及每个完成任务的费用。现有 `ttft_ms` 取自上游首片段，灰度前须补客户端可见首反馈的计时。只有完成率或延迟在合理成本内改善，才将 `active` 从小比例扩大；“选了不同模型”本身不算收益。

请求日志与路由试算应分别记录：原始打分首选、Jev 建议、最终实际服务模型、触发理由、决策耗时和回退理由。保留现有不存提示词原文的约定。

## 当前设计的适用范围

仲裁解决的是**合格模型之间的近分选择**。如果轻量请求被误分为高难度 `reason`，或强模型因策略权重获得明显领先，触发器不会调用 Jev；这两类问题需要分别校准 [`intent.py`](../thalamus/intent.py) 与 [`policy.yaml`](../config/policy.yaml)。在现有默认配置的离线试算中，短预订请求按模型去重后，DeepSeek Flash 与下一名 Kimi 的分差较大；按初版近分门槛预计不会触发 Jev。具体门槛仍需影子评测确定。

## 落地所需配合

设计、模拟实现和本地评测不需要真实 Jev 密钥。进入真实影子试验前，需要：

1. 在部署环境安全设置 TypeSafe API key；不要把密钥贴到聊天或提交到仓库。
2. 确认允许发送给 Jev 的字段范围。首版建议只发送上述去标识结构摘要；若希望 Jev 按具体问题判断模型，另需确认短目标摘要的来源与脱敏规则。区域受限的 key 默认跳过。

影子结果准备好后，再共同确定少量具有代表性的 Hermes 流程、怎样判定任务完成，以及首反馈允许增加的上限。当前的人工分类样例不足以代替真实 GPT-Live 多步骤流程。
