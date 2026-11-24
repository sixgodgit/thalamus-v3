# Thalamus v3

面向 Hermes 等 OpenAI 兼容 agent 的共享模型网关。上游是三类渠道：

| 渠道 | 计费方式 | 在网关里的角色 |
|---|---|---|
| DeepSeek 官方直充 | 按量付费，峰时 ×2 | **地基**：永远可用的兜底，便宜、快、支持图片 |
| GLM Coding Plan | 订阅，5 小时 + 每周积分窗口 | **闲时免费算力**：额度宽裕时承接代码 / agent |
| OpenCode Go | 订阅，5 小时 / 周 / 月额度 | **闲时免费算力**：Kimi Code、GLM-5.3、MiniMax 等强模型 |

管理台可以分发 key、设置每日额度，还能实时查看路由决策、渠道健康和订阅窗口用量。

设计思路见 [docs/DESIGN.md](docs/DESIGN.md)。可选的 Jev 打分后仲裁方案见 [docs/JEV_ARBITRATION.md](docs/JEV_ARBITRATION.md)，目前仅为设计，尚未接入。**上线前请先读主设计文档中的"风险"一节。**

---

## 快速开始（本地）

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env              # 填上游 key 和 THALAMUS_ADMIN_TOKEN
python scripts/probe.py --models  # 核对各渠道真实模型名
python scripts/probe.py --tools --cache   # 核对流式、usage、工具调用、缓存命中（会产生少量费用）
python main.py                    # http://127.0.0.1:9880
```

打开 `http://127.0.0.1:9880/admin`，用 `THALAMUS_ADMIN_TOKEN` 登录，然后新建 key。

## 客户端接入

```
OPENAI_BASE_URL = https://你的域名/v1
OPENAI_API_KEY  = sk-thl-xxxx
MODEL           = thalamus/auto
```

| model 字段 | 含义 |
|---|---|
| `thalamus/auto` | 自动判断任务与难度（推荐） |
| `thalamus/fast` | 更看重速度 |
| `thalamus/pro` | 按最高难度挑模型 |
| `thalamus/code`、`thalamus/reason-3` … | 指定任务（可带难度 1-3） |
| `glm-5.3`、`kimi-k2.7-code` … | 指定具体模型，网关只在提供它的渠道间选择与容错 |

可选请求头：`X-Session-Id`（会话粘性，强烈建议带上，缓存命中更高）、`X-Thalamus-Task`（指定任务）。

每个响应都带 `X-Thalamus-Model`（实际服务的模型）、`X-Thalamus-Task`、`X-Thalamus-Attempts`、`X-Thalamus-Quota-Mode`。
响应头里不暴露具体用的是哪个渠道。

### Hermes / Conduit 的自动分类与响应速度

`thalamus/auto` 根据最新的用户消息、图片、本轮工具调用、`reasoning_effort` 和输入长度分类，不额外调用模型：

| 请求 | 任务 |
|---|---|
| 短问答 | `chat` |
| 翻译或摘要 | `utility` |
| 分析权衡 | `reason` |
| 代码或报错 | `code` |
| 本轮看图、追问历史图片 | `vision` |
| 明确要求用工具执行预订、购买或付款 | `agent` |
| 输入超过长上下文阈值 | `long` |

一句话要求预订停车场仍是多步骤任务。旧轮次的工具调用或图片不会单独把新问题判成 `agent`/`vision`，但历史图片仍要求上游具备看图能力。

排查慢请求时，先看管理台「请求日志」里的任务、模型、首 token、总耗时和尝试次数；「路由试算」可用脱敏后的消息查看分类和候选排名，不会调用上游。若 Hermes 固定传 `reasoning_effort: high`，短问答也会升到难度 3。若每一步都要等整段回答结束才出现，检查 Hermes 发给网关的请求是否带 `stream: true`，以及反向代理是否关闭 SSE 缓冲。

语音中的工作进度由 Hermes / Conduit 负责反馈。网关只处理模型请求，不知道停车场网站或支付工具当前走到哪一步。

## 配置

- `config/upstreams.yaml`：模型能力与质量先验，以及每个渠道提供哪些模型、价格、订阅窗口
- `config/policy.yaml`：key 默认额度、订阅"影子价格"曲线、任务价值表、熔断、学习

改完在管理台"学习与配置"页点**重载配置**即可，不需要重启。配置有错时不会生效。

**加新渠道不用改文件**：控制台"渠道"页 → ＋ 添加渠道 → 填 Base URL 和 key → 点"探测可用模型" → 勾选要启用的模型并填价格 → 保存即热生效。渠道落在 `config/channels.d/<id>.yaml`（一渠道一文件，可手工编辑），key 存 `data/secrets.json`（0600，不进备份）。每个模型旁边的"测试"按钮会真发一条最小请求，回显 usage 字段与错误判定 —— 上线前核对上游行为就靠它。订阅渠道（按积分/百分比计额度）仍需手写 YAML。详见 `docs/DESIGN.md` 12.1。

两个会直接改变"能用哪些上游"的开关，改之前先看启动日志里的 warning：

- `global.min_context_tokens`（默认 `1000000`）：要求原生上下文 ≥ 1M；默认只对 `chat` / `utility` 自动路由豁免此门槛，所以短请求仍可选 MiniMax M3（200k），并可能走到 anthropic 适配器。点名模型时不豁免。详见 `docs/DESIGN.md` 1.2。
- `subscription_pacing.single_share`（默认 `0.15`）：一次请求最多占掉订阅窗口的比例。1M 上下文下一次请求就可能吃掉 5 小时窗口的四成，超过这个比例会被按平方抬价，自然溢出到按量渠道。
- `keys.allowed_regions`（默认留空 = 不限）：新建 key 的区域白名单；每把 key 也可在管理台单独设置，设置后是硬约束。

**上线前必须改的地方：**
1. `glm_plan.budget.windows`：按你购买的套餐档位填积分上限（Lite / Pro / Max）
2. `opencode_go.budget.scope`：在 Go 控制台确认额度是所有模型共用（`shared`），还是按模型分开计（`per_model`）
3. 各 `upstream` 模型名：以 `probe.py --models` 的输出为准
4. `global.daily_metered_usd_cap`：按量渠道每天最多花多少钱

## 命令行管理 key

```bash
python scripts/manage.py create --name "张三" --daily 800
python scripts/manage.py list
python scripts/manage.py set k_xxx --daily 1200 --plan pro
python scripts/manage.py disable k_xxx
```

## 部署（Linux VPS）

```bash
sudo useradd -r -s /usr/sbin/nologin thalamus
sudo mkdir -p /opt/thalamus && sudo chown thalamus /opt/thalamus
# 拷贝代码到 /opt/thalamus，创建 .venv 并安装依赖，填 .env
sudo cp deploy/thalamus.service /etc/systemd/system/
sudo systemctl enable --now thalamus
```

前面用 Caddy 做 HTTPS，配置见 `deploy/Caddyfile`。**SSE 不能被缓冲**；管理台只对你自己的 IP 开放。

数据在 `data/thalamus.db`（SQLite WAL）。每天自动备份到 `data/backups/`，保留 7 份。

## 监控

`GET /metrics`（Prometheus 格式），需要 `Authorization: Bearer <THALAMUS_METRICS_TOKEN>`；没有设置这个变量时使用管理口令。包括以下指标：
- 请求数（按状态、任务、模型分）
- credits 消耗、按量实付金额、订阅额度消耗、token 数
- 质量信号计数
- 首 token 延迟和网关路由开销的直方图
- 在途请求数、熔断状态、订阅窗口用量比例

## 评测、校准与压测

```bash
python scripts/eval_routing.py --min 0.9            # 路由分类离线评测（免费，秒级），低于门槛时退出码为 1
python scripts/calibrate.py --judge deepseek-v4-pro@deepseek   # 用真实上游给各模型打分，给出 quality 建议值（会产生费用）
python scripts/loadtest.py --levels 1,10,20 --warm  # 真实网关进程 + 模拟上游的压测
```

- 路由评测集在 `eval/routing_cases.jsonl`，覆盖中文、英文、荷兰语和结构信号。它是人工编写的种子数据，上线后请用真实请求（脱敏并取得用户同意）来扩充。
- 质量校准的题目在 `eval/quality_tasks.jsonl`。校准结果不会自动写入配置，需要人工抽查低分样本后再采纳。
- 压测会分别统计网关、模拟上游、客户端三个进程的 CPU 占用。客户端或上游的 CPU 超过 80% 时，会标注"该档数字不可信"。

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest
```

CI 配置在 `.github/workflows/tests.yml`，包含三步：测试、路由评测门槛、短时冒烟压测。

测试全部用模拟上游，覆盖以下场景：
- 分类黄金集（中文、英文、荷兰语），以及超长输入下的分类性能
- 路由打分与订阅节奏
- 并发预留不超额，内存账本批量写回后重启能读回
- OpenAI 和 Anthropic 两种格式的工具调用与图片转换
- 首 token 前换家、首 token 超时、对冲请求、流式中途失败
- 客户端断开后仍然结算
- 结果信号：工具参数非法、JSON 不合规、空回复、被纠正、被重问
- 防白嫖：不带 key 碰不到上游，超大输出会被按额度截断
- 硬边界：key 的区域白名单（`allowed_regions`）与 1M 上下文门槛（`min_context_tokens`）在换家、对冲、降档、点名模型任何路径上都不能越过
- 控制台加渠道：探测模型、勾选启用、热生效、换 key、停用删除，以及非法输入不留坏配置
- 长上下文：近 1M 输入仍给输出留空间，首 token 预算随输入长度放大
- 旧数据库自动迁移，`/metrics` 需要鉴权

改分类规则之后，务必跑 `tests/test_intent.py` 和 `scripts/eval_routing.py`；改任务价值表之后，务必跑 `tests/test_router.py`。

## 目录

```
main.py                 启动入口（读取 .env）
thalamus/
  features.py           请求特征提取（纯字符串，<2ms）
  intent.py             任务 + 难度判定，会话沿用
  router.py             候选打分：质量价值 − 成本 − 等待 − 风险
  capacity.py           订阅滚动窗口 + 影子价格
  health.py             熔断、延迟统计、会话记忆、隐式反馈学习
  outcome.py            结果信号（工具参数/JSON 合法性、空回复、截断），只存标记
  engine.py             编排：配额 → 路由 → 预留 → 首 token 前容错/对冲 → 流式 → 结算
  quota.py              每日额度（内存账本）、RPM、并发
  persist.py            后台批量写库（独立连接，每 250ms 一个事务）
  metrics.py            Prometheus 指标
  evaluation.py         路由评测与质量校准的纯函数
  store.py              SQLite（不存任何对话内容），旧库自动加列迁移
  adapters/             OpenAI Chat 透传 / Anthropic Messages 双向转换
  api_v1.py             /v1/chat/completions、/v1/models
  api_admin.py          管理 API
static/admin.html       管理台
scripts/probe.py        上线前核对真实上游
scripts/manage.py       命令行管理 key
scripts/eval_routing.py 路由分类评测（CI 门槛）
scripts/calibrate.py    模型质量校准（裁判模型打分）
scripts/loadtest.py     压测（网关子进程 + 模拟上游子进程）
eval/                   评测集与校准题目
deploy/                 systemd + Caddy
```
