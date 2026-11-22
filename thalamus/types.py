"""网关内部共用类型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureClass(str, Enum):
    TIMEOUT = "timeout"
    FIRST_TOKEN_TIMEOUT = "first_token_timeout"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    CONTEXT_OVERFLOW = "context_overflow"
    PROVIDER_DOWN = "provider_down"
    INVALID_RESPONSE = "invalid_response"
    AUTH_FAILED = "auth_failed"
    CLIENT_ERROR = "client_error"
    # 请求本身没问题，是"这个模型吃不下它"：换一个模型就能成。
    # 典型是思考模型要求回传 reasoning_content —— 400，但换非思考模型立刻就对了。
    MODEL_INCOMPATIBLE = "model_incompatible"


# 换一家就可能成功的失败。CLIENT_ERROR（请求本身有问题）换家也没用。
RETRYABLE = frozenset(FailureClass) - {FailureClass.CLIENT_ERROR}

# 这些失败说明整个渠道有问题，而不只是某个模型
CHANNEL_WIDE = frozenset(
    {FailureClass.AUTH_FAILED, FailureClass.QUOTA_EXHAUSTED}
)


class UpstreamError(Exception):
    def __init__(
        self,
        failure: FailureClass,
        message: str,
        status: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.failure = failure
        self.message = message
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.failure in RETRYABLE


class GatewayError(Exception):
    """返回给客户端的错误（OpenAI 错误结构）。"""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}

    def body(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": self.code, "code": self.code}}


@dataclass
class Usage:
    input_miss: int = 0
    input_cached: int = 0
    output: int = 0
    reasoning: int = 0

    @property
    def input_total(self) -> int:
        return self.input_miss + self.input_cached

    @property
    def empty(self) -> bool:
        return self.input_total == 0 and self.output == 0

    def merge(self, other: "Usage") -> "Usage":
        # 流式里 usage 可能分多帧给出（Anthropic 开头给输入、结尾给输出），逐字段取最大值
        return Usage(
            input_miss=max(self.input_miss, other.input_miss),
            input_cached=max(self.input_cached, other.input_cached),
            output=max(self.output, other.output),
            reasoning=max(self.reasoning, other.reasoning),
        )


@dataclass
class Delta:
    """适配器输出的归一化流式片段（OpenAI chat.completion.chunk 的 delta 语义）。"""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def meaningful(self) -> bool:
        """是否已经产生了会发给客户端的输出（首 token 判定用）。"""
        return bool(self.content or self.reasoning or self.tool_calls)
