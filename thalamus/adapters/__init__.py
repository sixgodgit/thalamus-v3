from .anthropic import AnthropicAdapter
from .base import Adapter, CallContext, ConnectionPool
from .openai_chat import OpenAIChatAdapter

ADAPTERS: dict[str, type[Adapter]] = {
    "openai_chat": OpenAIChatAdapter,
    "anthropic": AnthropicAdapter,
}


def make_adapter(offer, pool: ConnectionPool) -> Adapter:
    return ADAPTERS[offer.dialect](offer, pool)


__all__ = ["ADAPTERS", "Adapter", "CallContext", "ConnectionPool", "make_adapter"]
