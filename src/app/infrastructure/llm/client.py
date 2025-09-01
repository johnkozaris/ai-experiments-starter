"""Unified LLM client wrapping provider-specific adapters.

Initial version: skeleton only; will be wired into existing runner gradually.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.domain.models import LLMUsage

from .adapters import AzureAdapter, OpenAIAdapter


class LLMClientError(RuntimeError):
    """Base client error."""


class ProviderNotSupported(LLMClientError):
    pass


class GenerationError(LLMClientError):
    pass


@dataclass(slots=True)
class LLMResponse:
    output_text: str | None
    structured_output: dict[str, Any] | None
    usage: LLMUsage | None
    refused: bool
    latency_ms: float
    parse_fallback: bool
    error_message: str | None = None


class BaseAdapter:
    def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        instructions: str | None,
        schema: dict[str, Any] | None,
        pydantic_model: type[BaseModel] | None,
        **params: Any,
    ) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError


_ADAPTERS: dict[str, type[BaseAdapter]] = {
    "openai": OpenAIAdapter,
    "azure": AzureAdapter,
}

_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {},
    "azure": {},
}


class LLMClient:
    """Unified facade selecting adapter & enriching calls with defaults.

    Usage:
        client = LLMClient(provider)
        client.generate(model=..., messages=[...], instructions=..., schema=..., pydantic_model=...)
    """

    def __init__(self, provider: str):
        if provider not in _ADAPTERS:
            raise ProviderNotSupported(provider)
        self.provider = provider
        self._adapter = _ADAPTERS[provider]()
        self._defaults = dict(_PROVIDER_DEFAULTS.get(provider, {}))

    def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        instructions: str | None = None,
        schema: dict[str, Any] | None = None,
        pydantic_model: type[BaseModel] | None = None,
        **params: Any,
    ) -> LLMResponse:
        call_params = {**self._defaults, **params}
        try:
            return self._adapter.generate(
                model=model,
                messages=messages,
                instructions=instructions,
                schema=schema,
                pydantic_model=pydantic_model,
                **call_params,
            )
        except Exception as exc:  # pragma: no cover
            return LLMResponse(
                output_text=None,
                structured_output=None,
                usage=None,
                refused=False,
                latency_ms=0.0,
                parse_fallback=False,
                error_message=str(exc),
            )


__all__ = ["LLMClient", "LLMClientError", "LLMResponse"]
