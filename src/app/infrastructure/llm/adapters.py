"""Provider adapters (OpenAI, Azure) implementing a minimal generate contract.

Schema transformation (adding required, additionalProperties: false, nullables) plus
wrapping into the OpenAI Responses API shape happen upstream (runner). Adapters only
forward an already wrapped schema via ``response_format``.
"""
from __future__ import annotations

import json
import os
from contextlib import suppress
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from app.domain.models import LLMUsage


class LLMResponse:  # lightweight duplication to avoid circular import
    def __init__(
        self,
        *,
        output_text: str | None,
        structured_output: dict[str, Any] | None,
        usage: LLMUsage | None,
        refused: bool,
        latency_ms: float,
        parse_fallback: bool,
    error_message: str | None = None,
    ) -> None:
        self.output_text = output_text
        self.structured_output = structured_output
        self.usage = usage
        self.refused = refused
        self.latency_ms = latency_ms
        self.parse_fallback = parse_fallback
        self.error_message = error_message


class BaseAdapter:  # protocol-like; real one lives in client but avoids circular import
    def generate(  # pragma: no cover - interface only
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        instructions: str | None,
        schema: dict[str, Any] | None,
        pydantic_model: type[BaseModel] | None,
        **params: Any,
    ) -> LLMResponse:
        raise NotImplementedError


def _import_openai():  # lazy import
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=0)


class OpenAIAdapter(BaseAdapter):
    def generate(  # pragma: no cover - network
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        instructions: str | None,
        schema: dict[str, Any] | None,
        pydantic_model: type[BaseModel] | None,
        **params: Any,
    ) -> LLMResponse:
        client = _import_openai()
        start = perf_counter()
        error_message: str | None = None
        usage: LLMUsage | None = None
        structured: dict[str, Any] | None = None
        out_text: str | None = None
        refused = False
        try:
            kwargs: dict[str, Any] = {"model": model, "input": messages}
            if instructions:
                kwargs["instructions"] = instructions
            if schema is not None:
                if not ("name" in schema and "schema" in schema):
                    schema = {"name": "ResponseSchema", "schema": schema}
                # Use legacy text.format path for broader compatibility
                kwargs["text"] = {"format": {"type": "json_schema", **schema}}
            kwargs.update(params)
            resp = client.responses.create(**kwargs)  # type: ignore[arg-type]
            with suppress(Exception):
                out_text = getattr(resp, "output_text", None)
            if out_text is None:
                with suppress(Exception):
                    parts: list[str] = []
                    for item in getattr(resp, "output", []) or []:  # type: ignore
                        for seg in getattr(item, "content", []) or []:  # type: ignore
                            if getattr(seg, "type", None) == "output_text":
                                txt = getattr(getattr(seg, "text", None), "value", None)
                                if txt:
                                    parts.append(txt)
                    if parts:
                        out_text = "".join(parts)
            if schema and out_text:
                with suppress(Exception):
                    val = json.loads(out_text)
                    if isinstance(val, dict):
                        structured = val
            with suppress(Exception):
                for item in getattr(resp, "output", []) or []:  # type: ignore
                    for seg in getattr(item, "content", []) or []:  # type: ignore
                        if getattr(seg, "type", None) == "refusal":
                            refused = True
                            out_text = getattr(seg, "refusal", None)
                            break
            u = getattr(resp, "usage", None)
            if u is not None:
                prompt_tokens = (
                    getattr(u, "prompt_tokens", None)
                    or getattr(u, "input_tokens", 0)
                    or 0
                )
                completion_tokens = (
                    getattr(u, "completion_tokens", None)
                    or getattr(u, "output_tokens", 0)
                    or 0
                )
                total_tokens = getattr(u, "total_tokens", 0) or (
                    prompt_tokens + completion_tokens
                )
                usage = LLMUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
        except Exception as exc:  # pragma: no cover
            error_message = str(exc)
        latency_ms = (perf_counter() - start) * 1000
        return LLMResponse(
            output_text=out_text,
            structured_output=structured,
            usage=usage,
            refused=refused,
            latency_ms=latency_ms,
            parse_fallback=False,
            error_message=error_message,
        )


def _import_azure():  # lazy import
    from openai import AzureOpenAI  # type: ignore

    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        max_retries=0,
    )


class AzureAdapter(BaseAdapter):  # real implementation using AzureOpenAI
    def generate(  # pragma: no cover - network
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        instructions: str | None,
        schema: dict[str, Any] | None,
        pydantic_model: type[BaseModel] | None,
        **params: Any,
    ) -> LLMResponse:
        if not (os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")):
            return LLMResponse(
                output_text=None,
                structured_output=None,
                usage=None,
                refused=False,
                latency_ms=0.0,
                parse_fallback=False,
                error_message="missing Azure OpenAI credentials",
            )
        client = _import_azure()
        start = perf_counter()
        error_message: str | None = None
        usage: LLMUsage | None = None
        structured: dict[str, Any] | None = None
        out_text: str | None = None
        refused = False
        try:
            kwargs: dict[str, Any] = {"model": model, "input": messages}
            if instructions:
                kwargs["instructions"] = instructions
            if schema is not None:
                if not ("name" in schema and "schema" in schema):
                    schema = {"name": "ResponseSchema", "schema": schema}
                kwargs["text"] = {"format": {"type": "json_schema", **schema}}
            kwargs.update(params)
            resp = client.responses.create(**kwargs)  # type: ignore[arg-type]
            with suppress(Exception):
                out_text = getattr(resp, "output_text", None)
            if out_text is None:
                with suppress(Exception):
                    parts: list[str] = []
                    for item in getattr(resp, "output", []) or []:  # type: ignore
                        for seg in getattr(item, "content", []) or []:  # type: ignore
                            if getattr(seg, "type", None) == "output_text":
                                txt = getattr(getattr(seg, "text", None), "value", None)
                                if txt:
                                    parts.append(txt)
                    if parts:
                        out_text = "".join(parts)
            if schema and out_text:
                with suppress(Exception):
                    val = json.loads(out_text)
                    if isinstance(val, dict):
                        structured = val
            u = getattr(resp, "usage", None)
            if u is not None:
                prompt_tokens = (
                    getattr(u, "prompt_tokens", None)
                    or getattr(u, "input_tokens", 0)
                    or 0
                )
                completion_tokens = (
                    getattr(u, "completion_tokens", None)
                    or getattr(u, "output_tokens", 0)
                    or 0
                )
                total_tokens = getattr(u, "total_tokens", 0) or (
                    prompt_tokens + completion_tokens
                )
                usage = LLMUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            with suppress(Exception):
                for item in getattr(resp, "output", []) or []:  # type: ignore
                    for seg in getattr(item, "content", []) or []:  # type: ignore
                        if getattr(seg, "type", None) == "refusal":
                            refused = True
                            out_text = getattr(seg, "refusal", None)
                            break
        except Exception as exc:  # pragma: no cover
            error_message = str(exc)
        latency_ms = (perf_counter() - start) * 1000
        return LLMResponse(
            output_text=out_text,
            structured_output=structured,
            usage=usage,
            refused=refused,
            latency_ms=latency_ms,
            parse_fallback=False,
            error_message=error_message,
        )


__all__ = ["AzureAdapter", "OpenAIAdapter"]