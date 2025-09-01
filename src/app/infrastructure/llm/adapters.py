"""Provider adapters (OpenAI, Azure) implementing a minimal generate contract.

Includes naive structured parsing fallback using provided schema/pydantic model.
"""
from __future__ import annotations

import json
import logging
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

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class OpenAIAdapter(BaseAdapter):
    def generate(
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
        refused = False
        error_message: str | None = None
        usage: LLMUsage | None = None
        structured: dict[str, Any] | None = None
        out_text: str | None = None
        parse_fallback = False
        try:
            if pydantic_model is not None:  # preferred structured outputs path
                # responses.parse will enforce schema and return parsed model instance
                resp = client.responses.parse(  # type: ignore[attr-defined]
                    model=model,
                    input=messages,
                    instructions=instructions,
                    text_format=pydantic_model,
                    **params,
                )
                # output_text convenience (SDK provides .output_text sometimes)
                with suppress(Exception):
                    out_text = getattr(resp, "output_text", None)
                # parsed instance
                with suppress(Exception):
                    parsed_obj = getattr(resp, "output_parsed", None)
                    if parsed_obj is not None:
                        structured = json.loads(parsed_obj.model_dump_json())  # type: ignore
            else:
                # Manual JSON schema; wrap into text.format per latest spec
                kwargs: dict[str, Any] = {"model": model, "input": messages}
                if instructions:
                    kwargs["instructions"] = instructions
                if schema:
                    if "name" in schema and "schema" in schema:
                        format_schema = schema
                    else:
                        format_schema = {
                            "name": "structured_output",
                            "schema": schema,
                            "strict": True,
                        }
                    kwargs["text"] = {  # new style
                        "format": {
                            "type": "json_schema",
                            **format_schema,
                        }
                    }
                kwargs.update(params)
                resp = client.responses.create(**kwargs)  # type: ignore[arg-type]
                # Derive output text
                with suppress(Exception):
                    out_text = getattr(resp, "output_text", None)
                if out_text is None and getattr(resp, "output", None):
                    # walk first text segment
                    with suppress(Exception):
                        first = resp.output[0]
                        if first.content and first.content[0].type == "output_text":  # type: ignore
                            out_text = first.content[0].text.value  # type: ignore
                if schema and out_text:
                    with suppress(Exception):
                        cand = json.loads(out_text)
                        if isinstance(cand, dict):
                            structured = cand
            # refusal detection (Responses API: content item type 'refusal')
            with suppress(Exception):
                for item in getattr(resp, "output", []) or []:  # type: ignore
                    for seg in getattr(item, "content", []) or []:  # type: ignore
                        if getattr(seg, "type", None) == "refusal":
                            refused = True
                            out_text = getattr(seg, "refusal", None)
            # usage mapping (different attribute names possible)
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
            parse_fallback=parse_fallback,
            error_message=error_message,
        )


def _import_azure():  # lazy import
    from openai import AzureOpenAI  # type: ignore

    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )


class AzureAdapter(BaseAdapter):  # now real implementation using AzureOpenAI
    def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        instructions: str | None,
        schema: dict[str, Any] | None,
        pydantic_model: type[BaseModel] | None,
        **params: Any,
    ) -> LLMResponse:  # pragma: no cover - network call
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
        refused = False
        error_message: str | None = None
        usage: LLMUsage | None = None
        structured: dict[str, Any] | None = None
        out_text: str | None = None
        parse_fallback = False
        try:
            if pydantic_model is not None:
                # Attempt parse path (SDK parity with OpenAI)
                try:
                    resp = client.responses.parse(  # type: ignore[attr-defined]
                        model=model,
                        input=messages,
                        instructions=instructions,
                        text_format=pydantic_model,
                        **params,
                    )
                except Exception as exc:
                    # Fallback to plain create
                    logging.getLogger(__name__).debug("Azure parse fallback: %s", exc)
                    resp = None  # type: ignore
                if resp is not None:
                    with suppress(Exception):
                        out_text = getattr(resp, "output_text", None)
                    with suppress(Exception):
                        parsed_obj = getattr(resp, "output_parsed", None)
                        if parsed_obj is not None:
                            structured = json.loads(parsed_obj.model_dump_json())  # type: ignore
            if structured is None:  # build manual call
                kwargs: dict[str, Any] = {"model": model, "input": messages}
                if instructions:
                    kwargs["instructions"] = instructions
                if schema:
                    if "name" in schema and "schema" in schema:
                        format_schema = schema
                    else:
                        format_schema = {
                            "name": "structured_output",
                            "schema": schema,
                            "strict": True,
                        }
                    kwargs["text"] = {"format": {"type": "json_schema", **format_schema}}
                kwargs.update(params)
                resp = client.responses.create(**kwargs)  # type: ignore[arg-type]
                with suppress(Exception):
                    out_text = getattr(resp, "output_text", None)
                if out_text is None and getattr(resp, "output", None):
                    with suppress(Exception):
                        first = resp.output[0]
                        if first.content and first.content[0].type == "output_text":  # type: ignore
                            out_text = first.content[0].text.value  # type: ignore
                if schema and out_text:
                    with suppress(Exception):
                        cand = json.loads(out_text)
                        if isinstance(cand, dict):
                            structured = cand
            # refusal detection
            with suppress(Exception):
                for item in getattr(resp, "output", []) or []:  # type: ignore
                    for seg in getattr(item, "content", []) or []:  # type: ignore
                        if getattr(seg, "type", None) == "refusal":
                            refused = True
                            out_text = getattr(seg, "refusal", None)
            # usage mapping
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
            parse_fallback=parse_fallback,
            error_message=error_message,
        )


__all__ = ["AzureAdapter", "OpenAIAdapter"]