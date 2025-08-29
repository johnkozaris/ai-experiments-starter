from __future__ import annotations

import contextlib
import json as _json
import os
import time
from typing import Any

from openai import AzureOpenAI, BadRequestError, OpenAI  # type: ignore
from pydantic import BaseModel


class LLMClient:
    def __init__(self, provider: str) -> None:
        desired = (provider or os.getenv("DEFAULT_PROVIDER", "openai")).lower()
        if desired not in {"openai", "azure"}:
            raise ValueError("provider must be 'openai' or 'azure'")
        if desired == "openai" and not os.getenv("OPENAI_API_KEY"):
            az_key = os.getenv("AZURE_OPENAI_API_KEY")
            az_ep = os.getenv("AZURE_OPENAI_ENDPOINT")
            if az_key and az_ep:
                import logging as _logging
                _logging.info(
                    "OPENAI_API_KEY missing; falling back to Azure (credentials detected)"
                )
                desired = "azure"
        self.provider = desired
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self.client: Any = OpenAI(api_key=api_key, max_retries=0)
        else:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
            if not (api_key and endpoint):
                raise ValueError("Azure OpenAI credentials not set")
            self.azure_endpoint = endpoint.rstrip("/")
            self.azure_api_version = api_version
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.azure_api_version,
                max_retries=0,
            )

    def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        pydantic_model: type[BaseModel] | None = None,
        log_path: str | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Call Responses API (OpenAI/Azure) with optional structured parsing."""
        norm_input: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            content_list: list[dict[str, Any]]
            if isinstance(content, list):
                parts: list[dict[str, Any]] = []
                for part in content:
                    if part.get("type") == "text":
                        part = {**part, "type": "input_text"}
                    parts.append(part)
                content_list = parts
            else:
                content_list = [{"type": "input_text", "text": str(content)}]
            norm_input.append({"role": role, "content": content_list})
        use_parse = bool(pydantic_model)
        # Always perform exactly one network call (no fallback / retry logic)
        manual_schema = schema if (schema and not pydantic_model) else None
        kwargs: dict[str, Any] = {"model": model, "input": norm_input}
        if manual_schema:
            kwargs["text"] = {"format": {**manual_schema, "type": "json_schema"}}
        kwargs.update(params)

        def _append_log(record: dict[str, Any]) -> None:
            if not log_path:
                return
            try:
                import json as _j
                from datetime import datetime as _dt
                rec = {"ts": _dt.utcnow().isoformat(timespec="seconds"), **record}
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(_j.dumps(rec) + "\n")
            except Exception as _exc:  # pragma: no cover
                import logging as _l
                _l.debug("HTTP snapshot logging failed: %s", _exc)

        def _http_snapshot(kind: str, body: dict[str, Any]) -> None:
            try:
                if self.provider == "azure":
                    _endpoint = getattr(
                        self,
                        "azure_endpoint",
                        os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                    ).rstrip("/")
                    _version = getattr(
                        self,
                        "azure_api_version",
                        os.getenv("AZURE_OPENAI_API_VERSION", ""),
                    )
                    url = f"{_endpoint}/openai/responses?api-version={_version}"
                else:
                    url = "https://api.openai.com/v1/responses"
                _append_log(
                    {
                        "event": "http_request",
                        "attempt": kind,
                        "url": url,
                        "body": body,
                    }
                )
            except Exception as _exc:  # pragma: no cover
                import logging as _l
                _l.debug("Parse attempt failed (non-fatal): %s", _exc)

        _append_log(
            {
                "event": "raw_request",
                "provider": self.provider,
                "model": model,
                "schema_included": bool(schema or pydantic_model),
            }
        )
        azure_version_override = params.pop("azure_api_version", None)
        azure_endpoint_override = params.pop("azure_endpoint", None)
        if (azure_version_override or azure_endpoint_override) and self.provider == "azure":
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = (
                azure_endpoint_override or os.getenv("AZURE_OPENAI_ENDPOINT")
            ) or self.azure_endpoint
            version = (
                azure_version_override
                or os.getenv("AZURE_OPENAI_API_VERSION")
                or getattr(self, "azure_api_version", "2025-04-01-preview")
            )
            if api_key and endpoint:
                self.azure_endpoint = endpoint.rstrip("/")
                self.azure_api_version = version
                self.client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.azure_api_version,
                    max_retries=0,
                )
        start = time.perf_counter()
        response: Any | None = None
        try:
            if use_parse:
                _http_snapshot("parse", {"model": model})
                response = self.client.responses.parse(  # type: ignore[attr-defined]
                    model=model, input=norm_input, text_format=pydantic_model
                )
            else:
                _http_snapshot("create", {"model": model})
                response = self.client.responses.create(**kwargs)  # type: ignore[attr-defined]
        except BadRequestError as bre:  # type: ignore
            latency_ms = (time.perf_counter() - start) * 1000
            _append_log({"event": "response", "error": str(bre), "latency_ms": latency_ms})
            return {"error": {"message": str(bre)}, "kwargs": kwargs, "latency_ms": latency_ms}
        except Exception as e:  # pragma: no cover
            latency_ms = (time.perf_counter() - start) * 1000
            _append_log({"event": "response", "error": str(e), "latency_ms": latency_ms})
            return {"error": {"message": str(e)}, "kwargs": kwargs, "latency_ms": latency_ms}
        texts: list[str] = []
        refusal: str | None = None
        try:
            for item in getattr(response, "output", []):  # type: ignore[attr-defined]
                for c in getattr(item, "content", []):  # type: ignore[attr-defined]
                    if getattr(c, "type", None) == "output_text":
                        texts.append(getattr(c, "text", ""))
                    elif getattr(c, "type", None) == "refusal":
                        refusal = getattr(c, "refusal", None)
        except Exception as _exc:  # pragma: no cover
            import logging as _l
            _l.debug("Failed extracting segments: %s", _exc)
        full_text = "".join(texts)
        parsed_json: dict[str, Any] | None = None
        if pydantic_model and hasattr(response, "output_parsed") and not refusal:
            with contextlib.suppress(Exception):
                parsed_json = response.output_parsed.model_dump()  # type: ignore[attr-defined]
        elif manual_schema and not refusal:
            with contextlib.suppress(Exception):
                parsed_json = _json.loads(full_text)
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "raw": response,
            "text": full_text,
            "json": parsed_json,
            "latency_ms": latency_ms,
            "usage": getattr(response, "usage", None),
            "refusal": refusal,
            "parse_fallback": None,
            "kwargs": kwargs,
        }


def get_client(provider: str) -> LLMClient:
    return LLMClient(provider)
