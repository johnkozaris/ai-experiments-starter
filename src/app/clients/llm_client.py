from __future__ import annotations

import contextlib
import json as _json
import os
import time
from typing import Any

from openai import AzureOpenAI, BadRequestError, OpenAI  # type: ignore
from pydantic import BaseModel


class LLMClient:
    def __init__(self, provider: str):
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
            self.client: Any = OpenAI(api_key=api_key)
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
        """Call the Responses API with optional structured output.

        Preference:
        - If pydantic_model provided: try responses.parse (gives output_parsed + strict schema)
          retaining schema for fallback manual JSON schema mode if parse not supported.
        - Else if schema provided: send manual JSON schema.
        - Else: free-form text.
        """
        # Normalize messages to Responses input shape
        norm_input: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
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
        manual_schema = schema if pydantic_model else None
        kwargs: dict[str, Any] = {"model": model, "input": norm_input}
        if schema and not pydantic_model:
            format_block = {**schema, "type": "json_schema"}
            kwargs["text"] = {"format": format_block}
        kwargs.update(params)

        # Prepare optional request logging (JSONL). We log a sanitized snapshot of kwargs.
        def _append_log(record: dict[str, Any]):  # local helper
            if not log_path:
                return
            try:
                import json as _j
                from datetime import datetime as _dt
                rec = {"ts": _dt.utcnow().isoformat(timespec="seconds"), **record}
                # sanitize
                if "payload" in rec and isinstance(rec["payload"], dict):
                    rec["payload"].pop("api_key", None)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(_j.dumps(rec) + "\n")
            except Exception as _log_exc:  # pragma: no cover
                import logging as _l
                _l.debug("Failed writing llm request/response log: %s", _log_exc)

        # Log the exact (sanitized) request payload including normalized messages
        request_snapshot = {
            "event": "raw_request",
            "provider": self.provider,
            "model": model,
            "schema_included": bool(schema or pydantic_model),
            "request": {
                **{k: v for k, v in kwargs.items() if k != "input"},
                "input": norm_input,
            },
        }
        _append_log(request_snapshot)

        # Optional per-request Azure overrides
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
                )

        # Build canonical HTTP request representation (before sending)
        def _http_snapshot(kind: str, body: dict[str, Any]):
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
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer ****" if self.provider == "openai" else "api-key ****",
            }
            _append_log(
                {
                    "event": "http_request",
                    "attempt": kind,
                    "method": "POST",
                    "url": url,
                    "headers": headers,
                    "body": body,
                }
            )

        start = time.perf_counter()
        parse_failed = False
        response: Any | None = None
        if use_parse:
            # Log parse attempt HTTP body snapshot
            try:
                parse_body = {
                    "model": model,
                    "input": norm_input,
                    "text_format": (
                        pydantic_model.__name__ if pydantic_model else None
                    ),
                }
                _http_snapshot("parse", parse_body)
            except Exception as _parse_log_exc:  # pragma: no cover
                import logging as _logging
                _logging.debug("Failed to snapshot parse body: %s", _parse_log_exc)
            try:
                response = self.client.responses.parse(  # type: ignore[attr-defined]
                    model=model, input=norm_input, text_format=pydantic_model
                )
            except BadRequestError:
                parse_failed = True
            except Exception:  # pragma: no cover
                parse_failed = True
        if response is None:  # fallback path or normal path
            try:
                # Log create/fallback HTTP body snapshot
                try:
                    create_body = {"model": model, "input": norm_input} | (
                        {"text": kwargs.get("text")} if kwargs.get("text") else {}
                    )
                    _http_snapshot("create", create_body)
                except Exception as _create_log_exc:  # pragma: no cover
                    import logging as _logging
                    _logging.debug("Failed to snapshot create body: %s", _create_log_exc)
                if parse_failed and manual_schema:
                    format_block = {**manual_schema, "type": "json_schema"}
                    kwargs["text"] = {"format": format_block}
                response = self.client.responses.create(**kwargs)  # type: ignore
            except BadRequestError as bre:  # type: ignore
                import logging

                info: dict[str, Any] = {"message": str(bre), "type": bre.__class__.__name__}
                status = getattr(getattr(bre, "response", None), "status_code", None)
                if status is not None:
                    info["status_code"] = status
                body = None
                resp_obj = getattr(bre, "response", None)
                if resp_obj is not None:
                    for attr in ("json", "text"):
                        try:
                            if hasattr(resp_obj, attr):
                                val = getattr(resp_obj, attr)
                                body = val() if callable(val) else val
                                if body:
                                    break
                        except Exception as loop_exc:  # pragma: no cover
                            logging.debug(
                                "Failed extracting error detail attribute %s: %s",
                                attr,
                                loop_exc,
                            )
                err_field = getattr(bre, "error", None)
                if err_field and not body:
                    body = getattr(err_field, "__dict__", str(err_field))
                if body is not None:
                    info["body"] = body
                logging.error("LLM BadRequestError: %s", info)
                latency_ms = (time.perf_counter() - start) * 1000
                _append_log(
                    {
                        "event": "response",
                        "error": info,
                        "latency_ms": latency_ms,
                        "parse_fallback": parse_failed if use_parse else None,
                    }
                )
                return {"error": info, "kwargs": kwargs, "latency_ms": latency_ms}
            except Exception as e:  # pragma: no cover
                import logging

                logging.error("LLM request exception: %s", e)
                latency_ms = (time.perf_counter() - start) * 1000
                _append_log(
                    {
                        "event": "response",
                        "error": {"message": str(e)},
                        "latency_ms": latency_ms,
                        "parse_fallback": parse_failed if use_parse else None,
                    }
                )
                return {"error": {"message": str(e)}, "kwargs": kwargs, "latency_ms": latency_ms}

        # Extract outputs
        texts: list[str] = []
        refusal: str | None = None
        try:
            for item in getattr(response, "output", []):  # type: ignore[attr-defined]
                for c in getattr(item, "content", []):  # type: ignore[attr-defined]
                    c_type = getattr(c, "type", None)
                    if c_type == "output_text":
                        texts.append(getattr(c, "text", ""))
                    elif c_type == "refusal":
                        refusal = getattr(c, "refusal", None)
        except Exception as exc:  # pragma: no cover
            import logging

            logging.debug("Failed to extract text segments: %s", exc)
        full_text = "".join(texts)
        parsed_json: dict[str, Any] | None = None
        if pydantic_model and hasattr(response, "output_parsed") and not refusal:
            with contextlib.suppress(Exception):
                parsed_json = response.output_parsed.model_dump()  # type: ignore[attr-defined]
        elif (schema or manual_schema) and not refusal:
            with contextlib.suppress(Exception):
                parsed_json = _json.loads(full_text)

        latency_ms = (time.perf_counter() - start) * 1000
        usage = getattr(response, "usage", None)
        usage_dict = None
        try:
            if usage is not None:
                raw_usage = dict(usage)  # type: ignore[arg-type]
                def _jsonable(obj):  # small deep converter
                    if obj is None or isinstance(obj, str | int | float | bool):
                        return obj
                    if isinstance(obj, list | tuple):
                        return [_jsonable(x) for x in obj]
                    if isinstance(obj, dict):
                        return {k: _jsonable(v) for k, v in obj.items()}
                    # pydantic models or objects with model_dump
                    if hasattr(obj, "model_dump"):
                        with contextlib.suppress(Exception):
                            return _jsonable(obj.model_dump())  # type: ignore
                    # objects convertible to dict()
                    with contextlib.suppress(Exception):
                        as_dict = dict(obj)  # type: ignore[arg-type]
                        return {k: _jsonable(v) for k, v in as_dict.items()}
                    # fallback to string repr
                    return str(obj)
                usage_dict = _jsonable(raw_usage)
        except Exception:  # pragma: no cover
            usage_dict = None

        log_payload = {
            "event": "llm_response",
            "provider": self.provider,
            "model": model,
            "messages": len(messages),
            "input_chars": sum(len(str(m.get("content", ""))) for m in messages),
            "schema": bool(schema or manual_schema or pydantic_model),
            "latency_ms": round(latency_ms, 2),
            "usage": usage_dict,
            "response_text_preview": full_text[:400],
            "refusal": bool(refusal),
            "error": None,
        }
        with contextlib.suppress(Exception):
            import logging as _logging

            _logging.getLogger("llm").debug(_json.dumps(log_payload))

        # Attempt to serialize raw response object for logging (best-effort)
        raw_response_serialized: Any = None
        with contextlib.suppress(Exception):
            if hasattr(response, "model_dump"):
                raw_response_serialized = response.model_dump()  # type: ignore[attr-defined]
            elif hasattr(response, "to_dict"):
                raw_response_serialized = response.to_dict()  # type: ignore[attr-defined]
            else:  # fallback repr
                raw_response_serialized = str(response)
        _append_log(
            {
                "event": "raw_response",
                "latency_ms": latency_ms,
                "usage": usage_dict,
                "refusal": refusal,
                "parsed": bool(parsed_json),
                "parse_fallback": parse_failed if use_parse else None,
                "response": raw_response_serialized,
            }
        )

        return {
            "raw": response,
            "text": full_text,
            "json": parsed_json,
            "usage": usage_dict,
            "kwargs": kwargs,
            "latency_ms": latency_ms,
            "refusal": refusal,
            "parse_fallback": parse_failed if use_parse else None,
        }


def get_client(provider: str) -> LLMClient:
    return LLMClient(provider)
