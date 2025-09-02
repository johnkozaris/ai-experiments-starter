"""Execution service: iterate records, call LLM, produce RecordResult list + metrics."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from statistics import quantiles
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn

from app.domain.models import LLMUsage, RecordResult, RunMetrics
from app.infrastructure.llm.client import LLMClient
from app.services.prompt_builder import build_messages, render_instructions


class ExecutionResult:
    def __init__(self, results: list[RecordResult], metrics: RunMetrics):
        self.results = results
        self.metrics = metrics


def run_execution(
    *,
    client: LLMClient | None,
    model: str,
    records: list[dict[str, Any]],
    selected_indices: list[int],
    system_path: Path | None,
    user_path: Path,
    developer_path: Path | None,
    instruction_paths: list[Path],
    dry_run: bool,
    model_params: dict[str, Any] | None = None,
    log_path: Path | None = None,
    schema_json: dict[str, Any] | None = None,
    schema_model: Any | None = None,  # deprecated; kept for signature compatibility
    run_dir: Path | None = None,
    fail_fast: bool = True,
    throttle_ms: int | None = None,
    simulate_structured_output_on_dry_run: bool = True,
) -> ExecutionResult:
    model_params = model_params or {}
    results: list[RecordResult] = []
    usage_prompt_total = usage_completion_total = 0
    refusals = 0
    parse_fallbacks = 0
    validated = 0
    latencies: list[float] = []
    progress: Progress | None = None
    if selected_indices:
        progress = Progress(SpinnerColumn(), TextColumn("{task.description}"))
        progress.start()
        task_id = progress.add_task("executing", total=len(selected_indices))
    else:
        task_id = None

    throttle = (
        throttle_ms
        if throttle_ms is not None
        else int(os.getenv("EXPERIMENT_THROTTLE_MS", "0") or 0)
    )
    for i, idx in enumerate(selected_indices):
        if i > 0 and throttle > 0:
            time.sleep(throttle / 1000.0)
        row = records[idx]
        ctx = dict(row)
        ctx.setdefault("record", row)
        msgs = build_messages(
            system=system_path,
            user=user_path,
            developer=developer_path,
            vars=ctx,
        )
        instructions = render_instructions(instruction_paths, ctx)
        if dry_run or client is None:
            latency_ms = 0.0
            structured_sample: dict[str, Any] | None = None
            if simulate_structured_output_on_dry_run and schema_json:
                def _sample(schema: Any) -> Any:  # very small placeholder generator
                    if not isinstance(schema, dict):
                        return None
                    t = schema.get("type")
                    if t == "object":
                        props = schema.get("properties", {})
                        out: dict[str, Any] = {}
                        for k, v in props.items():
                            out[k] = _sample(v)
                        return out
                    if t == "array":
                        return []
                    if t == "string":
                        return "DRY_RUN"
                    if t == "number" or t == "integer":
                        return 0
                    if t == "boolean":
                        return False
                    if (
                        "anyOf" in schema
                        and isinstance(schema["anyOf"], list)
                        and schema["anyOf"]
                    ):
                        return _sample(schema["anyOf"][0])
                    return None

                structured_sample = _sample(schema_json)
            result = RecordResult(
                index=idx,
                input=row,
                output_text="(dry-run)",
                structured_output=structured_sample,
                validated=True,
            )
            validated += 1
        else:
            # ---------------- Raw payload intent (pre-SDK) ---------------- #
            if run_dir is not None:
                try:
                    req_dir = run_dir / "logs" / "requests"
                    req_dir.mkdir(parents=True, exist_ok=True)
                    # Build payload as the SDK call kwargs (approximate HTTP body)
                    payload: dict[str, Any] = {
                        "__intent__": "responses.create",
                        "model": model,
                        "input": msgs,
                    }
                    if instructions:
                        payload["instructions"] = instructions
                    if schema_json:
                        fmt = (
                            schema_json
                            if ("name" in schema_json and "schema" in schema_json)
                            else {"name": "ResponseSchema", "schema": schema_json}
                        )
                        payload["text"] = {"format": {"type": "json_schema", **fmt}}
                    if model_params:
                        payload["params"] = model_params
                    # Input record snapshot
                    payload["input_record_index"] = idx
                    payload["input_record"] = row
                    (req_dir / f"payload_{idx}.json").write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception as exc:  # pragma: no cover
                    logging.getLogger(__name__).debug("Failed writing raw payload: %s", exc)
            # If schema_json is a plain JSON Schema root with $defs, pass directly (adapter wraps).
            resp = client.generate(
                model=model,
                messages=msgs,
                instructions=instructions,
                schema=schema_json if schema_json else None,
                pydantic_model=None,  # no adapter fallback
                **model_params,
            )
            latency_ms = resp.latency_ms
            latencies.append(latency_ms)
            usage = resp.usage or LLMUsage()
            usage_prompt_total += usage.prompt_tokens
            usage_completion_total += usage.completion_tokens
            if resp.refused:
                refusals += 1
            if resp.parse_fallback:
                parse_fallbacks += 1
            result = RecordResult(
                index=idx,
                input=row,
                output_text=resp.output_text,
                structured_output=resp.structured_output,
                error_message=getattr(resp, "error_message", None),
                refused=resp.refused,
                validated=not resp.refused and getattr(resp, "error_message", None) is None,
                usage=usage if getattr(resp, "error_message", None) is None else None,
                latency_ms=latency_ms,
                parse_fallback=resp.parse_fallback,
            )
            if not resp.refused and getattr(resp, "error_message", None) is None:
                validated += 1
            if fail_fast and (result.error_message or result.refused):
                results.append(result)
                if progress and task_id is not None:
                    progress.stop()
                break
        results.append(result)
        if log_path is not None:
            # Write a human-readable plain text block per request capturing the raw prompt
            # (messages + instructions) and the raw output text. Avoid internal metrics.
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write("=== REQUEST START ===\n")
                    f.write(f"record_index: {idx}\n")
                    f.write(f"model: {model}\n")
                    provider = getattr(client, "provider", "openai") if client else "openai"
                    f.write(f"provider: {provider}\n")
                    for mi, m in enumerate(msgs):
                        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "?")
                        content = (
                            m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                        )
                        f.write(f"message[{mi}] ({role}):\n{content}\n")
                    if instructions:
                        f.write("--- instructions ---\n")
                        if isinstance(instructions, list):
                            for ii, inst in enumerate(instructions):
                                f.write(f"instruction[{ii}]:\n{inst}\n")
                        else:
                            f.write(str(instructions) + "\n")
                    f.write("--- output ---\n")
                    if result.output_text:
                        f.write(result.output_text + "\n")
                    if result.error_message:
                        f.write("(error) " + result.error_message + "\n")
                    if result.structured_output:
                        import json as _json
                        f.write("--- structured_output ---\n")
                        f.write(_json.dumps(result.structured_output, indent=2) + "\n")
                    if result.usage:
                        f.write("--- usage ---\n")
                        f.write(
                            f"prompt_tokens={result.usage.prompt_tokens} "
                            f"completion_tokens={result.usage.completion_tokens} "
                            f"total_tokens={result.usage.total_tokens}\n"
                        )
                    f.write("=== REQUEST END ===\n\n")
            except Exception as exc:  # pragma: no cover
                logging.getLogger(__name__).debug("Failed writing request log: %s", exc)
        if progress and task_id is not None:
            progress.advance(task_id)

    if progress:
        progress.stop()

    p50 = p95 = None
    if latencies:
        try:
            qs = quantiles(latencies, n=100)
            p50 = qs[49]
            p95 = qs[94]
        except Exception as exc:  # pragma: no cover
            logging.getLogger(__name__).debug("latency quantile calc failed: %s", exc)
    success_pct = None
    if results:
        processed = len(results) - refusals
        success_pct = (processed / len(results)) * 100
    metrics = RunMetrics(
        total_records=len(results),
        validated=validated,
        refusals=refusals,
        parse_fallbacks=parse_fallbacks,
        success_pct=success_pct,
        avg_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        prompt_tokens=usage_prompt_total,
        completion_tokens=usage_completion_total,
        total_tokens=usage_prompt_total + usage_completion_total,
        latencies_ms=latencies,
    )
    return ExecutionResult(results, metrics)
