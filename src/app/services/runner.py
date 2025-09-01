"""High-level orchestration: end-to-end experiment execution (new architecture).

Responsibilities:
    * Resolve record selection
    * Execute per-record loop (LLM calls / dry-run)
    * Build & persist artifacts (manifest, results JSONL, per-record detail, prompt snapshot)
    * Append run summary & update analytics
    * Support chaining via injected upstream records

Pure domain models in `app.domain.models` keep concerns isolated.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from app.domain.models import PromptSnapshot, RunManifest, RunSummary
from app.infrastructure.artifacts import (
    append_results_jsonl,
    append_run_summary,
    ensure_run_dir,
    write_manifest,
    write_prompt_snapshot,
    write_record_detail,
)
from app.infrastructure.cost import estimate_run_cost
from app.infrastructure.llm.client import LLMClient
from app.infrastructure.logging import get_console, log_experiment_start
from app.services.analytics import build_analytics
from app.services.execution import run_execution
from app.services.variable_analysis import analyze_variables
from app.utils.selection import parse_selection


class RunOutput:
    def __init__(
        self,
        *,
        run_dir: Path,
        manifest: RunManifest,
        summary: RunSummary,
        results: list[Any],
    ) -> None:
        self.run_dir = run_dir
        self.manifest = manifest
        self.summary = summary
        self.results = results


def run_experiment(
    *,
    spec,
    resolved,
    client,
    output_root: Path,
    injected_records: list[dict[str, Any]] | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> RunOutput:
    console = get_console()
    log_experiment_start(name=spec.name, description=spec.description)
    dataset = injected_records if injected_records is not None else (resolved.dataset_records or [])
    selected_indices = parse_selection(spec.record_selection, len(dataset))
    if limit is not None:
        selected_indices = selected_indices[:limit]
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = ensure_run_dir(output_root / spec.name, timestamp)
    # latest pointer
    (run_dir.parent / "latest.txt").write_text(timestamp, encoding="utf-8")
    # variable analysis (skip if chaining)
    prompt_paths: list[Path] = [Path(resolved.system_prompt_path), Path(resolved.user_prompt_path)]
    if resolved.developer_prompt_path:
        prompt_paths.append(Path(resolved.developer_prompt_path))
    prompt_paths.extend(Path(p) for p in resolved.instruction_paths)
    if console is not None:
        analyze_variables(
            expected_variables=spec.expected_variables,
            prompt_paths=prompt_paths,
            records=dataset,
            injected_records=injected_records is not None,
            console=console,
        )
    # execution
    # structured schema (attached dynamically by resolver)
    schema_json = resolved.model_schema_json
    schema_model_cls = resolved.schema_model_cls
    if client is None and not dry_run:
        client = LLMClient(spec.provider)
    exec_res = run_execution(
        client=client,
        model=spec.model,
        records=dataset,
        selected_indices=selected_indices,
        system_path=Path(resolved.system_prompt_path),
        user_path=Path(resolved.user_prompt_path),
        developer_path=(
            Path(resolved.developer_prompt_path)
            if resolved.developer_prompt_path
            else None
        ),
        instruction_paths=[Path(p) for p in resolved.instruction_paths],
        dry_run=dry_run,
    # Plain-text raw request/response log (no metrics metadata)
    log_path=run_dir / "logs" / "llm_requests.log",
        model_params={"temperature": spec.temperature} if spec.temperature is not None else None,
        schema_json=schema_json,
        schema_model=schema_model_cls,
    )
    metrics = exec_res.metrics
    est_cost = (
        estimate_run_cost(spec.model, metrics.prompt_tokens, metrics.completion_tokens)
        if not dry_run
        else None
    )
    manifest = RunManifest(
        experiment_name=spec.name,
        timestamp=timestamp,
        model=spec.model,
        provider=spec.provider,
        parameters={
            "temperature": spec.temperature,
            "max_output_tokens": spec.max_output_tokens,
            "top_p": getattr(spec, "top_p", None),
            "presence_penalty": getattr(spec, "presence_penalty", None),
            "frequency_penalty": getattr(spec, "frequency_penalty", None),
            "logprobs": getattr(spec, "logprobs", None),
        },
        dataset_size=len(dataset),
        selected_count=len(selected_indices),
    )
    success_pct = metrics.success_pct
    summary = RunSummary(
    run_dir=run_dir.name,
        timestamp=timestamp,
        count=len(selected_indices),
    avg_latency_ms=metrics.avg_latency_ms,
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        total_tokens=metrics.total_tokens,
        refusals=metrics.refusals,
        parse_fallbacks=metrics.parse_fallbacks,
        success_pct=success_pct,
        est_cost_total=est_cost,
    )
    # persist artifacts
    snapshot = PromptSnapshot(
        system=Path(resolved.system_prompt_path).read_text(encoding="utf-8"),
        user_template=Path(resolved.user_prompt_path).read_text(encoding="utf-8"),
        instructions=[Path(p).read_text(encoding="utf-8") for p in resolved.instruction_paths],
        developer=(
            Path(resolved.developer_prompt_path).read_text(encoding="utf-8")
            if resolved.developer_prompt_path
            else None
        ),
    )
    write_prompt_snapshot(run_dir, snapshot)
    write_manifest(run_dir, manifest)
    append_results_jsonl(run_dir, exec_res.results)
    for r in exec_res.results:
        write_record_detail(run_dir, r)
    # run summary aggregation
    summary_path = run_dir.parent / "runs_summary.jsonl"
    append_run_summary(summary_path, summary)
    analytics_path = run_dir.parent / "analytics.json"
    # recompute analytics (simple full rebuild)
    try:
        lines = []
        if summary_path.exists():
            for raw_line in summary_path.read_text(encoding="utf-8").splitlines():
                if raw_line.strip():
                    lines.append(json.loads(raw_line))
        from app.domain.models import RunSummary as RS
        summaries = [RS(**entry) for entry in lines]
        analytics = build_analytics(summaries)
        analytics_path.write_text(json.dumps(analytics.model_dump(), indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        logging.exception("analytics rebuild failed: %s", exc)
    # per-run metrics snapshot (canonical metrics & cost; manifest stays slim)
    try:
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "run_dir": run_dir.name,
                    "timestamp": timestamp,
                    "model": spec.model,
                    "provider": spec.provider,
                    # token usage
                    "prompt_tokens": metrics.prompt_tokens,
                    "completion_tokens": metrics.completion_tokens,
                    "total_tokens": metrics.total_tokens,
                    # latency stats
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "p50_latency_ms": metrics.p50_latency_ms,
                    "p95_latency_ms": metrics.p95_latency_ms,
                    # outcome stats
                    "total_records": metrics.total_records,
                    "validated": metrics.validated,
                    "refusals": metrics.refusals,
                    "parse_fallbacks": metrics.parse_fallbacks,
                    "success_pct": metrics.success_pct,
                    # cost
                    "est_cost_total": est_cost,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).debug("metrics write failed: %s", exc)
    return RunOutput(run_dir=run_dir, manifest=manifest, summary=summary, results=exec_res.results)


def chain_run(
    *,
    names: list[str],
    resolver,
    client_factory,
    output_root: Path,
    dry_run: bool = False,
    limit: int | None = None,
) -> list[Path]:
    paths: list[Path] = []
    injected: list[dict[str, Any]] | None = None
    for name in names:
        resolved = resolver(name)
        spec = resolved.spec
        client = None if dry_run else client_factory(spec.provider)
        out = run_experiment(
            spec=spec,
            resolved=resolved,
            client=client,
            output_root=output_root,
            injected_records=injected,
            dry_run=dry_run,
            limit=limit,
        )
        paths.append(out.run_dir)
        # build injected records for next stage (use structured_output if present else text)
        # Read results JSONL produced for this run
        results_path = out.run_dir / "experiment_results.jsonl"
        new_records: list[dict[str, Any]] = []
        if results_path.exists():
            for line in results_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                import json as _json

                try:
                    rec = _json.loads(line)
                except Exception as exc:  # pragma: no cover
                    logging.getLogger(__name__).debug("bad result line: %s", exc)
                    continue
                payload = rec.get("structured_output") or {"output_text": rec.get("output_text")}
                # Merge original input; expose 'output' + 'upstream_output' for templating
                base_input = rec.get("input") or {}
                merged = {**base_input, "output": payload, "upstream_output": payload}
                new_records.append(merged)
        injected = new_records
    return paths
