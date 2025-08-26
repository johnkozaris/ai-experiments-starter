"""Execution & chaining logic for experiments."""

from __future__ import annotations

import datetime as dt
import json
import statistics
from importlib import import_module
from pathlib import Path
from typing import Any

from jinja2 import Template
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from app.clients.llm_client import get_client
from app.utils.io import ensure_run_dir, read_jsonl
from app.utils.schema import import_model, model_json_schema

from .definitions import DependencySpec, ResolvedExperiment
from .loader import load_all_experiments, resolve_experiment


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return list(read_jsonl(path))
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return list(data)
        return [data]
    raise ValueError("Dataset must be .json or .jsonl")


def _select_path(obj: dict[str, Any], path: str) -> Any:
    cur: Any = obj
    if path in ("", "."):
        return cur
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _import_transform(path: str):  # returns callable
    mod_name, fn_name = path.split(":", 1)
    mod = import_module(mod_name)
    fn = getattr(mod, fn_name)
    return fn


def _records_from_dependency(
    dep: DependencySpec, output_root: Path
) -> tuple[list[dict[str, Any]], str]:
    dep_output_dir = output_root / dep.experiment
    if not dep_output_dir.exists():
        raise FileNotFoundError(f"Dependency output dir not found: {dep_output_dir}")
    # pick latest timestamp folder
    candidates = [p for p in dep_output_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError(f"No runs found for dependency {dep.experiment}")
    run_dir = sorted(candidates)[-1]
    # Support new single JSON artifact first (result.json), fallback to legacy results.jsonl
    json_result_path = run_dir / "result.json"
    base_records: list[dict[str, Any]]
    if json_result_path.exists():
        try:
            data = json.loads(json_result_path.read_text(encoding="utf-8"))
            base_records = list(data.get("records", [])) if isinstance(data, dict) else []
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed reading result.json for dependency {run_dir}: {exc}"
            ) from exc
    else:
        results_path = run_dir / "results.jsonl"
        if not results_path.exists():
            raise FileNotFoundError(
                f"result.json/results.jsonl missing for dependency run {run_dir}"
            )
        base_records = list(read_jsonl(results_path))
    derived: list[dict[str, Any]] = []
    for rec in base_records:
        sel = _select_path(rec, dep.select)
        if sel is None:
            continue
        if isinstance(sel, dict):
            derived.append(sel)
        elif isinstance(sel, list):
            for v in sel:
                if isinstance(v, dict):
                    derived.append(v)
                else:
                    derived.append({"value": v})
        else:
            derived.append({"value": sel})
    if dep.transform:
        fn = _import_transform(dep.transform)
        derived = fn(derived, dep.transform_config)  # type: ignore[arg-type]
    return derived, run_dir.name


def _build_messages(
    system_prompt: Path,
    user_prompt: Path,
    instructions: list[Path],
    vars: dict[str, Any],
) -> list[dict[str, str]]:
    system_msg = system_prompt.read_text(encoding="utf-8").strip()
    tmpl = Template(user_prompt.read_text(encoding="utf-8"))
    user_body = tmpl.render(**vars).strip()
    if instructions:
        ins_text = "\n\n".join(p.read_text(encoding="utf-8").strip() for p in instructions)
        if ins_text:
            user_body = f"{user_body}\n\n{ins_text}".strip()
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_body},
    ]


def _prepare_run_dir(base: Path, experiment_name: str) -> tuple[Path, str]:
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = base / experiment_name
    run_dir = ensure_run_dir(base_dir, ts)
    # update latest pointer (simple text file for portability)
    (base_dir / "latest.txt").write_text(ts, encoding="utf-8")
    return run_dir, ts


def run_resolved(
    ex: ResolvedExperiment,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    output_root: Path = Path("src/output"),
) -> Path:
    console = Console()
    console.print(
        Panel.fit(
            f"[bold cyan]Experiment:[/bold cyan] {ex.spec.name}\n"
            f"[dim]{ex.spec.description or ''}[/dim]",
            title="run",
            border_style="cyan",
        )
    )
    meta = Table(show_header=False, box=None)
    meta.add_row("Model", ex.spec.model)
    meta.add_row("Provider", ex.spec.provider)
    meta.add_row("Schema", ex.spec.schema_model or "-")
    meta.add_row("Dataset", str(ex.dataset_path) if ex.dataset_path else "(dependency outputs)")
    if limit is not None:
        meta.add_row("Limit", str(limit))
    meta.add_row("Dry Run", str(dry_run))
    console.print(meta)
    # Build dataset (local or from dependencies)
    records: list[dict[str, Any]] = []
    dependency_run_refs: dict[str, str] = {}
    if ex.spec.depends_on:
        for dep in ex.spec.depends_on:
            dep_records, run_id = _records_from_dependency(dep, output_root)
            dependency_run_refs[dep.experiment] = run_id
            records.extend(dep_records)
    if ex.dataset_path:
        records = records if records else _load_dataset(ex.dataset_path)
    if limit is not None:
        records = records[:limit]
    # Schema handling
    model_cls = None
    schema = None
    if ex.spec.schema_model:
        model_cls = import_model(ex.spec.schema_model)
        schema = model_json_schema(model_cls)
    client = None if dry_run else get_client(ex.spec.provider)
    run_dir, ts = _prepare_run_dir(output_root, ex.spec.name)
    # Per-run LLM request/response log (under logs/)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    llm_log_path = logs_dir / "llm_requests.jsonl"
    # snapshot prompts into single json artifact
    prompts_used = {
        "system": ex.system_prompt_path.read_text(encoding="utf-8"),
        "user_template": ex.user_prompt_path.read_text(encoding="utf-8"),
        "instructions": [
            p.read_text(encoding="utf-8") for p in (ex.instruction_paths or [])
        ],
    }
    (run_dir / "prompts_used.json").write_text(
        json.dumps(prompts_used, indent=2), encoding="utf-8"
    )
    # process
    aggregated: list[dict[str, Any]] = []  # per-record outputs accumulated in memory
    token_prompt_total = 0
    token_completion_total = 0
    token_total = 0
    latencies: list[float] = []
    progress: Progress | None = None
    if records:
        progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"))
        task_id = progress.add_task(f"{ex.spec.name}", total=len(records))
        progress.start()
    for _idx, rec in enumerate(records, start=1):
        messages = _build_messages(
            ex.system_prompt_path,
            ex.user_prompt_path,
            ex.instruction_paths,
            rec,
        )
        gen: dict[str, Any] | None = None
        try:
            if dry_run:
                attempt_out = model_cls().model_dump() if model_cls else {"text": ""}
            else:
                if schema and model_cls:
                    gen = client.generate(
                        messages=messages,
                        model=ex.spec.model,
                        schema=schema,
                        pydantic_model=model_cls,
                        log_path=str(llm_log_path),
                        temperature=ex.spec.temperature,
                        max_output_tokens=ex.spec.max_output_tokens,
                    )
                    if gen.get("refusal"):
                        attempt_out = {"refusal": gen.get("refusal")}
                    else:
                        attempt_out = gen["json"] if gen.get("json") is not None else gen["text"]
                else:
                    gen = client.generate(
                        messages=messages,
                        model=ex.spec.model,
                        log_path=str(llm_log_path),
                        temperature=ex.spec.temperature,
                        max_output_tokens=ex.spec.max_output_tokens,
                    )
                    attempt_out = {"text": gen["text"]}
        except Exception as e:  # pragma: no cover
            attempt_out = {"error": str(e)}
        # Live console preview of the raw LLM response text (or refusal / error)
        try:
            if not dry_run:
                if gen and gen.get("refusal"):
                    preview_text = f"[refusal] {gen.get('refusal')}"
                elif isinstance(attempt_out, dict) and "error" in attempt_out:
                    preview_text = f"[error] {attempt_out['error']}"
                # Truncate very long outputs for terminal readability
                if len(preview_text) > 1200:
                    preview_text = preview_text[:1200] + "\n... [truncated]"
                panel_title = f"record {_idx}/{len(records)}"
                (progress.console if progress else console).print(
                    Panel(preview_text or "(empty)", title=panel_title, border_style="magenta"),
                    soft_wrap=True,
                )
        except Exception as _preview_exc:  # pragma: no cover
            import logging as _logging
            _logging.debug("Preview rendering failed: %s", _preview_exc)
        usage_meta = None
        latency_ms = None
        if gen:
            usage_meta = gen.get("usage")
            latency_ms = gen.get("latency_ms")
            if isinstance(usage_meta, dict):
                prompt_tokens = (
                    usage_meta.get("prompt_tokens")
                    or usage_meta.get("prompt_tokens_total")
                    or usage_meta.get("input_tokens")
                )
                completion_tokens = (
                    usage_meta.get("completion_tokens")
                    or usage_meta.get("completion_tokens_total")
                    or usage_meta.get("output_tokens")
                )
                total_tokens = (
                    usage_meta.get("total_tokens")
                    or usage_meta.get("tokens")
                    or usage_meta.get("total_tokens_used")
                )
                for v_name, v_val in [
                    ("prompt", prompt_tokens),
                    ("completion", completion_tokens),
                    ("total", total_tokens),
                ]:
                    if isinstance(v_val, int):
                        if v_name == "prompt":
                            token_prompt_total += v_val
                        elif v_name == "completion":
                            token_completion_total += v_val
                        elif v_name == "total":
                            token_total += v_val
            if isinstance(latency_ms, int | float):
                latencies.append(float(latency_ms))
        valid = False
        validation_errors = None
        if model_cls and "error" not in attempt_out:
            try:
                model_cls.model_validate(attempt_out)
                valid = True
            except Exception as ve:  # pragma: no cover
                validation_errors = [str(ve)]
        rec_out = {
            "input": rec,
            "output": attempt_out,
            "validated": valid,
            "validation_errors": validation_errors,
            "meta": {
                "usage": usage_meta,
                "latency_ms": latency_ms,
                "refusal": gen.get("refusal") if gen else None,
                "parse_fallback": gen.get("parse_fallback") if gen else None,
            },
        }
        aggregated.append(rec_out)
        if progress:
            progress.update(task_id, advance=1)
    if progress:
        progress.stop()
    # manifest (run-level metadata, merged later into result.json)
    avg_latency = statistics.mean(latencies) if latencies else None
    manifest = {
        "experiment": ex.spec.name,
        "timestamp": ts,
        "config": ex.to_config_snapshot(),
        "dependencies": dependency_run_refs,
        "count": len(aggregated),
        "dry_run": dry_run,
        "model": ex.spec.model,
        "provider": ex.spec.provider,
        "token_usage": {
            "prompt": token_prompt_total or None,
            "completion": token_completion_total or None,
            "total": token_total or None,
        },
        "avg_latency_ms": round(avg_latency, 2) if avg_latency is not None else None,
    }
    metrics = {
        "records": len(aggregated),
        "prompt_tokens": token_prompt_total,
        "completion_tokens": token_completion_total,
        "total_tokens": token_total,
        "avg_latency_ms": avg_latency,
        "latencies_ms": latencies,
    }
    # unified artifact: result.json
    result_doc = {
        "manifest": manifest,
        "metrics": metrics,
        "records": aggregated,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result_doc, indent=2), encoding="utf-8"
    )
    # Append aggregate summary for this run at experiment root for longitudinal analysis
    aggregate_path = run_dir.parent / "runs_summary.jsonl"
    summary_entry = {
        "timestamp": ts,
        "count": len(aggregated),
        "prompt_tokens": token_prompt_total or None,
        "completion_tokens": token_completion_total or None,
        "total_tokens": token_total or None,
        "avg_latency_ms": round(avg_latency, 2) if avg_latency is not None else None,
        "model": ex.spec.model,
        "provider": ex.spec.provider,
        "schema": ex.spec.schema_model or None,
        "run_dir": run_dir.name,
        "success_pct": round(
            100
            * (sum(1 for r in aggregated if r.get("validated")) / len(aggregated)),
            2,
        )
        if aggregated
        else None,
        "refusals": sum(1 for r in aggregated if r.get("meta", {}).get("refusal")),
        "parse_fallbacks": sum(
            1 for r in aggregated if r.get("meta", {}).get("parse_fallback")
        ),
    }
    try:
        with aggregate_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary_entry) + "\n")
    except Exception as _agg_exc:  # pragma: no cover
        import logging as _l
        _l.debug("Failed to append run summary: %s", _agg_exc)

    # Derive analytics across all runs (lightweight) -> analytics.json (no embedding of recent runs)
    analytics_path = run_dir.parent / "analytics.json"
    try:
        runs: list[dict[str, Any]] = []
        if aggregate_path.exists():
            for line in aggregate_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    runs.append(json.loads(line))
                except Exception as _line_exc:  # pragma: no cover
                    import logging as _l
                    _l.debug("Skipping malformed summary line: %s", _line_exc)
                    continue
        if runs:
            runs_sorted = sorted(runs, key=lambda r: r.get("timestamp", ""))
            lat_values = [
                r["avg_latency_ms"]
                for r in runs_sorted
                if r.get("avg_latency_ms") is not None
            ]
            token_totals = [
                r["total_tokens"]
                for r in runs_sorted
                if r.get("total_tokens") is not None
            ]
            success_vals = [
                r["success_pct"]
                for r in runs_sorted
                if r.get("success_pct") is not None
            ]
            cumulative = {
                "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in runs_sorted),
                "completion_tokens": sum(
                    r.get("completion_tokens") or 0 for r in runs_sorted
                ),
                "total_tokens": sum(
                    r.get("total_tokens") or 0 for r in runs_sorted
                ),
                "refusals": sum(r.get("refusals") or 0 for r in runs_sorted),
                "parse_fallbacks": sum(
                    r.get("parse_fallbacks") or 0 for r in runs_sorted
                ),
            }
            def _extreme(key: str, reverse: bool = False):
                candidates = [r for r in runs_sorted if r.get(key) is not None]
                if not candidates:
                    return None
                target = max(candidates, key=lambda r: r[key]) if reverse else min(
                    candidates, key=lambda r: r[key]
                )
                return {"value": target[key], "run_dir": target.get("run_dir")}
            successes = [r for r in runs_sorted if r.get("success_pct") is not None]
            best_success = (
                max(successes, key=lambda r: r["success_pct"])
                if successes
                else None
            )
            worst_success = (
                min(successes, key=lambda r: r["success_pct"])
                if successes
                else None
            )
            last5 = runs_sorted[-5:]
            rolling5 = {
                "avg_latency_ms": round(
                    sum(r.get("avg_latency_ms") or 0 for r in last5) / len(last5),
                    2,
                ) if last5 else None,
                "avg_success_pct": round(
                    sum(r.get("success_pct") or 0 for r in last5) / len(last5),
                    2,
                ) if last5 else None,
            }
            analytics = {
                "total_runs": len(runs_sorted),
                "first_run": runs_sorted[0]["timestamp"],
                "last_run": runs_sorted[-1]["timestamp"],
                "averages": {
                    "avg_latency_ms": round(sum(lat_values) / len(lat_values), 2)
                    if lat_values
                    else None,
                    "avg_total_tokens": round(
                        sum(token_totals) / len(token_totals), 2
                    )
                    if token_totals
                    else None,
                    "avg_success_pct": round(
                        sum(success_vals) / len(success_vals), 2
                    )
                    if success_vals
                    else None,
                },
                "extremes": {
                    "min_avg_latency_ms": _extreme("avg_latency_ms"),
                    "max_avg_latency_ms": _extreme("avg_latency_ms", reverse=True),
                    "best_success_pct": (
                        {
                            "value": best_success["success_pct"],
                            "run_dir": best_success.get("run_dir"),
                        }
                        if best_success
                        else None
                    ),
                    "worst_success_pct": (
                        {
                            "value": worst_success["success_pct"],
                            "run_dir": worst_success.get("run_dir"),
                        }
                        if worst_success
                        else None
                    ),
                },
                "cumulative": cumulative,
                "rolling_last_5": rolling5,
            }
            analytics_path.write_text(json.dumps(analytics, indent=2), encoding="utf-8")
    except Exception as _an_exc:  # pragma: no cover
        import logging as _l
        _l.debug("Failed to build analytics: %s", _an_exc)
    return run_dir


def run_experiment(
    name: str,
    *,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> Path:
    ex = resolve_experiment(name, overrides=overrides)
    return run_resolved(ex, dry_run=dry_run, limit=limit)


def topo_sort(names: list[str]) -> list[str]:
    all_specs = {n: s for n, (s, _) in load_all_experiments().items()}
    visited: dict[str, int] = {}
    order: list[str] = []

    def visit(n: str):
        if n in visited:
            if visited[n] == 0:
                raise RuntimeError(f"Circular dependency involving {n}")
            return
        visited[n] = 0
        spec = all_specs.get(n)
        if not spec:
            raise ValueError(f"Unknown experiment {n}")
        for dep in spec.depends_on:
            visit(dep.experiment)
        visited[n] = 1
        order.append(n)

    for root in names:
        visit(root)
    return order


def run_chain(names: list[str], *, dry_run: bool = False, limit: int | None = None) -> list[Path]:
    ordered = topo_sort(names)
    out: list[Path] = []
    for name in ordered:
        out.append(run_experiment(name, dry_run=dry_run, limit=limit))
    return out
