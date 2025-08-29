"""Execution & chaining logic for experiments."""

from __future__ import annotations

import datetime as dt
import json
import statistics
import time
from collections.abc import Callable
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


def _import_transform(path: str) -> Callable[..., Any]:  # returns callable
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
    # New artifact layout: prefer aggregate JSONL then fallback to per-record folders
    base_records: list[dict[str, Any]] = []
    agg_path = run_dir / "experiment_results.jsonl"
    if agg_path.exists():
        try:
            for line in agg_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    base_records.append(json.loads(line))
                except Exception as _exc:  # pragma: no cover
                    import logging as _l

                    _l.debug("Skipping malformed dependency aggregate line: %s", _exc)
                    continue
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed reading experiment_results.jsonl for dependency {run_dir}: {exc}"
            ) from exc
    if not base_records:  # fallback scan
        for rec_file in sorted(run_dir.glob("record*/result.json")):
            try:
                data = json.loads(rec_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    base_records.append(data)
            except Exception as _exc:  # pragma: no cover
                import logging as _l

                _l.debug("Skipping malformed dependency record result: %s", _exc)
                continue
    if not base_records:
        raise RuntimeError(
            f"No dependency records found in {run_dir} "
            "(expected experiment_results.jsonl or record*/result.json)"
        )
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
    # Augment template variables: provide 'record' alias for entire row (helps downstream
    # chain templates using {{ record | tojson }}), plus 'upstream' alias if output present.
    _ctx = dict(vars)
    _ctx.setdefault("record", vars)
    if "output" in vars and isinstance(vars["output"], dict):
        _ctx.setdefault("upstream", vars["output"])  # convenience alias
    user_body = tmpl.render(**_ctx).strip()
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


def _parse_record_selection(selection: str | None, total: int) -> list[int]:
    """Parse selection expression into sorted unique 0-based indices.

    Supported forms (1-based in user spec):
      "1"              -> first record
      "1,3,5"          -> discrete indices
      "1-5"            -> inclusive range
      "1-3,7,9-10"     -> combination
      "0" / ranges starting at 0 treat 0 as 1 for convenience.
    Empty / None -> [0]. Out-of-range values are clipped. Duplicates removed.
    """
    if total <= 0:
        return []
    if not selection:
        return [0]
    idxs: set[int] = set()
    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            try:
                a = int(a_str)
                b = int(b_str)
            except ValueError:
                continue
            if a == 0:
                a = 1
            if b == 0:
                b = 1
            if a > b:
                a, b = b, a
            for v in range(a, b + 1):
                if v >= 1:
                    idxs.add(v - 1)
        else:
            try:
                v = int(part)
            except ValueError:
                continue
            if v == 0:
                v = 1
            if v >= 1:
                idxs.add(v - 1)
    # Clip to dataset length
    final = sorted(i for i in idxs if i < total)
    return final if final else [0]


def run_resolved(
    ex: ResolvedExperiment,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    output_root: Path = Path("src/output"),
    injected_records: list[dict[str, Any]] | None = None,
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
    if injected_records is not None:
        dataset_label = f"(chained {len(injected_records)} upstream records)"
    else:
        dataset_label = str(ex.dataset_path) if ex.dataset_path else "(no dataset)"
    meta.add_row("Dataset", dataset_label)
    if limit is not None:
        meta.add_row("Limit", str(limit))
    meta.add_row("Dry Run", str(dry_run))
    console.print(meta)
    # Build dataset from injected records (chaining) or local file
    if injected_records is not None:
        records: list[dict[str, Any]] = injected_records
    else:
        records = _load_dataset(ex.dataset_path) if ex.dataset_path else []
    total_records = len(records)
    if injected_records is not None and ex.spec.record_selection is None:
        # When chaining and no explicit selection, process all upstream records
        selected_indices = list(range(total_records))
    else:
        selected_indices = _parse_record_selection(ex.spec.record_selection, total_records)
    if limit is not None:
        selected_indices = selected_indices[:limit]
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
        "instructions": [p.read_text(encoding="utf-8") for p in (ex.instruction_paths or [])],
    }
    (run_dir / "prompts_used.json").write_text(json.dumps(prompts_used, indent=2), encoding="utf-8")
    # process (streaming per record)
    token_prompt_total = token_completion_total = token_total = 0
    latencies: list[float] = []
    validated_count = 0
    refusals = 0
    parse_fallbacks = 0
    progress: Progress | None = None
    if selected_indices:
        progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"))
        task_id = progress.add_task(f"{ex.spec.name}", total=len(selected_indices))
        progress.start()
    # Open aggregate results JSONL
    agg_results_path = run_dir / "experiment_results.jsonl"
    agg_f = agg_results_path.open("a", encoding="utf-8")
    for position, idx in enumerate(selected_indices, start=1):
        rec = records[idx]
        messages = _build_messages(
            ex.system_prompt_path,
            ex.user_prompt_path,
            ex.instruction_paths,
            rec,
        )
        gen: dict[str, Any] | None = None
        try:
            if dry_run:
                if model_cls:
                    # Produce generic schema-shaped placeholder:
                    # for each required string field -> dummy text; others -> null
                    attempt_out: dict[str, Any] = {}
                    try:
                        schema_obj = model_json_schema(model_cls)["schema"]
                        props = (
                            schema_obj.get("properties", {})
                            if isinstance(schema_obj, dict)
                            else {}
                        )
                        for key, val in props.items():  # type: ignore[assignment]
                            # Heuristic: string type -> placeholder string; else null
                            if isinstance(val, dict) and val.get("type") == "string":
                                attempt_out[key] = f"(dry-run) {key}"
                            else:
                                attempt_out[key] = None
                    except Exception:
                        attempt_out = {"placeholder": "(dry-run)"}
                else:
                    attempt_out = {"text": "(dry-run)"}
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
                panel_title = f"record {position}/{len(selected_indices)}"
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
        meta_block = {
            "usage": usage_meta,
            "latency_ms": latency_ms,
            "refusal": gen.get("refusal") if gen else None,
            "parse_fallback": gen.get("parse_fallback") if gen else None,
        }
        if meta_block["refusal"]:
            refusals += 1
        if meta_block["parse_fallback"]:
            parse_fallbacks += 1
        if valid:
            validated_count += 1
        # Per-record folder artifacts (renamed from inputN -> recordN for clarity)
        record_dir = run_dir / f"record{idx + 1}"
        record_dir.mkdir(parents=True, exist_ok=True)
        (record_dir / "input_manifest.json").write_text(
            json.dumps({"index": idx + 1, "input": rec}, indent=2), encoding="utf-8"
        )
        slim_result = {
            "index": idx + 1,
            "output": attempt_out,
            "usage": usage_meta,
            "latency_ms": latency_ms,
            "validated": valid,
            "validation_errors": validation_errors,
            "refusal": meta_block.get("refusal"),
            "parse_fallback": meta_block.get("parse_fallback"),
        }
        (record_dir / "result.json").write_text(
            json.dumps(slim_result, indent=2), encoding="utf-8"
        )
        # Append only slim_result to aggregate JSONL (omit full input to keep file small)
        agg_f.write(json.dumps(slim_result) + "\n")
        if progress:
            progress.update(task_id, advance=1)
        # Delay between requests (skip after last). Skip in dry_run for speed.
        if not dry_run and position < len(selected_indices):
            # Enforce a fixed 2s inter-request delay to avoid rate spikes
            time.sleep(2)
    if progress:
        progress.stop()
    agg_f.close()
    # manifest (run-level metadata)
    avg_latency = statistics.mean(latencies) if latencies else None
    manifest = {
        "experiment": ex.spec.name,
        "timestamp": ts,
        "config": ex.to_config_snapshot(),
        "count": len(selected_indices),
        "dry_run": dry_run,
        "model": ex.spec.model,
        "provider": ex.spec.provider,
        "token_usage": {
            "prompt": token_prompt_total or None,
            "completion": token_completion_total or None,
            "total": token_total or None,
        },
        "avg_latency_ms": round(avg_latency, 2) if avg_latency is not None else None,
        "record_selection": ex.spec.record_selection or "1",
    }
    metrics = {
        "records": len(selected_indices),
        "prompt_tokens": token_prompt_total,
        "completion_tokens": token_completion_total,
        "total_tokens": token_total,
        "avg_latency_ms": avg_latency,
        "latencies_ms": latencies,
        "validated": validated_count,
        "success_pct": round(100 * validated_count / len(selected_indices), 2)
        if selected_indices
        else None,
        "refusals": refusals,
        "parse_fallbacks": parse_fallbacks,
    }
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps({"manifest": manifest, "metrics": metrics}, indent=2),
        encoding="utf-8",
    )
    # Append aggregate summary for this run at experiment root for longitudinal analysis
    aggregate_path = run_dir.parent / "runs_summary.jsonl"
    summary_entry = {
        "timestamp": ts,
        "count": len(selected_indices),
        "prompt_tokens": token_prompt_total or None,
        "completion_tokens": token_completion_total or None,
        "total_tokens": token_total or None,
        "avg_latency_ms": round(avg_latency, 2) if avg_latency is not None else None,
        "model": ex.spec.model,
        "provider": ex.spec.provider,
        "schema": ex.spec.schema_model or None,
        "run_dir": run_dir.name,
        "success_pct": metrics.get("success_pct"),
        "refusals": refusals,
        "parse_fallbacks": parse_fallbacks,
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
                r["avg_latency_ms"] for r in runs_sorted if r.get("avg_latency_ms") is not None
            ]
            token_totals = [
                r["total_tokens"] for r in runs_sorted if r.get("total_tokens") is not None
            ]
            success_vals = [
                r["success_pct"] for r in runs_sorted if r.get("success_pct") is not None
            ]
            cumulative = {
                "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in runs_sorted),
                "completion_tokens": sum(r.get("completion_tokens") or 0 for r in runs_sorted),
                "total_tokens": sum(r.get("total_tokens") or 0 for r in runs_sorted),
                "refusals": sum(r.get("refusals") or 0 for r in runs_sorted),
                "parse_fallbacks": sum(r.get("parse_fallbacks") or 0 for r in runs_sorted),
            }

            def _extreme(key: str, reverse: bool = False):
                candidates = [r for r in runs_sorted if r.get(key) is not None]
                if not candidates:
                    return None
                target = (
                    max(candidates, key=lambda r: r[key])
                    if reverse
                    else min(candidates, key=lambda r: r[key])
                )
                return {"value": target[key], "run_dir": target.get("run_dir")}

            successes = [r for r in runs_sorted if r.get("success_pct") is not None]
            best_success = max(successes, key=lambda r: r["success_pct"]) if successes else None
            worst_success = min(successes, key=lambda r: r["success_pct"]) if successes else None
            last5 = runs_sorted[-5:]
            rolling5 = {
                "avg_latency_ms": round(
                    sum(r.get("avg_latency_ms") or 0 for r in last5) / len(last5),
                    2,
                )
                if last5
                else None,
                "avg_success_pct": round(
                    sum(r.get("success_pct") or 0 for r in last5) / len(last5),
                    2,
                )
                if last5
                else None,
            }
            analytics = {
                "total_runs": len(runs_sorted),
                "first_run": runs_sorted[0]["timestamp"],
                "last_run": runs_sorted[-1]["timestamp"],
                "averages": {
                    "avg_latency_ms": round(sum(lat_values) / len(lat_values), 2)
                    if lat_values
                    else None,
                    "avg_total_tokens": round(sum(token_totals) / len(token_totals), 2)
                    if token_totals
                    else None,
                    "avg_success_pct": round(sum(success_vals) / len(success_vals), 2)
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
    # NOTE: Final completion panel handled in CLI after optional sample output rendering
    return run_dir


def run_experiment(
    name: str,
    *,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    injected_records: list[dict[str, Any]] | None = None,
) -> Path:
    """Resolve & run a single experiment.

    injected_records: optional in-memory dataset used when chaining to supply
    upstream outputs instead of reading a dataset file.
    """
    ex = resolve_experiment(name, overrides=overrides)
    return run_resolved(
        ex,
        dry_run=dry_run,
        limit=limit,
        injected_records=injected_records,
    )


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


def _load_upstream_records(run_dir: Path) -> list[dict[str, Any]]:
    """Load slim per-record outputs from previous run for chaining."""
    records: list[dict[str, Any]] = []
    if not run_dir.exists():  # pragma: no cover
        return records
    for result_file in sorted(run_dir.glob("record*/result.json")):
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if "output" in data and isinstance(data["output"], dict):
                    data.setdefault("upstream_output", data["output"])
                records.append(data)
        except Exception as exc:  # pragma: no cover
            import logging as _l
            _l.debug("Skipping malformed upstream record %s: %s", result_file, exc)
            continue
    return records


def run_chain(names: list[str], *, dry_run: bool = False, limit: int | None = None) -> list[Path]:
    """Run experiments in topological order; pass outputs downstream.

    Semantics: each experiment's per-record slim outputs (result.json) become the
    dataset for the next experiment in the chain unless that experiment has its
    own dataset file (in which case injected records override only if provided
    explicitly via chaining). Record selection on a chained experiment applies to
    the injected upstream records; if no selection specified it processes all.
    """
    ordered = topo_sort(names)
    out: list[Path] = []
    upstream: list[dict[str, Any]] | None = None
    for name in ordered:
        run_path = run_experiment(
            name,
            dry_run=dry_run,
            limit=limit,
            injected_records=upstream,
        )
        out.append(run_path)
        upstream = _load_upstream_records(run_path)
    return out
