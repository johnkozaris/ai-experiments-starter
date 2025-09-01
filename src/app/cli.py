"""New CLI layer for app architecture (list/show/run/chain)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from app.infrastructure.llm.client import LLMClient
from app.infrastructure.logging import render_panel, setup_logging
from app.services.deps import topo_sort
from app.services.resolver import discover_experiments, resolve_experiment
from app.services.runner import chain_run, run_experiment

app = typer.Typer(help="app experiment CLI")


def _console() -> Console:
    return Console()


@app.callback()
def init() -> None:  # load env once
    load_dotenv(override=False)
    setup_logging()


def _client_factory(provider: str) -> LLMClient:
    return LLMClient(provider)


DEFAULT_ROOT = Path("src/experiments")


@app.command("list")
def list_cmd(
    root: Annotated[Path, typer.Option(help="Experiments root")] = Path("src/experiments"),
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON list")] = False,
) -> None:
    cons = _console()
    mapping = discover_experiments(root)
    if not mapping:
        cons.print("[yellow]No experiments found[/yellow]")
        return
    rows: list[dict[str, Any]] = []
    for name, _path in sorted(mapping.items()):
        resolved = resolve_experiment(name, root)
        rows.append(
            {
                "name": name,
                "model": resolved.spec.model,
                "provider": resolved.spec.provider,
                "schema": resolved.spec.schema_model or "-",
                "records": len(resolved.dataset_records or []),
            }
        )
    if json_out:
        cons.print(JSON.from_data(rows))
        return
    table = Table(title="Experiments")
    for col in ("Name", "Model", "Provider", "Schema", "Dataset Records"):
        table.add_column(col)
    for r in rows:
        table.add_row(r["name"], r["model"], r["provider"], r["schema"], str(r["records"]))
    cons.print(table)


@app.command("show")
def show_cmd(
    name: str,
    root: Annotated[Path, typer.Option(help="Experiments root")] = Path("src/experiments"),
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON spec")] = False,
) -> None:
    cons = _console()
    resolved = resolve_experiment(name, root)
    spec = resolved.spec.model_dump()
    spec["schema_attached"] = bool(resolved.schema_model_path)
    if json_out:
        cons.print(JSON.from_data(spec))
        return
    cons.print(JSON.from_data(spec))


ALLOWED_OVERRIDE_KEYS = {"temperature", "max_output_tokens", "model", "provider"}


def _parse_overrides(values: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        if k not in ALLOWED_OVERRIDE_KEYS:
            raise typer.BadParameter(f"Unknown override key: {k}")
        if v.isdigit():
            out[k] = int(v)
            continue
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        if v.lower() in {"true", "false"}:
            out[k] = v.lower() == "true"
        else:
            out[k] = v
    return out


@app.command("run")
def run_cmd(
    name: str,
    root: Annotated[Path, typer.Option(help="Experiments root")] = Path("src/experiments"),
    dry_run: Annotated[bool, typer.Option(help="Skip API calls")] = False,
    limit: Annotated[int | None, typer.Option(help="Limit records")] = None,
    override: Annotated[list[str] | None, typer.Option(help="key=value overrides")] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Emit manifest JSON only")] = False,
) -> None:
    # console not needed directly; render_panel centralizes output
    resolved = resolve_experiment(name, root)
    spec = resolved.spec.model_copy(update=_parse_overrides(override or []))
    client = None if dry_run else _client_factory(spec.provider)
    out = run_experiment(
        spec=spec,
        resolved=resolved,
        client=client,
    output_root=Path("output"),
        dry_run=dry_run,
        limit=limit,
    )
    # Pretty summary panel + sample outputs/errors (first 3)
    from rich import print as rprint  # local import
    sample: list[str] = []
    for r in getattr(out, "results", [])[:3]:  # type: ignore[index]
        if getattr(r, "error_message", None):
            sample.append(f"[red]record {r.index} error[/red]: {r.error_message}")
        elif getattr(r, "output_text", None):
            text = (r.output_text or "").strip().replace("\n", " ")
            if len(text) > 200:
                text = text[:197] + "..."
            sample.append(f"record {r.index} output: {text}")
    if sample:
        rprint(Panel("\n".join(sample), title="sample outputs", border_style="magenta"))
    # Pretty summary panel
    # Load metrics.json for summary (manifest is intentionally slim)
    metrics_path = out.run_dir / "metrics.json"
    metrics_data = {}
    if metrics_path.exists():
        import json as _json
        try:
            metrics_data = _json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics_data = {}
    info = [f"[bold cyan]{spec.name}[/bold cyan] run completed"]
    if metrics_data:
        info.append(
            f"records: {metrics_data.get('total_records')} (selected {out.manifest.selected_count})"
        )
        info.append(
            "tokens: p={p} c={c} t={t}".format(
                p=metrics_data.get("prompt_tokens"),
                c=metrics_data.get("completion_tokens"),
                t=metrics_data.get("total_tokens"),
            )
        )
        if metrics_data.get("success_pct") is not None:
            info.append(
                "success: {sp:.1f}% refusals={rf}".format(
                    sp=metrics_data.get("success_pct"), rf=metrics_data.get("refusals")
                )
            )
        if metrics_data.get("avg_latency_ms") is not None:
            info.append(
                "latency ms: "
                f"avg={metrics_data.get('avg_latency_ms') or 0.0:.1f} "
                f"p50={metrics_data.get('p50_latency_ms') or 0.0:.1f} "
                f"p95={metrics_data.get('p95_latency_ms') or 0.0:.1f}"
            )
        if metrics_data.get("est_cost_total") is not None:
            info.append(f"est cost: ${metrics_data.get('est_cost_total'):.4f}")
    if json_out:
        rprint(JSON.from_data(out.manifest.model_dump()))
    else:
        render_panel("run summary", "\n".join(info), style="green")
        # Show clickable paths panel (first record result if present)
        run_dir = out.run_dir
        first_result = None
        rec1 = run_dir / "record1" / "result.json"
        if rec1.exists():
            first_result = rec1
        body_lines: list[str] = []
        run_dir_url = run_dir.resolve().as_uri()
        body_lines.append(f"Run Dir: [link={run_dir_url}]{run_dir}[/link]")
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            metrics_url = metrics_file.resolve().as_uri()
            body_lines.append(f"Metrics: [link={metrics_url}]{metrics_file}[/link]")
        if first_result:
            result_url = first_result.resolve().as_uri()
            body_lines.append(f"Record1 Result: [link={result_url}]{first_result}[/link]")
        from rich import print as rprint  # local import
        rprint(Panel("\n".join(body_lines), title="artifacts", border_style="blue"))


@app.command("chain")
def chain_cmd(
    names: list[str],
    root: Annotated[Path, typer.Option(help="Experiments root")] = Path("src/experiments"),
    dry_run: Annotated[bool, typer.Option(help="Skip API calls")] = False,
    limit: Annotated[int | None, typer.Option(help="Limit records per experiment")] = None,
    show_order: Annotated[
        bool,
        typer.Option(
            "--show-order",
            help="Only print execution order",
        ),
    ] = False,
) -> None:
    # console not needed directly; render_panel centralizes output
    def _resolver(n: str):
        return resolve_experiment(n, root)
    order = topo_sort(names, root=root)
    if show_order:
        render_panel("execution order", " -> ".join(order), style="cyan")
        return
    paths = chain_run(
        names=order,
        resolver=_resolver,
        client_factory=_client_factory,
    output_root=Path("output"),
        dry_run=dry_run,
        limit=limit,
    )
    # Build clickable list
    from rich import print as rprint
    lines: list[str] = []
    for p in paths:
        p_url = p.resolve().as_uri()
        lines.append(f"[link={p_url}]{p}[/link]")
    rprint(Panel("\n".join(lines), title="chain run dirs", border_style="green"))


if __name__ == "__main__":  # pragma: no cover
    app()
