from __future__ import annotations

from typing import Annotated

import typer
from rich import print
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from app.experiments.loader import load_all_experiments, resolve_experiment
from app.experiments.runner import run_chain, run_experiment
from app.pipelines.evaluate_run import app as eval_app
from app.runtime import bootstrap

app = typer.Typer(help="Experiment CLI root (list/show/run/chain)")


@app.callback(invoke_without_command=True)
def init_callback() -> None:
    bootstrap()


app.add_typer(eval_app, name="eval")


chain_app = typer.Typer(help="Run one or more experiments respecting dependencies")


@app.command("list")
def list_experiments(
    json_out: Annotated[bool, typer.Option("--json", help="Output raw JSON list")] = False,
) -> None:
    console = Console()
    exps = load_all_experiments()
    if not exps:
        console.print("[yellow]No experiments found[/yellow]")
        return
    rows = []
    for name, (spec, path) in sorted(exps.items()):
        rows.append(
            {
                "name": name,
                "description": spec.description,
                "path": str(path),
                "model": spec.model,
                "schema_model": spec.schema_model,
                "depends_on": [d.experiment for d in spec.depends_on],
            }
        )
    if json_out:
        console.print(JSON.from_data(rows))
        return
    table = Table(title="Experiments", show_lines=False)
    table.add_column("Name", style="bold cyan")
    table.add_column("Description", overflow="fold")
    table.add_column("Deps", style="yellow")
    table.add_column("Model", style="magenta")
    table.add_column("Schema", style="green")
    table.add_column("Path", style="dim")
    for r in rows:
        table.add_row(
            r["name"],
            r["description"] or "",
            ",".join(r["depends_on"]) or "-",
            r["model"],
            r["schema_model"] or "-",
            r["path"],
        )
    console.print(table)


@app.command("show")
def show_experiment(
    name: str,
    json_out: Annotated[bool, typer.Option("--json", help="Output raw JSON snapshot")] = False,
) -> None:
    console = Console()
    resolved = resolve_experiment(name)
    snapshot = resolved.to_config_snapshot()
    if json_out:
        console.print(JSON.from_data(snapshot))
        return
    deps = snapshot.get("depends_on", [])
    dep_names = [d["experiment"] for d in deps] if deps else []
    header = f"[bold]{snapshot['name']}[/bold]" + (
        f" (deps: {', '.join(dep_names)})" if dep_names else ""
    )
    console.print(Panel.fit(header, subtitle="experiment", border_style="cyan"))
    console.print(JSON.from_data(snapshot))


def _parse_overrides(values: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in values:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        # naive type coercion
        if v.isdigit():
            out[k] = int(v)
        else:
            try:
                out[k] = float(v)
            except ValueError:
                if v.lower() in {"true", "false"}:
                    out[k] = v.lower() == "true"
                else:
                    out[k] = v
    return out


@app.command("run")
def run_experiment_cmd(
    name: str,
    override: Annotated[list[str] | None, typer.Option(help="key=value overrides")] = None,
    dry_run: Annotated[bool, typer.Option()] = False,
    limit: Annotated[int | None, typer.Option()] = None,
) -> None:
    overrides = _parse_overrides(override or [])
    path = run_experiment(name, overrides=overrides, dry_run=dry_run, limit=limit)
    # Enhanced summary output (consume result.json if present)
    import json as _j
    import logging as _logging
    from pathlib import Path as _P

    manifest_path = _P(path) / "experiment_manifest.json"
    results_path = _P(path) / "experiment_results.jsonl"
    if manifest_path.exists() and results_path.exists():
        try:
            data = _j.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = data.get("manifest", {})
            metrics = data.get("metrics", {})
            console = Console()
            info_table = Table(box=None, show_header=False)
            info_table.add_row("Records", str(metrics.get("records")))
            tok = manifest.get("token_usage", {})
            info_table.add_row(
                "Tokens",
                "p:{p} c:{c} t:{t}".format(
                    p=tok.get("prompt") or 0,
                    c=tok.get("completion") or 0,
                    t=tok.get("total") or 0,
                ),
            )
            if metrics.get("avg_latency_ms"):
                info_table.add_row("Avg latency (ms)", f"{metrics['avg_latency_ms']:.2f}")
            if metrics.get("success_pct") is not None:
                info_table.add_row("Success %", f"{metrics['success_pct']:.2f}")
            console.print(info_table)
            # Read first line sample output
            first_output = None
            with results_path.open("r", encoding="utf-8") as rf:
                first_line = rf.readline().strip()
                if first_line:
                    import contextlib as _ctx
                    with _ctx.suppress(Exception):
                        first_output = _j.loads(first_line).get("output")
            if first_output is not None:
                sample_render = Panel(
                    JSON.from_data(first_output)
                    if isinstance(first_output, dict)
                    else str(first_output),
                    title="LLM Output",
                    border_style="magenta",
                )
                console.print(sample_render)
            try:
                completion_text = (
                    f"[bold green]Run Completed[/bold green]\n"
                    f"[cyan]Run Dir:[/cyan] {path}\n"
                    f"[cyan]Manifest:[/cyan] {manifest_path}\n"
                    f"[cyan]Results:[/cyan] {results_path}"
                )
                completion_panel = Panel(
                    completion_text,
                    title="completed",
                    border_style="green",
                )
                console.print(completion_panel)
            except Exception as _panel_exc:  # pragma: no cover
                _logging.debug("Failed rendering completion panel: %s", _panel_exc)
            return
        except Exception as _summary_exc:  # pragma: no cover
            _logging.debug("Failed enhanced run summary: %s", _summary_exc)
    print(f"[green]Run complete[/green] -> {path}")


@chain_app.command("run")
def chain_run(
    names: Annotated[list[str], typer.Argument(help="Experiment names (space separated)")],
    dry_run: Annotated[bool, typer.Option(help="Skip API calls")] = False,
    limit: Annotated[int | None, typer.Option(help="Limit records per experiment")] = None,
) -> None:
    """Run experiments ensuring dependencies execute first.

    Provide one or more experiment names. Any transitive dependencies
    not explicitly listed are auto-included.
    """
    paths = run_chain(names, dry_run=dry_run, limit=limit)
    for p in paths:
        print(f"[green]Run complete[/green] -> {p}")


app.add_typer(chain_app, name="chain")


if __name__ == "__main__":  # pragma: no cover
    app()
