"""Ad-hoc batch command (direct prompts + dataset without registry).

Prefer defining experiments under `experiments/<name>/` and using
`uv run exp experiment run <name>` or `uv run exp chain run <a> <b>` for chaining.
This command remains for quick one-off generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from app.experiments.definitions import ExperimentSpec, ResolvedExperiment
from app.experiments.runner import run_resolved
from app.runtime import AppContext, bootstrap
from app.utils.logging import setup_logging

app = typer.Typer(help="Run batch structured extraction src.")


@app.command(name="run")
def batch_run(
    model: Annotated[str, typer.Option(help="Model name")] = "gpt-4o-mini",
    dataset_path: Annotated[
        Path, typer.Option(help="Input JSONL dataset")
    ] = Path("src/datasets/samples.jsonl"),
    system_prompt: Annotated[
        Path, typer.Option(help="System prompt path")
    ] = Path("src/experiments/acq_yaml/prompts/system/extraction.txt"),
    user_prompt: Annotated[
        Path, typer.Option(help="User prompt path")
    ] = Path("src/experiments/acq_yaml/prompts/user/extraction.jinja"),
    instructions: Annotated[
        list[Path] | None,
        typer.Option(
            help="Optional one or more instruction snippet files (can repeat flag)",
            rich_help_panel="Prompt Composition",
        ),
    ] = None,
    schema_model: Annotated[
        str,
        typer.Option(help="Pydantic model path (omit or empty for free-form)")
    ] = "",
    provider: Annotated[str, typer.Option(help="openai|azure")] = "openai",
    run_base: Annotated[str, typer.Option(help="Run output base dir")] = "src/output",
    limit: Annotated[int | None, typer.Option(help="Limit number of records") ] = None,
    temperature: Annotated[float, typer.Option(help="Generation temperature")] = 0.0,
    max_output_tokens: Annotated[int, typer.Option(help="Max output tokens")] = 512,
    dry_run: Annotated[
        bool, typer.Option(help="Skip API calls and emit empty schema instances for testing")
    ] = False,
) -> None:
    bootstrap()
    setup_logging()
    AppContext.get()  # ensure context loaded
    spec = ExperimentSpec(
        name="adhoc",
        model=model,
        provider=provider,
        schema_model=schema_model if schema_model else None,
        system_prompt=str(system_prompt),
        user_prompt=str(user_prompt),
        instructions=[str(p) for p in (instructions or [])],
        dataset=str(dataset_path),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    resolved = ResolvedExperiment(
        spec=spec,
        root_dir=Path("."),
        registry_dir=Path("."),
        system_prompt_path=spec.system_prompt and Path(spec.system_prompt),
        user_prompt_path=spec.user_prompt and Path(spec.user_prompt),
        instruction_paths=[Path(p) for p in spec.instructions or []],
        dataset_path=Path(spec.dataset) if spec.dataset else None,
    )
    run_dir = run_resolved(
        resolved,
        dry_run=dry_run,
        limit=limit,
        output_root=Path(run_base),
    )
    console = Console()
    config = {"model": model, "provider": provider, "schema_model": schema_model or None}
    table = Table(title="Batch Summary (adhoc)")
    table.add_column("Key")
    table.add_column("Value")
    for k, v in config.items():
        table.add_row(k, str(v))
    table.add_row("run_dir", str(run_dir))
    console.print(table)
    print(f"[green]Run complete[/green] -> {run_dir}")


if __name__ == "__main__":  # pragma: no cover
    app()
