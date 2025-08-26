from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from app.utils.io import read_jsonl

app = typer.Typer(help="Evaluate a completed run directory (compute basic metrics)")


@app.command()
def run(run_dir: str) -> None:
    run_dir_path = Path(run_dir)
    # Prefer consolidated result.json; fallback to legacy results.jsonl
    result_json_path = run_dir_path / "result.json"
    legacy_path = run_dir_path / "results.jsonl"
    records_iter = []
    if result_json_path.exists():
        try:
            data = json.loads(result_json_path.read_text(encoding="utf-8"))
            records_iter = data.get("records", []) if isinstance(data, dict) else []
        except Exception as exc:  # pragma: no cover
            raise typer.BadParameter(f"Failed reading result.json: {exc}") from exc
    elif legacy_path.exists():
        records_iter = list(read_jsonl(legacy_path))
    else:  # pragma: no cover
        raise typer.BadParameter(
            f"No result.json or results.jsonl found in {run_dir_path}"
        )

    total = 0
    errors = 0
    validated_success = 0
    field_presence: dict[str, int] = Counter()
    field_value_counts: dict[str, Counter] = defaultdict(Counter)

    for rec in records_iter:
        total += 1
        out = rec.get("output", {})
        if "error" in out:
            errors += 1
            continue
        if rec.get("validated"):
            validated_success += 1
            for k, v in out.items():
                if v not in (None, [], ""):
                    field_presence[k] += 1
                    field_value_counts[k][type(v).__name__] += 1

    metrics = {
        "total": total,
        "errors": errors,
        "error_rate": (errors / total) if total else 0.0,
        "validated_success": validated_success,
        "validation_rate": (validated_success / total) if total else 0.0,
        "field_non_null_ratio": {
            k: field_presence[k] / total for k in sorted(field_presence.keys())
        },
        "field_type_distribution": {
            k: dict(field_value_counts[k]) for k in sorted(field_value_counts.keys())
        },
    }

    # Attach evaluation into existing result.json if present (non-destructive augment)
    if result_json_path.exists():
        try:
            data = json.loads(result_json_path.read_text(encoding="utf-8"))
            data["evaluation"] = metrics
            result_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as _augment_exc:  # pragma: no cover
            import logging as _logging
            _logging.debug("Failed to augment result.json with evaluation: %s", _augment_exc)
    else:  # legacy path fallback write
        (run_dir_path / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

    console = Console()
    table = Table(title="Run Metrics")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("total", str(metrics["total"]))
    table.add_row("errors", str(metrics["errors"]))
    table.add_row("error_rate", f"{metrics['error_rate']:.2%}")
    table.add_row("validated_success", str(metrics["validated_success"]))
    table.add_row("validation_rate", f"{metrics['validation_rate']:.2%}")
    for k, v in metrics["field_non_null_ratio"].items():
        table.add_row(f"field:{k}", f"{v:.2%}")
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
