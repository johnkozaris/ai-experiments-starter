from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help=(
        "Evaluate a run using new artifact layout: experiment_manifest.json, "
        "experiment_results.jsonl, recordN/result.json"
    )
)


def _iter_results(run_dir: Path):
    """Yield per-record slim results.

    Priority order:
    1. experiment_results.jsonl (stream aggregate)
    2. Fallback: iterate record*/result.json (if JSONL missing / partial)
    """
    jsonl_path = run_dir / "experiment_results.jsonl"
    if jsonl_path.exists():
        import logging as _logging

        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as _exc:  # pragma: no cover
                _logging.debug("Skipping malformed aggregate line: %s", _exc)
                continue
        return
    # Fallback scan (robust to partial writes / manual edits)
    import logging as _logging

    for rec_dir in sorted(run_dir.glob("record*/result.json")):
        try:
            data = json.loads(rec_dir.read_text(encoding="utf-8"))
            yield data
        except Exception as _exc:  # pragma: no cover
            _logging.debug("Skipping malformed record result: %s", _exc)
            continue


@app.command()
def run(run_dir: str) -> None:
    run_dir_path = Path(run_dir)
    if not run_dir_path.exists():  # pragma: no cover
        raise typer.BadParameter(f"Run directory not found: {run_dir_path}")
    total = 0
    errors = 0
    validated_success = 0
    field_presence: dict[str, int] = Counter()
    field_value_counts: dict[str, Counter] = defaultdict(Counter)

    for rec in _iter_results(run_dir_path):
        total += 1
        out = rec.get("output", {}) if isinstance(rec, dict) else {}
        if isinstance(out, dict) and "error" in out:
            errors += 1
            continue
        if rec.get("validated"):
            validated_success += 1
            if isinstance(out, dict):
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

    # Write dedicated evaluation artifact (non-destructive)
    (run_dir_path / "evaluation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    console = Console()
    table = Table(title="Run Evaluation")
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
