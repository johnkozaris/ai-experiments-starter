"""Evaluation CLI for app runs (produces evaluation.json)."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Evaluate an app run directory")


def _iter_results(run_dir: Path) -> Iterable[dict]:
    jsonl = run_dir / "experiment_results.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as exc:  # pragma: no cover - skip malformed line
                logging.getLogger(__name__).debug("bad result line: %s", exc)
                continue
        return
    for rec in sorted(run_dir.glob("record*/result.json")):
        try:
            yield json.loads(rec.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            logging.getLogger(__name__).debug("bad record file: %s", exc)
            continue


@app.command("run")
def eval_run(run_dir: Path) -> None:
    if not run_dir.exists():  # pragma: no cover
        raise typer.BadParameter(f"Run dir not found: {run_dir}")
    total = 0
    validated = 0
    field_presence: dict[str, int] = Counter()
    field_types: dict[str, Counter] = defaultdict(Counter)
    for rec in _iter_results(run_dir):
        total += 1
        out_struct = rec.get("structured_output") or {}
        if not isinstance(out_struct, dict):
            continue
        # Treat existence of any structured_output as validated
        if out_struct:
            validated += 1
        for k, v in out_struct.items():
            if v not in (None, "", [], {}):
                field_presence[k] += 1
                field_types[k][type(v).__name__] += 1
    metrics = {
        "total": total,
        "validated": validated,
        "validation_rate": (validated / total) if total else 0.0,
        "field_non_null_ratio": {k: field_presence[k] / total for k in field_presence},
        "field_type_distribution": {k: dict(field_types[k]) for k in field_types},
    }
    (run_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    console = Console()
    table = Table(title="Evaluation")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("total", str(metrics["total"]))
    table.add_row("validated", str(metrics["validated"]))
    table.add_row("validation_rate", f"{metrics['validation_rate']:.2%}")
    for k, v in metrics["field_non_null_ratio"].items():
        table.add_row(f"field:{k}", f"{v:.2%}")
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
