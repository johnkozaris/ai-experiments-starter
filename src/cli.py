from __future__ import annotations

import typer

from src.app.pipelines.evaluate_run import app as eval_app
from src.app.pipelines.run_batch import app as batch_app
from src.app.runtime import bootstrap

app = typer.Typer(help="Experiment CLI root")


@app.callback(invoke_without_command=True)
def init_callback() -> None:
    """Bootstrap environment (dotenv + config + logging) before any command."""
    bootstrap()
app.add_typer(batch_app, name="batch")
app.add_typer(eval_app, name="eval")


if __name__ == "__main__":
    app()
