from __future__ import annotations

import logging
import os
from pathlib import Path

from rich.logging import RichHandler

_DEF_LEVEL = logging.INFO


def setup_logging(level: int = _DEF_LEVEL) -> None:
    if logging.getLogger().handlers:
        return
    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
    ]
    log_dir = os.getenv("LOG_DIR", "logs")
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / "app.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        handlers.append(file_handler)
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).debug("File logging setup failed: %s", exc)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
