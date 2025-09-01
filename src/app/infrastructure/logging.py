"""Logging & console helpers for app architecture.

Features:
    * RichHandler based structured console logging (color, tracebacks)
    * Optional JSON logging mode (machine ingest)
    * Helper utilities (`get_console`, `render_panel`) so service layers avoid
        importing rich directly, keeping presentation concerns centralized.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

try:  # pragma: no cover
    from rich.console import Console  # type: ignore
    from rich.logging import RichHandler  # type: ignore
    from rich.panel import Panel  # type: ignore
except Exception:  # pragma: no cover
    Console = None  # type: ignore
    RichHandler = None  # type: ignore
    Panel = None  # type: ignore


_INITIALIZED = False
_JSON_MODE = False
_CONSOLE: Any | None = None  # rich Console instance if available


class _JsonHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - simple
        try:
            data = {
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                data["exc_info"] = logging.Formatter().formatException(record.exc_info)
            line = json.dumps(data, ensure_ascii=False)
            print(line)
        except Exception as exc:  # pragma: no cover
            logging.getLogger(__name__).debug("json logging emit failed: %s", exc)


def setup_logging(level: str | None = None, json_mode: bool | None = None) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    global _JSON_MODE
    if json_mode is not None:
        _JSON_MODE = json_mode
    lvl_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    try:
        lvl = getattr(logging, lvl_name, logging.INFO)
    except Exception:  # pragma: no cover
        lvl = logging.INFO
    handlers: list[logging.Handler] = []
    if _JSON_MODE:
        handlers.append(_JsonHandler())
    elif RichHandler is not None:
        handlers.append(RichHandler(rich_tracebacks=True, markup=True))
    logging.basicConfig(level=lvl, handlers=handlers or None, force=True,
                        format="%(levelname)s %(name)s: %(message)s")
    _INITIALIZED = True


def enable_json_logging() -> None:
    """Switch to JSON logging (idempotent)."""
    setup_logging(json_mode=True)


def get_console():  # -> Console | None (typed loosely to avoid runtime when rich absent)
    """Return a shared rich Console if rich is installed.

    Services should *not* import rich directly; use this accessor to keep
    presentation optional & centralized.
    """
    global _CONSOLE
    if Console is None:  # pragma: no cover - rich missing
        return None
    if _CONSOLE is None:
        _CONSOLE = Console()
    return _CONSOLE


def render_panel(title: str, body: str, *, style: str = "cyan") -> None:
    """Render a panel to console (noop if rich unavailable).

    Falls back to a plain log line when rich is not installed.
    """
    if Panel is None or get_console() is None:  # pragma: no cover
        logging.getLogger("app.console").info("%s | %s", title, body)
        return
    get_console().print(Panel.fit(body, title=title, border_style=style))


def log_experiment_start(*, name: str, description: str | None) -> None:
    """Standard experiment start banner."""
    desc = description or ""
    body = f"[bold cyan]Experiment:[/bold cyan] {name}\n[dim]{desc}[/dim]"
    render_panel("run", body, style="cyan")

__all__ = [
    "enable_json_logging",
    "get_console",
    "log_experiment_start",
    "render_panel",
    "setup_logging",
]
