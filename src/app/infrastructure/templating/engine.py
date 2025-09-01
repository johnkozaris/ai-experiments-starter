"""Templating engine abstraction (Jinja2) used by prompt & instruction builders.

Current goals:
  * Centralize Jinja environment so filters/globals are consistent.
  * Lightweight file caching (recompile on mtime change).
  * Safe fallbacks: never raise during rendering in runtime path (returns raw text).
  * Provide a small set of convenience filters used in legacy harness (tojson, truncate).

NOT YET MIGRATED (future work, if needed):
  * StrictUndefined / diagnostics mode toggle.
  * Custom per-experiment filter registration via spec.
  * Partial include helper (e.g. {{ include('file.txt') }}).
  * Macro library (date normalization, numeric parsing, etc.).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import orjson
from jinja2 import Environment, Template, select_autoescape

_env = Environment(autoescape=select_autoescape(enabled_extensions=("html", "xml")))


# ---------------------------- Filters & Globals ---------------------------- #

def _f_tojson(value: Any, *, indent: int = 0) -> str:
    try:
        opt = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(value, option=opt).decode("utf-8")
    except Exception:  # pragma: no cover
        return str(value)


def _f_truncate(value: Any, length: int = 120, suffix: str = "…") -> str:
    s = str(value)
    return s if len(s) <= length else s[: max(0, length - len(suffix))] + suffix


_env.filters.setdefault("tojson", _f_tojson)
_env.filters.setdefault("truncate", _f_truncate)


def register_filter(name: str, fn: Callable[[Any], Any]) -> None:
    """Public helper to register additional filters (idempotent)."""

    if name not in _env.filters:
        _env.filters[name] = fn  # pragma: no cover - trivial


# ------------------------------ File Caching ------------------------------- #
_CACHE: dict[Path, tuple[float, Template]] = {}


def _get_template(path: Path) -> Template:
    try:
        stat = path.stat()
    except FileNotFoundError:  # pragma: no cover
        return _env.from_string("")
    mtime = stat.st_mtime
    cached = _CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    text = path.read_text(encoding="utf-8")
    tmpl = _env.from_string(text)
    _CACHE[path] = (mtime, tmpl)
    return tmpl


# ----------------------------- Render Function ----------------------------- #

def render_template(path: Path, ctx: dict[str, Any]) -> str:
    """Render template at path with context.

    Silently falls back to raw file contents on any exception (keeps run moving).
    """
    try:
        template = _get_template(path)
        return template.render(**ctx)
    except Exception:  # pragma: no cover
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""


__all__ = ["register_filter", "render_template"]
