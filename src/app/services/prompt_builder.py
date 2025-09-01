"""Build messages & instructions from prompt templates."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.infrastructure.templating.engine import render_template


def build_messages(
    *,
    system: Path,
    user: Path,
    developer: Path | None,
    vars: dict[str, Any],
) -> list[dict[str, str]]:
    ctx = dict(vars)
    # Provide legacy-friendly alias for chained output
    if "output" in ctx and "upstream_output" not in ctx:
        ctx["upstream_output"] = ctx["output"]
    ctx.setdefault("record", vars)
    if "output" in vars and isinstance(vars["output"], dict):
        ctx["output_keys"] = list(vars["output"].keys())
    system_text = render_template(system, ctx)
    user_text = render_template(user, ctx)
    messages: list[dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    if developer is not None:
        dev_text = render_template(developer, ctx)
        if dev_text:
            messages.append({"role": "developer", "content": dev_text})
    messages.append({"role": "user", "content": user_text})
    return messages


def render_instructions(paths: Iterable[Path], ctx: dict[str, Any]) -> str | None:
    parts: list[str] = []
    for p in paths:
        parts.append(render_template(p, ctx))
    body = "\n\n".join(x.strip() for x in parts if x.strip())
    return body or None
