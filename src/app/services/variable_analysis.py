"""Prompt variable analysis (pure, no legacy dependency)."""
from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from rich.console import Console


def collect_template_variables(paths: Iterable[Path]) -> set[str]:
    import re

    var_pattern = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)")
    found: set[str] = set()
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            logging.getLogger(__name__).debug("read fail %s: %s", p, exc)
            continue
        for m in var_pattern.finditer(text):
            found.add(m.group(1))
    return found


def analyze_variables(
    *,
    expected_variables: list[str] | None,
    prompt_paths: list[Path],
    records: list[dict],
    injected_records: bool,
    console: Console,
) -> None:
    if injected_records:
        return
    used = collect_template_variables(prompt_paths)
    dataset_keys: set[str] = set()
    for r in records:
        dataset_keys.update(r.keys())
    if expected_variables:
        unused = set(expected_variables) - used
        if unused:
            console.print(
                f"[yellow]Warning:[/yellow] expected vars unused: {sorted(unused)}"
            )
    missing = {v for v in used if v not in dataset_keys and v not in {"record", "upstream"}}
    if missing:
        console.print(
            f"[yellow]Warning:[/yellow] prompt vars missing in dataset: {sorted(missing)}"
        )

__all__ = ["analyze_variables", "collect_template_variables"]
