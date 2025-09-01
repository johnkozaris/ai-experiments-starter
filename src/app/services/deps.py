"""Dependency resolution (topological ordering) for app experiments."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.services.spec_loader import load_experiment_spec


def _load_spec(path: Path):  # lightweight helper
    return load_experiment_spec(path / "experiment.yaml")


def topo_sort(names: Sequence[str], *, root: Path) -> list[str]:
    # Build graph from only reachable experiments
    visited: set[str] = set()
    order: list[str] = []
    cache: dict[str, Path] = {}
    for d in root.iterdir():
        if d.is_dir() and (d / "experiment.yaml").exists():
            cache[d.name] = d

    def visit(n: str):
        if n in order:
            return
        if n in visited:  # cycle
            raise RuntimeError(f"Cycle detected involving {n}")
        visited.add(n)
        if n not in cache:
            raise ValueError(f"Experiment '{n}' not found under {root}")
        spec = _load_spec(cache[n])
        for dep in spec.depends_on:
            visit(dep.experiment)
        order.append(n)

    for name in names:
        visit(name)
    return order


__all__ = ["topo_sort"]
