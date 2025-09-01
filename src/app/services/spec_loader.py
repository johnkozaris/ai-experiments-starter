"""Experiment spec loading & basic validation service (new architecture).

This will later replace legacy loader functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.domain.models import DependencySpec, ExperimentSpec

ALLOWED_SPEC_FIELDS = set(ExperimentSpec.model_fields.keys()) | {"dataset", "record_selection"}


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise ValueError("Experiment spec root must be a mapping")
    return data


def _coerce_dependency(raw: dict[str, Any]) -> DependencySpec:
    return DependencySpec(
        experiment=raw["experiment"],
        select=raw.get("select", "output"),
        transform_ref=(raw.get("transform") if isinstance(raw.get("transform"), str) else None),
        transform_config=(
            raw.get("transform", {}).get("config")
            if isinstance(raw.get("transform"), dict)
            else None
        ),
    )


def load_experiment_spec(path: Path) -> ExperimentSpec:
    raw = _load_yaml(path)
    unknown = set(raw.keys()) - ALLOWED_SPEC_FIELDS - {"depends_on"}
    if unknown:
        raise ValueError(f"Unknown spec fields: {sorted(unknown)}")
    dep_objs = [_coerce_dependency(d) for d in raw.get("depends_on", [])]
    raw["depends_on"] = dep_objs
    # dataset legacy key mapping
    if "dataset" in raw and "dataset_path" not in raw:
        raw["dataset_path"] = raw.pop("dataset")
    return ExperimentSpec(**raw)


__all__ = ["load_experiment_spec"]
