"""Discovery & resolution of experiment specs from experiment directories."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import yaml  # type: ignore

from .definitions import ExperimentSpec, ResolvedExperiment, coerce_experiment_dict

REGISTRY_ROOT = Path(__file__).resolve().parents[2] / "experiments"


def discover_registry(root: Path | None = None) -> list[Path]:
    r = root or REGISTRY_ROOT
    if not r.exists():
        return []
    return [p for p in r.iterdir() if p.is_dir()]


def load_experiment_dir(dir_path: Path) -> ExperimentSpec | None:
    # Python definition has precedence
    py_path = dir_path / "experiment.py"
    if py_path.exists():
        # Build a clean package path: experiments.<dir>.experiment
        # The experiments package is top-level (declared in build config) so we avoid
        # fragile path slicing.
        module_name = f"experiments.{dir_path.name}.experiment"
        mod = import_module(module_name)  # may raise ImportError which Typer will show
        if hasattr(mod, "get_experiment"):
            obj = mod.get_experiment()  # type: ignore
            if isinstance(obj, ExperimentSpec):
                return obj
            if isinstance(obj, dict):
                return coerce_experiment_dict(obj)  # type: ignore
            raise TypeError("get_experiment() must return ExperimentSpec or dict")
    yml_path = dir_path / "experiment.yaml"
    if yml_path.exists():
        data = yaml.safe_load(yml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):  # pragma: no cover
            raise ValueError("experiment.yaml root must be mapping")
        return coerce_experiment_dict(data)
    return None


def load_all_experiments(root: Path | None = None) -> dict[str, tuple[ExperimentSpec, Path]]:
    out: dict[str, tuple[ExperimentSpec, Path]] = {}
    for d in discover_registry(root):
        spec = load_experiment_dir(d)
        if spec is None:
            continue
        out[spec.name] = (spec, d)
    return out


def resolve_instructions(spec: ExperimentSpec, root_dir: Path) -> list[Path]:
    ins = spec.instructions
    if ins is None:
        return []
    paths: list[Path] = []
    if isinstance(ins, str):
        # treat as glob
        paths = sorted(root_dir.glob(ins))
    else:
        for frag in ins:
            if any(ch in frag for ch in ["*", "?", "["]):
                paths.extend(sorted(root_dir.glob(frag)))
            else:
                p = root_dir / frag
                if p.exists():
                    paths.append(p)
    return paths


def resolve_experiment(
    name: str,
    overrides: dict[str, Any] | None = None,
    root: Path | None = None,
) -> ResolvedExperiment:
    all_specs = load_all_experiments(root)
    if name not in all_specs:
        raise ValueError(f"Experiment '{name}' not found in registry")
    spec, dir_path = all_specs[name]
    if overrides:
        spec.apply_overrides(overrides)
    system_prompt_path = (dir_path / spec.system_prompt).resolve()
    user_prompt_path = (dir_path / spec.user_prompt).resolve()
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt not found: {system_prompt_path}")
    if not user_prompt_path.exists():
        raise FileNotFoundError(f"User prompt not found: {user_prompt_path}")
    instruction_paths = resolve_instructions(spec, dir_path)
    dataset_path = (dir_path / spec.dataset).resolve() if spec.dataset else None
    return ResolvedExperiment(
        spec=spec,
        root_dir=dir_path,
        registry_dir=(root or REGISTRY_ROOT),
        system_prompt_path=system_prompt_path,
        user_prompt_path=user_prompt_path,
        instruction_paths=instruction_paths,
        dataset_path=dataset_path,
    )
