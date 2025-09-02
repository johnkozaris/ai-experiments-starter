"""Resolve experiment directory, prompts, dataset and dependencies."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.domain.models import ResolvedExperiment
from app.infrastructure.dataset_loader import load_dataset
from app.services.spec_loader import load_experiment_spec
from app.utils.schema import import_model, model_json_schema  # reuse legacy until migrated


def discover_experiments(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if (d / "experiment.yaml").exists() or (d / "experiment.py").exists():
            out[d.name] = d
    return out


def resolve_experiment(name: str, root: Path) -> ResolvedExperiment:
    mapping = discover_experiments(root)
    if name not in mapping:
        raise ValueError(f"Experiment '{name}' not found in {root}")
    dir_path = mapping[name]
    yaml_path = dir_path / "experiment.yaml"
    py_path = dir_path / "experiment.py"
    if yaml_path.exists():
        spec = load_experiment_spec(yaml_path)
    elif py_path.exists():
        # Dynamic load python experiment spec
        import importlib.util
        mod_name = f"experiments.{dir_path.name}.experiment_dynamic"
        spec_obj = importlib.util.spec_from_file_location(mod_name, py_path)
        if spec_obj is None or spec_obj.loader is None:  # pragma: no cover
            raise ValueError(f"Cannot load python experiment at {py_path}")
        module = importlib.util.module_from_spec(spec_obj)
        try:
            spec_obj.loader.exec_module(module)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Executing experiment module failed: {exc}") from exc
        if not hasattr(module, "get_experiment"):
            raise ValueError(f"Python experiment missing get_experiment(): {py_path}")
        py_spec = module.get_experiment()
        # If legacy field 'dataset' used, map to dataset_path
        if getattr(py_spec, "dataset_path", None) is None and getattr(py_spec, "dataset", None):
            # Map legacy attribute name to current field (best-effort)
            import contextlib
            with contextlib.suppress(Exception):  # pragma: no cover
                py_spec.dataset_path = py_spec.dataset  # type: ignore[attr-defined]
        spec = py_spec
    else:
        raise ValueError(f"No experiment.yaml or experiment.py found in {dir_path}")
    system = (dir_path / spec.system_prompt).resolve() if spec.system_prompt else None
    user = (dir_path / spec.user_prompt).resolve()
    developer = (dir_path / spec.developer_prompt).resolve() if spec.developer_prompt else None
    instruction_paths: list[str] = []
    if spec.instructions is not None:
        if isinstance(spec.instructions, str):
            instruction_paths.append(str((dir_path / spec.instructions).resolve()))
        else:
            for ins in spec.instructions:
                instruction_paths.append(str((dir_path / ins).resolve()))
    dataset_records: list[dict[str, Any]] | None = None
    schema_cls: type[BaseModel] | None = None
    schema_json: dict[str, Any] | None = None
    if spec.dataset_path:
        dataset_records = load_dataset((dir_path / spec.dataset_path).resolve())
    if spec.schema_model:
        try:
            schema_cls = import_model(spec.schema_model)
            # Capture pydantic JSON schema snapshot (top-level dict) for provider usage.
            schema_json = model_json_schema(schema_cls)
        except Exception:  # pragma: no cover - soft fail
            schema_cls = None
            schema_json = None
    return ResolvedExperiment(
        spec=spec,
        root_dir=str(dir_path),
        system_prompt_path=(str(system) if system else None),
        user_prompt_path=str(user),
        developer_prompt_path=str(developer) if developer else None,
        instruction_paths=instruction_paths,
        dataset_records=dataset_records,
        schema_model_path=spec.schema_model,
        model_schema_json=schema_json,
        schema_model_cls=schema_cls,
    )
