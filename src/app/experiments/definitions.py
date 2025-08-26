"""Experiment definitions, dependency spec & resolution utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DependencySpec:
    experiment: str
    select: str = "output"  # path expression relative to record root
    transform: str | None = None  # module:function path
    transform_config: dict[str, Any] | None = None


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    description: str | None = None
    model: str = "gpt-4o-mini"
    provider: str = "openai"  # openai|azure
    azure_api_version: str | None = None  # optional per-experiment override
    azure_endpoint: str | None = None  # optional per-experiment endpoint override
    schema_model: str | None = None
    system_prompt: str = "prompts/system/extraction.txt"
    user_prompt: str = "prompts/user/extraction.txt"
    instructions: list[str] | str | None = None  # list or glob pattern
    dataset: str | None = None  # path to dataset (json/jsonl) relative to experiment dir
    temperature: float = 0.0
    max_output_tokens: int = 512
    depends_on: list[DependencySpec] = field(default_factory=list)
    # name template for output directories (variables: experiment, timestamp)
    output_template: str = "{experiment}/{timestamp}"

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        for k, v in overrides.items():
            if hasattr(self, k):  # shallow only
                setattr(self, k, v)


@dataclass(slots=True)
class ResolvedExperiment:
    spec: ExperimentSpec
    root_dir: Path  # directory containing the experiment spec files
    registry_dir: Path  # top-level experiments root (post-restructure)
    system_prompt_path: Path
    user_prompt_path: Path
    instruction_paths: list[Path]
    dataset_path: Path | None

    def to_config_snapshot(self) -> dict[str, Any]:  # persisted in manifest
        # Some experiments (e.g. python_extraction) reuse prompts located in a sibling
        # experiment directory. After the refactor we allow that by storing a registry-
        # relative path when the prompt file is not under the experiment root.
        def _rel_or_abs(p: Path) -> str:
            try:
                return str(p.relative_to(self.root_dir))
            except ValueError:
                # Fallback to registry-relative (stable across machines) if possible
                try:
                    return str(p.relative_to(self.registry_dir))
                except ValueError:
                    # Try project src root (parent of registry) for shared assets like datasets
                    try:
                        return str(p.relative_to(self.registry_dir.parent))
                    except ValueError:
                        return str(p)

        def _norm(path_str: str | None) -> str | None:
            if path_str is None:
                return None
            return path_str.replace("\\", "/")

        return {
            "name": self.spec.name,
            "description": self.spec.description,
            "model": self.spec.model,
            "provider": self.spec.provider,
            "azure_api_version": self.spec.azure_api_version,
            "azure_endpoint": self.spec.azure_endpoint,
            "schema_model": self.spec.schema_model,
            "system_prompt": _norm(_rel_or_abs(self.system_prompt_path)),
            "user_prompt": _norm(_rel_or_abs(self.user_prompt_path)),
            "instructions": [_norm(_rel_or_abs(p)) for p in self.instruction_paths],
            "dataset": _norm(_rel_or_abs(self.dataset_path)) if self.dataset_path else None,
            "temperature": self.spec.temperature,
            "max_output_tokens": self.spec.max_output_tokens,
            "depends_on": [
                {
                    "experiment": d.experiment,
                    "select": d.select,
                    "transform": d.transform,
                    "transform_config": d.transform_config,
                }
                for d in self.spec.depends_on
            ],
            "output_template": self.spec.output_template,
        }


TransformFn = Callable[[list[dict[str, Any]], dict[str, Any] | None], list[dict[str, Any]]]


def parse_dependency(obj: dict[str, Any]) -> DependencySpec:
    t_field = obj.get("transform")
    transform = None
    t_cfg: dict[str, Any] | None = None
    if isinstance(t_field, str):
        transform = t_field
    elif isinstance(t_field, dict):
        transform = t_field.get("fn")
        t_cfg = t_field.get("config")
    return DependencySpec(
        experiment=obj["experiment"],
        select=obj.get("select", "output"),
        transform=transform,
        transform_config=t_cfg,
    )


def coerce_experiment_dict(d: dict[str, Any]) -> ExperimentSpec:
    # Convert depends_on list of dicts
    deps = [parse_dependency(x) for x in d.get("depends_on", [])]
    return ExperimentSpec(
        name=d["name"],
        description=d.get("description"),
        model=d.get("model", "gpt-4o-mini"),
        provider=d.get("provider", "openai"),
        azure_api_version=d.get("azure_api_version"),
        azure_endpoint=d.get("azure_endpoint"),
        schema_model=d.get("schema_model"),
        system_prompt=d.get("system_prompt", "prompts/system/extraction.txt"),
        user_prompt=d.get("user_prompt", "prompts/user/extraction.txt"),
        instructions=d.get("instructions"),
        dataset=d.get("dataset"),
        temperature=float(d.get("temperature", 0.0)),
        max_output_tokens=int(d.get("max_output_tokens", 512)),
        depends_on=deps,
        output_template=d.get("output_template", "{experiment}/{timestamp}"),
    )
