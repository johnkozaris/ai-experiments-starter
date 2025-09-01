"""Schema import and JSON schema helper utilities (restored after migration).

This provides dynamic model import given a dotted path, plus safe JSON schema
extraction using Pydantic v2's model_json_schema. Kept intentionally tiny so
other layers can rely on it without pulling extra deps.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

from pydantic import BaseModel


def import_model(dotted: str) -> type[BaseModel]:
    """Import a Pydantic model class from a dotted path.

    Example: ``experiments.agent_recommendation.schemas.models:Output`` or
    ``experiments.agent_recommendation.schemas.models.Output``.
    Both ``:`` and final ``.`` forms are supported for ergonomics.
    """
    if ":" in dotted:
        module_path, attr = dotted.split(":", 1)
    else:
        parts = dotted.split(".")
        module_path, attr = "".join(parts[:-1]) if len(parts) > 1 else dotted, parts[-1]
        module_path = ".".join(parts[:-1])
    if not module_path:
        raise ValueError(f"Invalid schema model path: {dotted}")
    module = import_module(module_path)
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ImportError(f"Attribute '{attr}' not found in module '{module_path}'") from exc
    if not isinstance(obj, type) or not issubclass(obj, BaseModel):
        raise TypeError(f"Imported object '{attr}' is not a Pydantic BaseModel subclass")
    return obj  # type: ignore[return-value]


def model_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Return the JSON schema for a Pydantic model class."""
    return model_cls.model_json_schema()  # type: ignore[no-any-return]
