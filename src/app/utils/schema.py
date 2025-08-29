from __future__ import annotations

from collections.abc import MutableMapping
from importlib import import_module
from typing import Any

from pydantic import BaseModel


def import_model(path: str) -> type[BaseModel]:
    module_path, class_name = path.rsplit(".", 1)
    module = import_module(module_path)
    model_cls = getattr(module, class_name)
    if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):  # type: ignore[arg-type]
        raise TypeError(f"{path} is not a Pydantic BaseModel subclass")
    return model_cls


def _enforce_required_and_no_extra(obj: MutableMapping[str, Any]) -> None:
    """Recursively enforce OpenAI structured output constraints.

    - Every object: additionalProperties = False
    - Every property listed in required (even if nullable -> union with null)
    - Recurse into nested objects / array item schemas / anyOf branches
    """
    if obj.get("type") == "object":
        props = obj.get("properties", {}) or {}
        if isinstance(props, dict):
            # Required must include ALL property keys
            keys = list(props.keys())
            obj["required"] = keys
            obj["additionalProperties"] = False
            for p_schema in props.values():
                if isinstance(p_schema, dict):
                    _recurse(p_schema)
    # Arrays
    if obj.get("type") == "array" and isinstance(obj.get("items"), dict):
        _recurse(obj["items"])  # type: ignore[arg-type]
    # anyOf branches
    if isinstance(obj.get("anyOf"), list):
        for branch in obj["anyOf"]:  # type: ignore[index]
            if isinstance(branch, dict):
                _recurse(branch)


def _recurse(node: MutableMapping[str, Any]) -> None:
    _enforce_required_and_no_extra(node)


def model_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Produce a schema object compliant with OpenAI / Azure structured outputs.

    We take the Pydantic-generated JSON schema and then:
    * Force all object properties to be required (spec requires this; nullable handled via union).
    * Set additionalProperties: false on every object.
    * Leave $defs and $ref structure intact (supported by platform) to reduce size.
    """
    raw = model_cls.model_json_schema()
    # Root adjustments
    _recurse(raw)
    # Root object itself (top-level may have title we can safely drop)
    raw.pop("title", None)
    # Wrap in the response_format envelope expected by client.generate
    return {"name": model_cls.__name__, "schema": raw, "strict": True}
