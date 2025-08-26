from __future__ import annotations

from importlib import import_module
from typing import Any

from pydantic import BaseModel


def import_model(path: str) -> type[BaseModel]:
    module_path, class_name = path.rsplit(".", 1)
    module = import_module(module_path)
    model_cls = getattr(module, class_name)
    if not issubclass(model_cls, BaseModel):  # type: ignore[arg-type]
        raise TypeError(f"{path} is not a Pydantic BaseModel subclass")
    return model_cls  # type: ignore[return-value]


def model_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    schema = model_cls.model_json_schema()
    title = model_cls.__name__
    return {
        "name": title,
        "schema": {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
            "additionalProperties": False,
        },
        "strict": True,
    }
