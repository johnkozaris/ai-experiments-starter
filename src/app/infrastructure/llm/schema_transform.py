"""Transform Pydantic JSON Schema into OpenAI Structured Outputs compliant schema.

OpenAI Structured Outputs constraints (subset):
  * Root must be an object (not anyOf) and include additionalProperties:false.
  * Every property must be listed in required.
  * To emulate optional, property type must allow null (string|null) etc.
  * Nested objects must also set additionalProperties:false.

We take a Pydantic-generated schema (model_json_schema) and:
  1. For every object: ensure additionalProperties = False.
  2. Record original required set; for properties not in required, wrap their type to allow null
     (e.g., {"type":"string"} -> {"anyOf":[{"type":"string","description":...},{"type":"null"}]} )
     and add them to required.
  3. Recurse into properties, items, $defs / definitions.
Idempotent: re-running will not duplicate wrappers.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def _wrap_nullable(prop: dict[str, Any]) -> dict[str, Any]:
    # If already nullable via anyOf or includes null type, leave in place.
    if "anyOf" in prop:
        return prop
    t = prop.get("type")
    if isinstance(t, list):
        if "null" in t:
            return prop
        return {**prop, "type": [*t, "null"]}
    if isinstance(t, str):
        # Convert to anyOf to preserve metadata cleanly
        desc = prop.get("description")
        base = {k: v for k, v in prop.items() if k != "description"}
        any_of = [base, {"type": "null"}]
        if desc:
            any_of[0]["description"] = desc
        return {"anyOf": any_of}
    return prop


def _recurse(node: dict[str, Any]) -> None:
    if node.get("type") == "object" and isinstance(node.get("properties"), dict):
        node.setdefault("additionalProperties", False)
        props: dict[str, Any] = node["properties"]  # type: ignore[assignment]
        required = set(node.get("required", []))
        # Ensure each property required; wrap nullable for those not originally required.
        for name, prop in list(props.items()):
            if name not in required:
                props[name] = _wrap_nullable(prop)  # may replace
                required.add(name)
            # Recurse inside (after potential wrap)
            target = props[name]
            if isinstance(target, dict):
                if target.get("type") == "object":
                    _recurse(target)
                elif "anyOf" in target:
                    for variant in target["anyOf"]:
                        if isinstance(variant, dict):
                            _recurse(variant)
        node["required"] = list(required)
    # arrays
    if node.get("type") == "array" and isinstance(node.get("items"), dict):
        _recurse(node["items"])  # type: ignore[arg-type]
    # defs
    for defs_key in ("$defs", "definitions"):
        if isinstance(node.get(defs_key), dict):
            for sub in node[defs_key].values():  # type: ignore[index]
                if isinstance(sub, dict):
                    _recurse(sub)


def adapt_schema_for_structured_outputs(schema: dict[str, Any]) -> dict[str, Any]:
    clone = deepcopy(schema)
    _recurse(clone)
    if clone.get("type") == "object":
        clone.setdefault("additionalProperties", False)
        if isinstance(clone.get("properties"), dict):
            props = clone["properties"]  # type: ignore[assignment]
            req = set(clone.get("required", []))
            for k in props:
                req.add(k)
            clone["required"] = list(req)
    return clone


__all__ = ["adapt_schema_for_structured_outputs"]
