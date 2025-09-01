"""Dataset loading (JSON / JSONL) for new architecture."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import orjson


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        out: list[dict[str, Any]] = []
        with path.open("rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(orjson.loads(line))
        return out
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return list(data)
        if isinstance(data, dict):
            return [data]
        raise ValueError("JSON dataset must be object or list")
    raise ValueError("Dataset must be .json or .jsonl")
