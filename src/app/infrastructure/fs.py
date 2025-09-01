"""File-system helpers (JSON / JSONL) isolated from domain logic."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson


def write_json(path: str | Path, data: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    return p


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("ab") as f:
        for rec in records:
            f.write(orjson.dumps(rec) + b"\n")
    return p


def append_jsonl_line(path: str | Path, record: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("ab") as f:
        f.write(orjson.dumps(record) + b"\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    with p.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(orjson.loads(line))
    return out
