from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import orjson


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    with p.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        for rec in records:
            f.write(orjson.dumps(rec) + b"\n")


def ensure_run_dir(base: str | Path, timestamp: str) -> Path:
    run_dir = Path(base) / timestamp
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir
