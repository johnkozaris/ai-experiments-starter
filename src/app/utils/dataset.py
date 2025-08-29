from __future__ import annotations

import csv
import json
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

from .io import write_jsonl

"""Generic, experiment-agnostic dataset helpers.

Keep utilities here narrowly focused and free of domain / experiment semantics.
These helpers intentionally avoid adding opinionated transformation logic; callers
layer domain parsing / enrichment inside experiment code.
"""

JsonLike = dict[str, Any]
Row = dict[str, Any]


def ensure_max_csv_field_size(target: int | None = None) -> None:
    """Best-effort raise the CSV parser field size limit.

    Silent if platform refuses (e.g. narrow builds). Callers can ignore failures.
    """
    with suppress(Exception):  # pragma: no cover - defensive
        csv.field_size_limit(target or sys.maxsize)


def read_text_file(path: str | Path, *, strip: bool = True, default: str = "") -> str:
    """Read a UTF-8 text file returning default on failure."""
    p = Path(path)
    try:
        data = p.read_text(encoding="utf-8")
        return data.strip() if strip else data
    except Exception:  # pragma: no cover - defensive
        return default


def load_json_file(path: str | Path, *, default: Any = None) -> Any:
    """Load JSON returning default on failure."""
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive
        return default


def iter_files(root: str | Path, *, suffixes: Iterable[str] | None = None) -> Iterator[Path]:
    """Iterate immediate children of root filtered by suffix list (case-insensitive)."""
    root_path = Path(root)
    wanted = {s.lower() for s in (suffixes or [])}
    for p in root_path.iterdir():
        if not wanted or p.suffix.lower() in wanted:
            yield p


def csv_rows(path: str | Path, *, encoding: str = "utf-8-sig") -> Iterator[Row]:
    """Yield CSV rows as dictionaries (utf-8 with BOM tolerant by default)."""
    with Path(path).open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def csv_to_records(
    path: str | Path,
    *,
    transform: Callable[[Row], Row] | None = None,
) -> list[Row]:
    """Return list of row dicts, optionally transformed.

    transform receives the raw row dict and must return the (possibly) mutated row.
    """
    out: list[Row] = []
    for r in csv_rows(path):
        out.append(transform(r) if transform else r)
    return out


def csv_files_to_jsonl(
    csv_paths: Iterable[str | Path],
    out_path: str | Path,
    *,
    transform: Callable[[Row], Row] | None = None,
    sort_key: Callable[[Row], Any] | None = None,
) -> Path:
    """Aggregate multiple CSV files into a single JSONL file.

    Parameters
    ----------
    csv_paths: iterable of file paths
    out_path: destination JSONL path
    transform: optional per-row transformer
    sort_key: optional key function for deterministic ordering
    """
    records: list[Row] = []
    for p in csv_paths:
        records.extend(csv_to_records(p, transform=transform))
    if sort_key:
        records.sort(key=sort_key)
    write_jsonl(out_path, records)
    return Path(out_path)


# ----------------------- lightweight content helpers ----------------------- #

def truncate_text(text: str, *, max_len: int, marker: str = "\n...[truncated]") -> str:
    """Truncate text to max_len characters appending marker if truncated."""
    if len(text) <= max_len:
        return text
    # Leave room for marker length
    slice_len = max(0, max_len - len(marker))
    return text[:slice_len] + marker


def summarize_messages(
    messages: list[dict[str, Any]], *, head: int = 2, tail: int = 2, max_chars: int = 4000
) -> str:
    """Generate a concise head/tail summary of a message list.

    Each message expected to have 'role' and 'text'. Non-conforming entries skipped.
    """
    if not messages:
        return ""
    if len(messages) <= head + tail:
        seq = messages
    else:
        seq = [*messages[:head], {"role": "...", "text": "..."}, *messages[-tail:]]
    out = "\n".join(
        f"[{m.get('role')}] {m.get('text','')}" for m in seq if m.get("text")
    )
    return out[:max_chars]


def find_steps_with_empty_args(
    steps: list[dict[str, Any]],
    *,
    arg_key: str = "arguments",
    id_keys: tuple[str, str] = ("step_id", "taskDialogId"),
    empties: tuple[Any, ...] = ("", None, "{}"),
    max_ids: int = 30,
) -> list[str]:
    """Return list of step identifiers whose argument mapping is empty or placeholders."""
    out: list[str] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        args = s.get(arg_key) or {}
        if not args or all(v in empties for v in args.values()):
            for k in id_keys:
                sid = s.get(k)
                if sid:
                    out.append(str(sid))
                    break
        if len(out) >= max_ids:
            return out[:max_ids]
    return out
