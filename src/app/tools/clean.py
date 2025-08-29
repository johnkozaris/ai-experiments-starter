"""Clean build / analysis caches and __pycache__ directories (packaged version).

Invoked via console script `exp-clean`.
"""

from __future__ import annotations

import logging
import shutil
from contextlib import suppress
from pathlib import Path

CACHE_DIRS = [
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".cache",
]


def _rm(path: Path) -> None:  # best-effort
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        with suppress(Exception):
            path.unlink()


def main() -> None:
    # Operate relative to current working directory (project root when invoked)
    for d in CACHE_DIRS:
        _rm(Path(d))
    for pyc in list(Path(".").rglob("__pycache__")):
        shutil.rmtree(pyc, ignore_errors=True)
    logging.getLogger(__name__).info("Caches removed")
    print("[clean] Removed caches & __pycache__ directories")


if __name__ == "__main__":  # pragma: no cover
    main()
