"""Runtime context & bootstrap utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:  # pragma: no cover
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from app.utils.logging import setup_logging


@dataclass(slots=True)
class RuntimeConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def model(self) -> str:
        return self.raw.get("model", "gpt-4o-mini")


class AppContext:
    _instance: AppContext | None = None

    def __init__(self, config: RuntimeConfig):
        self.config = config

    @classmethod
    def init(cls, config: RuntimeConfig) -> AppContext:
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def get(cls) -> AppContext:
        if cls._instance is None:
            raise RuntimeError("AppContext not initialized")
        return cls._instance


def load_config(path: Path) -> RuntimeConfig:
    if not path.exists():
        return RuntimeConfig(raw={}, path=path)
    if yaml is None:
        raise RuntimeError("PyYAML not installed; cannot load config")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):  # pragma: no cover
        raise ValueError("Config root must be a mapping")
    return RuntimeConfig(raw=data, path=path)


def bootstrap(force: bool = False) -> AppContext:
    if not force:
        try:
            return AppContext.get()
        except RuntimeError:
            pass
    load_dotenv(override=False)
    setup_logging()
    # Updated default config path after restructure: configs now at src/configs/
    cfg_path = Path(os.getenv("EXPERIMENTS_CONFIG", "src/configs/default.yaml"))
    config = load_config(cfg_path)
    return AppContext.init(config)
