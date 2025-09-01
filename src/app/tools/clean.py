"""Clean caches, __pycache__, and (optionally) experiment outputs for app runtime.

Usage (examples):
  uv run python -m app.tools.clean                 # remove caches only
  uv run python -m app.tools.clean --outputs       # also remove all experiment run outputs
  uv run python -m app.tools.clean --experiment acq_yaml  # remove outputs for a single experiment
  uv run python -m app.tools.clean --outputs --keep-latest  # keep latest run per experiment

Flags:
  --outputs / -o       Remove directories under src/output/* (scoped by --experiment if provided)
  --experiment / -e    Name of a single experiment to scope output deletion
    --keep-latest        Keep directory whose name matches latest.txt per experiment (if present)
  --yes / -y           Do not prompt for confirmation (force)
"""
from __future__ import annotations

import argparse
import contextlib
import shutil
from collections.abc import Iterable
from pathlib import Path

CACHE_DIRS: list[str] = [
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".cache",
]

OUTPUT_ROOT = Path("src/output")


def rm(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        with contextlib.suppress(Exception):  # type: ignore[name-defined]
            path.unlink()


def iter_experiment_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for child in root.iterdir():
        if child.is_dir():
            yield child


def delete_outputs(experiment: str | None, keep_latest: bool) -> list[str]:
    removed: list[str] = []
    if not OUTPUT_ROOT.exists():
        return removed
    targets: Iterable[Path]
    if experiment:
        exp_dir = OUTPUT_ROOT / experiment
        if not exp_dir.exists():
            return removed
        targets = [exp_dir]
    else:
        targets = iter_experiment_dirs(OUTPUT_ROOT)
    for exp_dir in targets:
        latest_name: str | None = None
        if keep_latest:
            latest_file = exp_dir / "latest.txt"
            if latest_file.exists():
                latest_name = latest_file.read_text(encoding="utf-8").strip()
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if keep_latest and run_dir.name == latest_name:
                continue
            shutil.rmtree(run_dir, ignore_errors=True)
            removed.append(str(run_dir))
        # remove analytics if all runs gone
        if not any(p.is_dir() for p in exp_dir.iterdir()):
            for f in ["analytics.json", "runs_summary.jsonl", "latest.txt"]:
                fp = exp_dir / f
                if fp.exists():
                    fp.unlink(missing_ok=True)  # type: ignore[arg-type]
            # remove experiment dir if empty
            if not any(exp_dir.iterdir()):
                exp_dir.rmdir()
    # global analytics cleanup
    ga = OUTPUT_ROOT / "analytics_all.json"
    if ga.exists():
        ga.unlink()
    return removed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Clean caches and (optionally) outputs for app runtime"
    )
    parser.add_argument(
        "--outputs",
        "-o",
        action="store_true",
        help="Remove experiment output directories",
    )
    parser.add_argument("--experiment", "-e", help="Target single experiment outputs only")
    parser.add_argument(
        "--keep-latest",
        action="store_true",
        help="Keep the latest run (per latest.txt)",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args(argv)

    # Caches
    for d in CACHE_DIRS:
        rm(Path(d))
    for pyc in list(Path(".").rglob("__pycache__")):
        shutil.rmtree(pyc, ignore_errors=True)

    removed_outputs: list[str] = []
    if args.outputs:
        if not args.yes:
            target_label = args.experiment or "ALL experiments"
            confirm = input(f"Delete outputs for {target_label}? (y/N): ").strip().lower()
            if confirm != "y":
                print("[clean] Aborted output deletion")
            else:
                removed_outputs = delete_outputs(args.experiment, args.keep_latest)
        else:
            removed_outputs = delete_outputs(args.experiment, args.keep_latest)

    print("[clean] Removed caches")
    if removed_outputs:
        print(f"[clean] Deleted {len(removed_outputs)} run directories")
    else:
        if args.outputs:
            print("[clean] No outputs deleted")


if __name__ == "__main__":  # pragma: no cover
    main()
