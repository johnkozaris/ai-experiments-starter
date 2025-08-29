---
applyTo: '**'
---
# Project Instruction Guide for LLM Assistants

You are assisting on the **Experiment Harness** project. You define, run, chain, and evaluate LLM experiments (OpenAI & Azure OpenAI) producing structured (Pydantic) or free-form outputs. Follow these instructions exactly when generating or modifying code.

## 1. Tech Stack & Conventions

- Python 3.13 (manage with `uv`).
- Packages: `app` (runtime) and `experiments` (registry) under `src/` (configured in `pyproject.toml`).
- Core libs: typer, jinja2, pydantic, openai (Azure/OpenAI), orjson, rich, tqdm, pyyaml, dotenv.
- Style: Ruff for format+lint
- JSONL via orjson; prompts plain text (system/instructions) + Jinja for user prompts.
- Multi-artifact run layout: `experiment_manifest.json`, `experiment_results.jsonl`, per-record `recordN/` folders (each with `input_manifest.json` + `result.json`), plus summary & analytics files.

### Add / Modify Code Guidelines

- Put reusable runtime code in `src/app/*` (clients, utils, experiments, pipelines, schemas, tools).
- Put experiment definitions + prompts only in `src/experiments/<name>/`.
- Avoid circular imports (utilities live in `app.utils`).
- Keep functions small, typed, side-effect minimal.
- Preserve existing public APIs; extend carefully; keep artifact schema stable.

## 2. Folder / Module Overview

```text
src/
  app/
    cli.py                # Root CLI (list/show/run/chain + eval)
    runtime.py            # .env + config bootstrap
    clients/llm_client.py # Unified OpenAI/Azure client + structured parsing
    experiments/
      definitions.py      # ExperimentSpec / DependencySpec dataclasses
      loader.py           # Discover + load specs (YAML or Python)
      runner.py           # Execution loop, templating, API calls, artifact build
    pipelines/
      run_batch.py        # Ad-hoc (no registry) batch command
      evaluate_run.py     # Post-run evaluation (validation & field metrics)
    schemas/               # these are schemas for llm structured outputs based on the way openai expects a schemas to avoid invalid json schema reports for structured outputs
      extraction.py       # schema for  llm structured outputs
    tools/clean.py        # Cache cleanup (exp-clean)
    utils/
      io.py               # JSONL helpers, run dir creation
      logging.py          # Logging setup (rich + file)
      schema.py           # Model import + JSON schema coercion
  experiments/            # New experiments go under here in their own subfolder look at an example experiment for the structure
    acq_yaml/             # example acq experiment
    investor_followup/    # example investor experiment n acq_yaml + transform
    python_extraction/    # example  extraction experiment
  datasets/
    samples.jsonl         # Example dataset other datasets go to raw if pure and processed if we manipulate them
  configs/
    default.yaml          # Optional runtime config
  output/                 # Run artifacts (gitignored)
```
Every experiment subfolder
Place new:
- Generic helpers -> `src/app/utils/`.
- New pipeline (new CLI group) -> `src/app/pipelines/` then register Typer sub-app in `app/cli.py`.
- Experiments -> `src/experiments/<exp_name>/`.

## 3. Experiment Lifecycle

1. You define an `ExperimentSpec` (YAML `experiment.yaml` or Python `experiment.py`).
2. CLI loads all specs (`loader.load_all_experiments`).
3. Resolution yields `ResolvedExperiment` (paths, dataset, prompts, dependency chain).
4. Runner renders user prompt via Jinja with row variables + appended instruction text.
5. `LLMClient.generate` executes API call:
   - If `schema_model`: try Responses parse with Pydantic model; fallback to manual JSON schema.
   - Else: free-form text response.
6. Collect per-record outputs (validation, usage, latency) streaming; write per-record `recordN/result.json`.
7. Append slim result lines to `experiment_results.jsonl` (one JSON object per record).
8. Write `experiment_manifest.json` containing manifest + metrics (no embedded records).
9. Append summary line to `runs_summary.jsonl`; recompute `analytics.json`.
10. Optional evaluation writes `evaluation.json` (does NOT mutate existing artifacts).

Dependency chaining: For each `DependencySpec`, load latest upstream run, select path (dot traversal), optional transform builds new dataset list for downstream experiment.

## 4. Output Artifacts

- `src/output/<exp>/<ts>/experiment_manifest.json` – run config + metrics.
- `src/output/<exp>/<ts>/experiment_results.jsonl` – slim per-record outputs.
- `src/output/<exp>/<ts>/recordN/` – per-record artifacts (`input_manifest.json`, `result.json`).
- `src/output/<exp>/<ts>/prompts_used.json` – prompts snapshot.
- `src/output/<exp>/<ts>/logs/llm_requests.jsonl` – per-request logs.
- `src/output/<exp>/runs_summary.jsonl` – one line per run (lives at experiment root).
- `src/output/<exp>/analytics.json` – aggregated stats (tokens, latency, success rate, rolling window).
- `src/output/<exp>/latest.txt` – latest timestamp pointer.
- `src/output/<exp>/<ts>/evaluation.json` – evaluation metrics (created by eval pipeline).

Legacy artifacts `result.json` / `results.jsonl` should not be written; code may still read them only if explicitly implementing migration logic (avoid adding new dependencies on them).

## 5. CLI & Scripts

Root command: `exp`.

Subcommands:
- `exp list`
- `exp show <name>`
- `exp run <name> [--override k=v]* [--limit N] [--dry-run]`
- `exp chain run <a> <b> ...`
- `exp eval run <run_dir>`

Ad-hoc (no registry) batch: `uv run python -m app.pipelines.run_batch run ...`

Overrides: `--override temperature=0.2 --override max_output_tokens=256` etc. Only scalar fields. Captured in manifest snapshot.

Helper commands (add to uv scripts or run directly):
```bash
uv run exp list
uv run exp run acq_yaml --limit 3
uv run exp chain run acq_yaml investor_followup
uv run exp eval run src/output/acq_yaml/{timestamp}
uv run ruff format .
uv run ruff check .
```

## 6. Adding a New Experiment

1. Create folder `src/experiments/<name>/`.
2. Add prompts:
   - `prompts/system/<file>.txt`
   - `prompts/user/<file>.jinja` (use placeholders for dataset keys)
   - Optional `prompts/instructions/*.txt` snippets.
3. Add or reference dataset (JSONL) relative to experiment directory (or rely on dependencies).
4. Add `experiment.yaml` OR `experiment.py` returning `ExperimentSpec` (`get_experiment`).
5. (Optional) Add transforms under `<exp>/transforms/*.py` with `fn(records, config)`. if transformations are specific to the dataset if you are creating a reusable transformation its better to put it under app utils or split into reusable and specific parts and put it under both
6. After all coding is completed Validate: `uv run exp show <name>` then run: `uv run exp run <name>`.

## 7. Coding Style & Quality

- Run format/lint before commit.
- Strict typing: avoid untyped public functions; use generics where helpful.
- Fail fast with clear exceptions; avoid silent `pass`.
- No network I/O in transforms (pure data reshaping only).
- Keep runner output schema stable.
- Write modern 2025 python 3.13
- use the packages and helpers provided in pyproject.toml to help you code easier. From the packages always use modern way of implementing them.

## 8. Structured Output Extension

Preferred patterns depend on reuse scope:

1. Reusable / shared schema: create Pydantic model file in `src/app/schemas/` and reference via `schema_model: app.schemas.<module>.<Model>`.
2. Experiment-specific schema (only used by a single experiment): place under `src/experiments/<exp_name>/schemas/` and reference via `schema_model: experiments.<exp_name>.schemas.<module>.<Model>`.
3. Runner imports dynamically (no experiment-specific logic) and builds JSON schema + attempts parse API.
4. Validation errors captured per record (`validated` false + `validation_errors`).

## 9. Dependency Dataset Transforms

Selection: `select` traverses dot path from each record root (e.g. `output`, `output.field`). Lists of primitives become `{"value": item}` objects.

Transform forms:
```yaml
depends_on:
  - experiment: acq_yaml
    select: output
    transform:
      fn: experiments.some_exp.transforms.build_dataset:prep
      config:
        min_items: 1
```
Function signature: `def build_dataset(records: list[dict[str, Any]], config: dict|None) -> list[dict]`.

## 10. Environment & Config

- `.env` loaded at bootstrap (OPENAI_API_KEY, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, etc.).
- Optional runtime YAML config: `src/configs/default.yaml` (override path via `EXPERIMENTS_CONFIG`).
- Per-experiment Azure overrides: `azure_api_version`, `azure_endpoint` fields.

## 11. When You Generate Code (MANDATORY RULES)

You MUST:
- Use relative imports inside project (e.g. `from app.utils.io import read_jsonl`) and imports go to the top of the file.
- Use modern python 3.13
- Prefer to use doc comments to document methods or classes instead of inline comments
- Never put experiment specs or prompts under `src/app`.
- Follow current multi-artifact pattern (manifest + aggregate JSONL + per-record folders). Do not reintroduce monolithic `result.json` writes.
- Support both providers; avoid hardcoding endpoints (use env + spec overrides).
- Add Typer sub-apps for new CLI groups and register in `app/cli.py`.
- Keep transforms deterministic / side-effect free.
- Update this instructions file if you add user-facing features.
- Re-use implementation found in the codebase instead of creating new one.
- When you need to create new implementation make sure there are not parallel code implementations or code paths.

You MUST NOT:
- Commit secrets.
- Introduce blocking sleeps in hot loops.
- Hardcode absolute filesystem paths.
- Change existing artifact field names without explicit migration handling.

## 13. Do / Don't

Do:
- Reuse existing utilities before adding new ones.
- Keep schemas minimal, optional fields for uncertain extractions.
- Log debug info instead of printing in library code.

Don't:
- Duplicate logic across runner + batch pipeline.
- Return partial artifacts (always complete JSON objects).

