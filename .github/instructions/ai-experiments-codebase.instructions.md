depends_on:
  - experiment: acq_yaml
    select: output
    transform:
      fn: experiments.some_exp.transforms.build_dataset:prep
      config:
        min_items: 1
```
# Experiment Harness – Engineering Guide

Authoritative guide for contributing to the Experiment Harness. Follow exactly. All runtime logic uses a layered architecture under `src/app/` with experiment registry under `src/experiments/`.

## 1. Tech Stack
- Python 3.13 (managed via `uv`)
- Key libs: typer, rich, jinja2, pydantic, openai (incl. Azure), orjson, tqdm, pyyaml, python-dotenv
- Lint/format: Ruff; Type checking: mypy (strict where practical)
- No notebooks required for core runtime; experiments are file based

## 2. Layered Architecture
| Layer | Namespace | Responsibilities |
|-------|-----------|------------------|
| Domain | `app.domain.models` | Pure Pydantic contracts: specs, resolved experiment, prompts snapshot, record result, metrics, analytics. No I/O or side-effects. |
| Infrastructure | `app.infrastructure.*` | LLM adapters (OpenAI/Azure), client facade, cost estimation, dataset loading, artifact persistence, templating (Jinja), logging helpers. |
| Services | `app.services.*` | Orchestration: spec loading, resolver, prompt building, variable analysis, record selection parsing, execution loop, runner, per-experiment analytics, global analytics. |
| CLI | `app.cli` | Typer commands: list, show, run, chain run, eval run. |

Principles:
- Domain layer is stability boundary; other layers depend inward only.
- Services are thin coordinators delegating to infra utilities.
- Adapters isolate provider-specific behaviors and normalize responses.

## 3. Core Concepts
- Experiment Spec (`ExperimentSpec`): declarative config (model, provider, prompts, dataset path, schema model path, dependencies, variables, selection, params).
- Resolved Experiment (`ResolvedExperiment`): spec + absolute paths + loaded dataset + optional schema snapshot (`model_schema_json`) + imported schema class.
- Record Execution: exactly one LLM call per selected record (no hidden retries) for deterministic cost metrics.
- Structured Output: if schema provided we request structured parse; fallback to plain text flagged via `parse_fallback` at record level.

## 4. Prompts & Templates
- Multi-role messages: system, optional developer, user.
- Instruction snippets concatenated and passed via Responses `instructions` parameter.
- All prompt files treated as Jinja templates; context includes dataset row keys plus `record` (row alias) and (when chaining) `upstream_output`.
- Variable analysis warns (non-fatal) about missing or unused declared variables (skipped for chained runs).

## 5. Datasets & Selection
- Supported dataset files: `.jsonl` or `.json` (list of objects).
- Selection string grammar: ranges & commas (e.g. `1-5,10,12-14`) -> 1-based incoming, converted to 0-based indices.
- When chaining and downstream spec omits `record_selection`, all upstream records are processed.

## 6. Chaining & Dependencies
- Declare dependencies via `depends_on` list; each entry selects a path from upstream record outputs (dot traversal from root of per-record result object: typically `output` or nested `output.field`).
- Selected value rules:
  * Dict -> single derived record
  * List[primitive] -> each primitive wrapped as `{ "value": primitive }`
  * List[dict] -> each dict as record
  * Primitive -> `{ "value": primitive }`
- Optional transform function `experiments.<exp>.transforms.<module>:<fn>` returning a new list of records (pure, deterministic, no network I/O).
- Injected records override any local dataset path for the downstream experiment.

## 7. Execution & Logging
- Each selected record => single `LLMClient.generate` call.
- Per-request plain-text log: `logs/llm_requests.txt` containing a block with: record index, model/provider, each message (role + content), instructions (if present), output text, and structured_output JSON (if present). No metrics or token counts in this log.
- Pacing: Minimal necessary (no artificial sleeps unless explicitly added for rate limiting in future enhancements).

## 8. Artifacts Layout
Per run directory: `src/output/<experiment>/<timestamp>/`
1. `experiment_manifest.json` – run metadata (config snapshot + metrics + cost estimate).
2. `experiment_results.jsonl` – newline-delimited slim objects: `{index, output_text, structured_output}`.
3. `recordN/input_manifest.json` – original input row (1-based N).
4. `recordN/result.json` – slim per-record result (mirrors JSONL line for that index).
5. `prompts_used.json` – snapshot: system, user template, developer (optional), instructions list.
6. `logs/llm_requests.txt` – raw request/response blocks.
7. `runs_summary.jsonl` – append-only per-run summary (tokens, latency, success pct, refusals, parse fallbacks, cost).
8. `analytics.json` – aggregated longitudinal experiment analytics.
9. `evaluation.json` – produced by eval pipeline (post-run, additive only).
10. `latest.txt` – pointer to latest run timestamp.
11. `metrics.json` – per-run snapshot (duplicate key metrics for quick inspection).

## 9. Cost & Pricing
- Pricing registry in `app.infrastructure.cost` (and/or `app.utils.pricing` legacy) provides per‑million token input/output rates.
- Runner estimates cost with prompt/completion token totals (per run) and writes into manifest + run summary (`est_cost_total`). Unknown models => null costs.

## 10. Structured Output
- `schema_model` dotted path imported dynamically; JSON schema snapshot stored as `model_schema_json` on resolved experiment object (for reproducibility, optional embed in manifest if needed in future).
- Record contains both `output_text` (raw text) and `structured_output` (dict) when validation succeeds; if refused or fallback occurs, `structured_output` may be null.

## 11. CLI Commands (Typer)
- `exp list` – list experiment names.
- `exp show <name>` – show spec & prompt paths.
- `exp run <name> [--limit N] [--dry-run] [--override key=value]*` – run an experiment.
- `exp chain run <a> <b> ...` – sequentially chain experiments (outputs feed next inputs).
- `exp eval run <run_dir>` – compute evaluation metrics into `evaluation.json`.

Overrides: scalar fields only (temperature, max_output_tokens, etc.). They are recorded in the run manifest parameters.

## 12. Evaluation
- Reads `experiment_results.jsonl` (or per-record `result.json` as fallback) to compute validation stats and (optionally) field coverage for structured outputs.
- Writes `evaluation.json` (never mutates existing artifacts).

## 13. Analytics
- `runs_summary.jsonl` appended each run (one line per run).
- `metrics.json` written inside each run directory (single-run metrics convenience view).
- `analytics.json` rebuilt fully each run (longitudinal per‑experiment aggregates: tokens, success, latencies, refusals, parse fallbacks, extremes, cost aggregates, average cost).

## 14. Coding Standards
You MUST:
- Use relative imports within project namespaces.
- Keep functions small, typed, side-effect minimal.
- Avoid silent failure (log at debug if suppressing, otherwise raise).
- Maintain artifact schemas; any change requires explicit coordinated update here.
- Add Typer subcommands in `app.cli` for new CLI features.
- Keep transforms deterministic & pure.
- Update this file when adding user-visible behaviors.

You MUST NOT:
- Commit secrets or credentials.
- Introduce blocking sleeps in hot loops.
- Hardcode absolute paths.
- Add duplicate logic present in another layer (prefer reuse).
- Expand per-record artifact beyond defined slim shape without justification.

## 15. Adding a New Experiment
1. Create `src/experiments/<name>/`.
2. Add prompts:
   - `prompts/system/<file>.txt`
   - `prompts/user/<file>.jinja`
   - Optional: `prompts/developer/<file>.txt`, `prompts/instructions/*.txt`.
3. Add dataset file (relative path) or rely on dependencies.
4. Add `experiment.yaml` (or a Python `experiment.py` exposing `get_experiment()` returning `ExperimentSpec`).
5. (Optional) Add transforms under `transforms/` (pure functions).
6. Validate: `uv run exp show <name>` then run: `uv run exp run <name>`.

## 16. Structured Output Patterns
- Shared schema: place model in `experiments/<name>/schemas/` unless intended for broad reuse.
- Experiment-specific: place in `experiments/<name>/schemas/`.
- Validation errors cause `structured_output` to be null (future: store errors if needed).

## 17. Environment & Config
- `.env` for API keys & Azure endpoint/version.
---
