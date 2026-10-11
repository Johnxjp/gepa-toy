# AGENTS.md

## Overview

A learning project: implement GEPA (reflective prompt optimisation) from scratch on a support-ticket queue classifier, then compare it with DSPy's implementation. See [README.md](README.md) for the task, stages and rules.

## Structure

| Path | What it holds |
| --- | --- |
| `gepa_toy/` | Reusable package code |
| `gepa_toy/config.py` | Data paths, shared settings, and the only place models are named (`TASK_MODEL`, `REFLECTION_MODEL`) |
| `gepa_toy/data_processing/` | TSV helpers, ticket formatting, near-duplicate detection |
| `gepa_toy/zero_shot.py` | Zero-shot classification prompt and `SEED_INSTRUCTION`, the text GEPA starts from |
| `gepa_toy/model.py` | `Model`: a model's settings plus `complete(prompt)`, the OpenRouter call. Also strict parsing of the JSON reply |
| `gepa_toy/cache.py` | SQLite cache of replies, and `CachedModel`, which wraps a task `Model` with it. The reflection model is never cached |
| `gepa_toy/evaluate.py` | Runs a split through the model, scores it (per-ticket `gepa_score`, per-field metrics) and saves the run. Any API error stops the run |
| `gepa_toy/gepa.py` | The GEPA loop: `GEPA` (one run, `run()` returns the best `Candidate`, `gepa.log` holds the log), `CandidatePool` (selection logic), `Rollout`, and the reflection prompt builder and reply parser |
| `gepa_toy/reflect.py` | The two reflection prompt templates, ours and DSPy's default |
| `gepa_toy/feedback.py` | Per-ticket feedback text for the reflector: score only, or score breakdown with predicted and true labels |
| `gepa_toy/dspy_task.py` | Stage 4: the classifier as a DSPy program, the metric, and `StopOnLMError`, which exits the run on any API error |
| `scripts/` | Entry points: download, prepare and split the data; `evaluate.py` runs an evaluation; `run_dspy.py` runs stage 4 (`--resume <run dir>` continues a stopped run) |
| `tests/` | Unit tests, plus `fake_openrouter.py`, a local stand-in for OpenRouter that sends errors through DSPy's real call path. Run with `uv run python -m unittest discover -s tests` |
| `data/customer_support_tickets/` | Raw and generated data (mostly gitignored) |
| `results/` | One JSON file per evaluation run: metadata, predictions, results; `results/gepa/` holds one folder per GEPA run (gitignored) |
| `docs/` | Contains artefacts relating to design and implementation decisions  |

## Coding standards

- Use `uv` for everything: `uv add <pkg>` for dependencies, `uv run <script>` to run code.
- Put reusable logic in `gepa_toy/`.
- Take paths from `gepa_toy/config.py`; don't hard-code them.
- After changing code, run `uvx ruff format` and `uvx ruff check --fix`, and fix anything left.
