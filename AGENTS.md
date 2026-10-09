# AGENTS.md

## Overview

A learning project: implement GEPA (reflective prompt optimisation) from scratch on a support-ticket queue classifier, then compare it with DSPy's implementation. See [README.md](README.md) for the task, stages and rules.

## Structure

| Path | What it holds |
| --- | --- |
| `gepa_toy/` | Reusable package code |
| `gepa_toy/config.py` | Data paths and shared settings |
| `gepa_toy/data_processing/` | TSV helpers, ticket formatting, near-duplicate detection |
| `gepa_toy/zero_shot.py` | Zero-shot classification prompt |
| `gepa_toy/model.py` | OpenRouter call (`ModelConfig`, `classify`) and strict parsing of the JSON reply |
| `gepa_toy/evaluate.py` | Runs a split through the model, scores it (per-ticket `gepa_score`, per-field metrics) and saves the run |
| `scripts/` | Entry points: download, prepare and split the data; `evaluate.py` runs an evaluation |
| `data/customer_support_tickets/` | Raw and generated data (mostly gitignored) |
| `results/` | One JSON file per evaluation run: metadata, predictions, results (gitignored) |
| `docs/` | Contains artefacts relating to design and implementation decisions  |

## Coding standards

- Use `uv` for everything: `uv add <pkg>` for dependencies, `uv run <script>` to run code.
- Put reusable logic in `gepa_toy/`.
- Take paths from `gepa_toy/config.py`; don't hard-code them.
- After changing code, run `uvx ruff format` and `uvx ruff check --fix`, and fix anything left.
