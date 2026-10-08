# AGENTS.md

## Overview

A learning project: implement GEPA (reflective prompt optimisation) from scratch on a support-ticket queue classifier, then compare it with DSPy's implementation. See [README.md](README.md) for the task, stages and rules.

## Structure

| Path | What it holds |
| --- | --- |
| `gepa_toy/` | Reusable package code |
| `gepa_toy/config.py` | Data paths and shared settings |
| `gepa_toy/data_processing/` | TSV helpers, ticket formatting, near-duplicate detection |
| `scripts/` | One-off entry points: download, prepare and split the data |
| `data/customer_support_tickets/` | Raw and generated data (mostly gitignored) |
| `docs/` | Contains artefacts relating to design and implementation decisions  |

## Coding standards

- Use `uv` for everything: `uv add <pkg>` for dependencies, `uv run <script>` to run code.
- Put reusable logic in `gepa_toy/`. Keep scripts to settings and a `main()`.
- Take paths from `gepa_toy/config.py`; don't hard-code them.
- After changing code, run `uvx ruff format` and `uvx ruff check --fix`, and fix anything left.
