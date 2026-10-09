# GEPA implementation plan

Build order for [gepa_design.md](gepa_design.md). Each phase ends with a check from the design's test tables (section 7). A phase is done when its tests pass, `ruff format` and `ruff check` are clean, and an adversarial review has found nothing unproven. Phases in the same wave touch different files and run in parallel.

## Phases

| Phase | Build | Files | Depends on | Check |
| --- | --- | --- | --- | --- |
| 0 | Shared types: `Rollout`, `Task`, `Propose`, `Settings` | `gepa.py` | none | imports |
| A | `build(instruction, subject, body)`, `PLAIN_INSTRUCTION`, `SEED_INSTRUCTION` with the README descriptions in the instruction | `zero_shot.py` | none | `build_prompt` byte-for-byte unchanged; braces in the instruction appear verbatim; every description and label exactly once in the seed prompt |
| B | `evaluate_rows(rows, build, model_config, workers)`, `model.complete`, `queue_confusion` in `summarise` | `evaluate.py`, `model.py` | none | `scripts/evaluate.py` gives the same `results` as before with a stubbed model; confusion counts from hand-made predictions |
| C | SQLite cache keyed by prompt plus model config | `cache.py`, `config.py` | none | hit; miss per config field; transient error not stored; parse failure stored; concurrent same-key writes |
| E | `optimise` with greedy selection: minibatches, budget, accept, skip-perfect, crash-safe log, final pick | `gepa.py` | 0 | fake task and reflector; budget never exceeded; determinism; planted answer with `best` |
| F | `make_reflector`: reflection prompt, extract last fenced block | `reflect.py` | 0 | extraction; no block raises |
| D | Feedback (written and score), `TicketTask` | `ticket_task.py` | A, B, C | feedback case table; score mode never names a label |
| H | Pareto selection with dominated-candidate removal | `gepa.py` | E | hand-written score matrix; planted answer keeps a diverse pool |
| G | `run_gepa.py` stages 1, 2, 3, 3c; test run links to the log | `scripts/run_gepa.py`, `config.py` | D, E, F, H | stage metadata matches the phases table; smoke run at budget 80 |
| I | `summarise_runs.py` | `scripts/summarise_runs.py` | G | mean and spread per stage |
| J | DSPy signature, metric wrapper, `run_dspy.py` | `dspy_task.py`, `scripts/run_dspy.py` | A, D | reasoning flag check; metric wrapper equals our feedback; smoke run |
| K | Integration review and stage 3 smoke run | all | G, I, J | design section 7 smoke checklist |

## Waves

| Wave | Phases | Why |
| --- | --- | --- |
| 1 | A, B, C, E, F | Independent of each other; E and F share only the phase 0 types |
| 2 | D, H | D needs the prompt, in-memory evaluation and cache; H extends the loop |
| 3 | G, J | G wires the stages together; J needs the seed instruction and feedback |
| 4 | I, K | Summary needs run files; final review and smoke run |

Phase 0 is done by hand before wave 1 so E, F and D agree on `Rollout`.

## Rules for every phase

- Tests use fakes and never call a model, apart from the smoke runs.
- `uv run python -m unittest discover -s tests`, then `uvx ruff format` and `uvx ruff check --fix`.
- Adversarial review before the phase is merged: look for ways it breaks, prove each with a failing test.
- No behaviour beyond the design. No logging, retries or validation the design does not ask for.
