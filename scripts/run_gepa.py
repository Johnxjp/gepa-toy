"""Run one GEPA stage with one seed, save the log, then score the best prompt on test.

Stage 0 skips optimisation and scores the seed instruction, so every stage's test
run is saved by the same code. Stages follow the phases table in docs/gepa_design.md.
"""

import argparse
import json
from datetime import UTC, datetime

from gepa_toy import config, model
from gepa_toy.cache import Cache
from gepa_toy.config import (
    CACHE_PATH,
    GEPA_RESULTS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    SPLITS_DIR,
)
from gepa_toy.data_processing.tsv import read_tsv
from gepa_toy.evaluate import evaluate, save_run
from gepa_toy.gepa import Settings, optimise
from gepa_toy.model import ModelConfig
from gepa_toy.reflect import DSPY_REFLECTION_PROMPT, REFLECTION_PROMPT, make_reflector
from gepa_toy.ticket_task import TicketTask
from gepa_toy.zero_shot import SEED_INSTRUCTION, build

STAGES = {
    "1": {"feedback": "score", "selection": "best", "reflection_prompt": "ours"},
    "2": {"feedback": "written", "selection": "best", "reflection_prompt": "ours"},
    "3": {"feedback": "written", "selection": "pareto", "reflection_prompt": "ours"},
    "3c": {"feedback": "written", "selection": "pareto", "reflection_prompt": "dspy"},
}
REFLECTION_PROMPTS = {"ours": REFLECTION_PROMPT, "dspy": DSPY_REFLECTION_PROMPT}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["0", *STAGES])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--model", default=ModelConfig().model)
    parser.add_argument("--reflection-model", default=config.REFLECTION_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    task_config = ModelConfig(model=args.model)
    cache = Cache(CACHE_PATH)
    best_text, best_id, log_path = SEED_INSTRUCTION, None, None
    if args.stage != "0":
        stage = STAGES[args.stage]
        reflection_config = model.reflection_config(args.reflection_model)
        task = TicketTask(task_config, stage["feedback"], cache, args.workers)
        # A bad reflection model or key must crash here: inside the loop every proposal
        # failure is logged and the loop carries on until the task budget is spent.
        model.complete("Reply with OK.", reflection_config)
        propose = make_reflector(
            lambda prompt: model.complete(prompt, reflection_config),
            REFLECTION_PROMPTS[stage["reflection_prompt"]],
        )
        settings = Settings(
            args.budget, minibatch_size=5, selection=stage["selection"], seed=args.seed
        )
        started = datetime.now(UTC)
        # Microseconds, like save_run: same stage and seed twice in one second would share a log.
        log_path = (
            GEPA_RESULTS_DIR
            / f"{started:%Y%m%d-%H%M%S-%f}_stage{args.stage}_seed{args.seed}.json"
        )
        GEPA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        log = optimise(
            SEED_INSTRUCTION,
            read_tsv(SPLITS_DIR / "train.tsv"),
            read_tsv(SPLITS_DIR / "pareto.tsv"),
            task,
            propose,
            settings,
            log_path,
            metadata={
                "stage": args.stage,
                "seed": args.seed,
                "reflection_model": reflection_config.model_dump(),
                "reflection_prompt": stage["reflection_prompt"],
            },
        )
        best_id = log["best"]
        best_text = log["candidates"][best_id]["text"]

    run = evaluate(
        SPLITS_DIR / "test.tsv",
        lambda subject, body: build(best_text, subject, body),
        task_config,
        args.workers,
        complete=lambda prompt, cfg: cache.complete(prompt, cfg, model.complete),
    )
    run["metadata"].update(
        stage=args.stage,
        seed=args.seed,
        log_path=None if log_path is None else str(log_path.relative_to(PROJECT_ROOT)),
        best_candidate=best_id,
        budget=args.budget,
    )
    path = save_run(RESULTS_DIR, run)
    print(json.dumps(run["results"], indent=2))
    print(f"Log: {log_path}\nSaved to {path}")


if __name__ == "__main__":
    main()
