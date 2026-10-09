"""Stage 4: optimise the ticket instruction with dspy.GEPA, then score the result on test."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from gepa_toy.config import (
    GEPA_RESULTS_DIR,
    PROJECT_ROOT,
    REFLECTION_MODEL,
    RESULTS_DIR,
    SPLITS_DIR,
)
from gepa_toy.data_processing.tsv import read_tsv

# dspy comes through dspy_task, which imports it in the order LiteLLM needs.
from gepa_toy.dspy_task import (
    StopOnLMError,
    build_program,
    candidate_log,
    dspy,
    exit_run,
    make_lm,
    metric,
    predict_row,
    to_example,
)
from gepa_toy.evaluate import save_run, summarise
from gepa_toy.model import ModelConfig
from gepa_toy.utils import git_version

MINIBATCH_SIZE = 5


def main() -> None:
    defaults = ModelConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=1000, help="task-model calls")
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--reflection-model", default=REFLECTION_MODEL)
    # DSPy's reply format is longer than our JSON (design section 8); raise it here only.
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    task_config = ModelConfig(model=args.model, max_tokens=args.max_tokens)
    # Reasoning off: with it on, DeepSeek spends its whole token budget thinking and returns
    # no content (even at 16k tokens). Room to write a whole instruction; 60s would cut it short.
    reflection_config = ModelConfig(
        model=args.reflection_model, reasoning=False, max_tokens=4000, timeout=300.0
    )
    started_at = datetime.now(UTC)
    stem = f"{started_at:%Y%m%d-%H%M%S-%f}_stage4_seed{args.seed}"
    log_path = GEPA_RESULTS_DIR / f"{stem}.json"
    # Gathered before any model calls, so a failure here doesn't waste a paid run.
    metadata = {
        "started_at": started_at.isoformat(),
        "git": git_version(),
        "stage": "4",
        "seed": args.seed,
        "budget": args.budget,
        "minibatch_size": MINIBATCH_SIZE,
        "dspy_version": dspy.__version__,
        "task": {"model": task_config.model_dump(), "feedback": "written"},
        "reflection_model": reflection_config.model_dump(),
    }

    train = read_tsv(SPLITS_DIR / "train.tsv")
    pareto = read_tsv(SPLITS_DIR / "pareto.tsv")
    program = build_program(make_lm(task_config, [StopOnLMError(exit_run)]))
    optimiser = dspy.GEPA(
        metric=metric,
        max_metric_calls=args.budget,
        reflection_lm=make_lm(reflection_config),
        reflection_minibatch_size=MINIBATCH_SIZE,
        candidate_selection_strategy="pareto",
        skip_perfect_score=True,
        use_merge=False,
        track_stats=True,
        num_threads=args.workers,
        seed=args.seed,
        # A fresh directory per run: GEPA resumes from whatever state it finds here.
        log_dir=str(GEPA_RESULTS_DIR / f"{stem}_dspy"),
    )
    start = time.monotonic()
    optimised = optimiser.compile(
        program,
        trainset=[to_example(r) for r in train],
        valset=[to_example(r) for r in pareto],
    )
    metadata["runtime_seconds"] = round(time.monotonic() - start, 2)
    # Test scoring records transient errors instead of stopping, as stages 0 to 3 do.
    optimised.set_lm(make_lm(task_config))

    log = {"metadata": metadata, **candidate_log(optimised.detailed_results)}
    GEPA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2) + "\n")

    test_file = SPLITS_DIR / "test.tsv"
    test_started = datetime.now(UTC)
    start = time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool:
        predictions = list(
            pool.map(lambda r: predict_row(optimised, r), read_tsv(test_file))
        )
    run = {
        "metadata": {
            "started_at": test_started.isoformat(),
            "git": metadata["git"],
            "data_file": str(test_file.relative_to(PROJECT_ROOT)),
            "model": task_config.model_dump(),
            "prompt": optimised.signature.instructions,
            "stage": "4",
            "seed": args.seed,
            "log_path": str(log_path.relative_to(PROJECT_ROOT)),
            "best_candidate": log["best"],
            "runtime_seconds": round(time.monotonic() - start, 2),
        },
        "predictions": predictions,
        "results": summarise(predictions),
    }
    path = save_run(RESULTS_DIR, run)
    print(json.dumps(run["results"], indent=2))
    print(
        f"Best candidate {log['best']} of {len(log['candidates'])},"
        f" {log['calls_used']} calls used"
    )
    print(f"Saved log to {log_path}\nSaved run to {path}")


if __name__ == "__main__":
    main()
