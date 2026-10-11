"""Stage 4: optimise the ticket instruction with dspy.GEPA, then score the result on a split.

Any model API error stops the run. Rerun with `--resume <run dir>` to continue from the
last checkpoint; calls that already succeeded come back from DSPy's cache.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from gepa_toy.config import (
    GEPA_RESULTS_DIR,
    PROJECT_ROOT,
    REFLECTION_MODEL,
    RESULTS_DIR,
    SPLITS_DIR,
    TASK_MODEL,
)
from gepa_toy.data_processing.tickets import load_split

# dspy comes through dspy_task, which imports it in the order LiteLLM needs.
from gepa_toy.dspy_task import (
    ProgressLog,
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
from gepa_toy.evaluate import map_until_error, save_run, summarise
from gepa_toy.model import ModelConfig
from gepa_toy.utils import git_version


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget", type=int, default=1000, help="task-model metric calls"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-split", default="test", choices=["test", "label_key"])
    # DSPy's reply format is longer than our JSON: 300 truncated some replies.
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--resume",
        type=Path,
        help="run dir of a stopped run; its saved arguments are used",
    )
    args = parser.parse_args(argv)
    models = {"task_model": TASK_MODEL, "reflection_model": REFLECTION_MODEL}
    if args.resume:
        # A resumed run must use exactly the settings it started with.
        others = {k: v for k, v in vars(args).items() if k != "resume"}
        if others != {k: parser.get_default(k) for k in others}:
            parser.error(
                "--resume takes no other options: the run's saved ones are used"
            )
        saved = json.loads((args.resume / "args.json").read_text())
        if {k: saved.pop(k) for k in models} != models:
            parser.error("config.py models differ from the ones this run started with")
        return argparse.Namespace(**saved), args.resume
    started_at = datetime.now(UTC)
    run_dir = GEPA_RESULTS_DIR / f"{started_at:%Y%m%d-%H%M%S-%f}_stage4_seed{args.seed}"
    run_dir.mkdir(parents=True)
    saved = {k: v for k, v in vars(args).items() if k != "resume"}
    (run_dir / "args.json").write_text(json.dumps(saved | models, indent=2) + "\n")
    return argparse.Namespace(**saved), run_dir


def lm_usage(lm: dspy.LM) -> dict:
    usage = [call["usage"] for call in lm.history]
    return {
        "calls": len(usage),
        # The `~...-latest` aliases move: this records which model actually answered.
        "resolved_models": sorted({call["response_model"] for call in lm.history}),
        **{
            key: sum(u.get(key) or 0 for u in usage)
            for key in ("prompt_tokens", "completion_tokens", "cost")
        },
    }


def main(argv: list[str] | None = None) -> None:
    args, run_dir = parse_args(argv)
    task_config = ModelConfig(model=TASK_MODEL, max_tokens=args.max_tokens)
    # Reflection writes a whole instruction, so it gets room to think and some randomness:
    # each seed's run should propose different prompts.
    reflection_config = ModelConfig(
        model=REFLECTION_MODEL,
        temperature=1.0,
        # Earlier reflections averaged ~6,100 tokens with thinking; running out of room
        # gives an empty reply, which stops the run.
        max_tokens=32000,
        reasoning=True,
        timeout=300.0,
    )
    # Gathered before any model calls, so a failure here doesn't waste a paid run.
    metadata = {
        "started_at": datetime.now(UTC).isoformat(),
        "git": git_version(),
        "stage": 4,
        "dspy_version": dspy.__version__,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "args": vars(args),
        "task_model": task_config.model_dump(),
        "reflection_model": reflection_config.model_dump(),
        "minibatch_size": 5,
    }

    stop = [StopOnLMError(exit_run(run_dir / "stops.log"))]
    task_lm = make_lm(task_config, stop)
    reflection_lm = make_lm(reflection_config, stop)
    train, pareto = load_split("train"), load_split("pareto")
    eval_rows = load_split(args.eval_split)
    progress = ProgressLog(args.budget, len(eval_rows), run_dir / "progress.log")
    optimiser = dspy.GEPA(
        metric=metric,
        max_metric_calls=args.budget,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=5,
        candidate_selection_strategy="pareto",
        skip_perfect_score=True,
        use_merge=False,
        num_threads=args.workers,
        track_stats=True,
        seed=args.seed,
        # GEPA resumes from the checkpoint it finds here.
        log_dir=str(run_dir / "dspy_state"),
        gepa_kwargs={"callbacks": [progress]},
    )
    start = time.monotonic()
    optimised = optimiser.compile(
        build_program(task_lm),
        trainset=[to_example(r) for r in train],
        valset=[to_example(r) for r in pareto],
    )
    metadata["optimise_seconds"] = round(time.monotonic() - start, 2)
    metadata["reflection_usage"] = lm_usage(reflection_lm)

    log_path = run_dir / "candidates.json"
    log = {
        "metadata": metadata,
        **candidate_log(optimised.detailed_results, [r["id"] for r in pareto]),
    }
    log_path.write_text(json.dumps(log, indent=2) + "\n")

    eval_file = SPLITS_DIR / f"{args.eval_split}.tsv"
    print(f"Scoring the best candidate on {len(eval_rows)} tickets", flush=True)
    start = time.monotonic()
    predictions = map_until_error(
        lambda r: predict_row(optimised, r), eval_rows, args.workers
    )
    run = {
        "metadata": {
            **metadata,
            "data_file": str(eval_file.relative_to(PROJECT_ROOT)),
            "model": task_config.model_dump(),
            "task_usage": lm_usage(task_lm),
            "prompt": optimised.predictors()[0].signature.instructions,
            "gepa_log": str(log_path.relative_to(PROJECT_ROOT)),
            "runtime_seconds": round(time.monotonic() - start, 2),
        },
        "predictions": predictions,
        "results": summarise(predictions),
    }
    path = save_run(RESULTS_DIR, run)
    print(json.dumps(run["results"], indent=2))
    print(
        f"Best candidate {log['best']} of {len(log['candidates'])}, "
        f"{log['total_metric_calls']} metric calls"
    )
    print(f"Saved log to {log_path}\nSaved run to {path}")


if __name__ == "__main__":
    main()
