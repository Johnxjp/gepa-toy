"""Per stage: n, mean and sample standard deviation of queue accuracy and gepa_score.

Only test-split runs count. Errors is the mean number of failed predictions (API errors
and unparseable replies) per run; a run with many is a casualty, not a result. Smoke
runs and discarded runs are deleted from results/ by hand (design section 4.5).
"""

import json
from pathlib import Path
from statistics import mean, stdev

from gepa_toy.config import RESULTS_DIR

STAGE_ORDER = ["0", "1", "2", "3", "3c", "4"]


def spread(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def summarise_runs(runs: list[dict]) -> list[dict]:
    by_stage: dict[str, list[dict]] = {}
    for run in runs:
        by_stage.setdefault(str(run["metadata"]["stage"]), []).append(run)
    rows = []
    for stage in sorted(by_stage, key=STAGE_ORDER.index):
        accuracy = [r["results"]["queue"]["accuracy"] for r in by_stage[stage]]
        score = [r["results"]["gepa_score"] for r in by_stage[stage]]
        errors = [r["results"]["errors"] for r in by_stage[stage]]
        rows.append(
            {
                "stage": stage,
                "n": len(accuracy),
                "queue_accuracy": mean(accuracy),
                "queue_accuracy_sd": spread(accuracy),
                "gepa_score": mean(score),
                "gepa_score_sd": spread(score),
                "errors": mean(errors),
            }
        )
    return rows


def main() -> None:
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    rows = summarise_runs(
        [
            r
            for r in runs
            if "stage" in r["metadata"]
            and Path(r["metadata"]["data_file"]).stem == "test"
        ]
    )
    print(
        f"{'stage':<6}{'n':>3}{'queue acc':>12}{'sd':>8}{'gepa_score':>12}{'sd':>8}"
        f"{'errors':>8}"
    )
    for r in rows:
        print(
            f"{r['stage']:<6}{r['n']:>3}{r['queue_accuracy']:>12.3f}"
            f"{r['queue_accuracy_sd']:>8.3f}{r['gepa_score']:>12.3f}"
            f"{r['gepa_score_sd']:>8.3f}{r['errors']:>8.1f}"
        )


if __name__ == "__main__":
    main()
