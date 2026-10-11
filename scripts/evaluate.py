"""Run the zero-shot prompt over a split and save metadata, per-ticket predictions and scores."""

import argparse
import json

from gepa_toy.config import RESULTS_DIR, SPLITS_DIR, TASK_MODEL
from gepa_toy.evaluate import evaluate, save_run
from gepa_toy.model import Model
from gepa_toy.zero_shot import build_expanded_prompt, build_prompt

PROMPTS = {"zero_shot": build_prompt, "expanded": build_expanded_prompt}


def main() -> None:
    defaults = Model(model=TASK_MODEL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", default="train", choices=["train", "pareto", "test", "label_key"]
    )
    parser.add_argument("--prompt", default="zero_shot", choices=PROMPTS)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--timeout", type=float, default=defaults.timeout)
    parser.add_argument("--reasoning", action="store_true", default=defaults.reasoning)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    model = Model(
        model=TASK_MODEL,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        reasoning=args.reasoning,
    )
    run = evaluate(
        SPLITS_DIR / f"{args.split}.tsv",
        PROMPTS[args.prompt],
        model,
        args.workers,
    )
    path = save_run(RESULTS_DIR, run)
    print(json.dumps(run["results"], indent=2))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
