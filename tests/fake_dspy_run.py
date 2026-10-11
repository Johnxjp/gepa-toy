"""Runs scripts/run_dspy.py against the fake OpenRouter, for the stop-and-resume test.

Usage: fake_dspy_run.py OUT_DIR FAIL_AT [run_dspy args...]

The task model answers every ticket with its true labels except priority, so every
honest score is 0.75 and a 0 can only come from a failed call. Request number FAIL_AT
(counting from 1; 0 means never) gets a 401.
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_openrouter import answer, chat, error, fake_openrouter

from gepa_toy.data_processing.tickets import load_split
from gepa_toy.evaluate import labels_of
from scripts import run_dspy

out_dir, fail_at = Path(sys.argv[1]), int(sys.argv[2])
labels_by_body = {
    row["body"]: labels_of(row)
    for split in ("train", "pareto", "label_key")
    for row in load_split(split)
}
count = {"n": 0}
lock = threading.Lock()


def handler(body):
    with lock:
        count["n"] += 1
        n = count["n"]
    if n == fail_at:
        return 401, error(401, "User not found")
    if body["model"] == run_dspy.REFLECTION_MODEL:
        return 200, chat(f"```\nNEW INSTRUCTION {n}\n```")
    user = body["messages"][-1]["content"]
    ticket_body = user.split("[[ ## body ## ]]\n", 1)[1].split("\n\nRespond with", 1)[0]
    truth = labels_by_body[ticket_body]
    wrong_priority = "medium" if truth["priority"] == "low" else "low"
    return 200, chat(
        answer(truth["queue"], truth["type"], wrong_priority, truth["tags"])
    )


# Keep every output out of the real results folder.
run_dspy.PROJECT_ROOT = Path("/")
run_dspy.GEPA_RESULTS_DIR = out_dir / "gepa"
run_dspy.RESULTS_DIR = out_dir
with fake_openrouter(handler):
    run_dspy.main(sys.argv[3:])
