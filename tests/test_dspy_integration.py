"""DSPy's real call path against a fake OpenRouter: which failures stop a run, and resuming."""

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from fake_openrouter import answer, chat, error, fake_openrouter

from gepa_toy.dspy_task import StopOnLMError, build_program, make_lm, predict_row
from gepa_toy.model import Model

ROW = {
    "id": "7",
    "subject": "VPN down",
    "body": "Nobody can connect.",
    "queue": "IT Support",
    "type": "Incident",
    "priority": "high",
    "tag_groups": "Security",
}
GOOD = answer("IT Support", "Incident", "high", ["Security"])


def classify_once(handler, timeout: float = 60.0):
    """One ticket through a real dspy.LM; returns (record or exception, stop reasons, requests)."""
    stops = []
    lm = make_lm(
        Model(model="fake/task", max_tokens=1000, timeout=timeout),
        [StopOnLMError(stops.append)],
    )
    with fake_openrouter(handler) as requests:
        try:
            outcome = predict_row(build_program(lm), ROW)
        # Any type: which exception DSPy raises is what these tests are finding out.
        except Exception as e:  # noqa: BLE001
            outcome = e
    return outcome, stops, requests


class StopsTheRunTest(unittest.TestCase):
    def test_api_errors_stop_after_dspy_retries(self):
        for status in (400, 401, 402, 403, 404, 429, 500, 502, 503):
            with self.subTest(status=status):
                outcome, stops, _ = classify_once(lambda _, s=status: (s, error(s)))
                self.assertIsInstance(outcome, Exception)
                self.assertEqual(len(stops), 1)
                self.assertTrue(stops[0].startswith("LM"), stops[0])

    def test_rate_limit_that_clears_on_retry_does_not_stop(self):
        replies = iter([(429, error(429)), (429, error(429)), (200, chat(GOOD))])
        outcome, stops, requests = classify_once(lambda _: next(replies))
        self.assertEqual(stops, [])
        self.assertEqual(outcome["predicted"]["queue"], "IT Support")
        self.assertEqual(len(requests), 3)

    def test_unusable_200_replies_stop(self):
        for name, payload in (
            ("no choices", {**chat(GOOD), "choices": []}),
            ("html page", b"<html>Bad gateway</html>"),
            ("error body", error(502)),
            (
                "upstream error",
                {
                    **chat(""),
                    "choices": [{**chat("")["choices"][0], "finish_reason": "error"}],
                },
            ),
            ("empty text", chat("")),
            ("null text", chat(None, finish="length")),
        ):
            with self.subTest(name):
                _, stops, _ = classify_once(lambda _, p=payload: (200, p))
                self.assertEqual(len(stops), 1, stops)

    def test_timeout_stops_after_retries(self):
        def slow(_):
            time.sleep(1)
            return 200, chat(GOOD)

        _, stops, requests = classify_once(slow, timeout=0.2)
        self.assertEqual(len(stops), 1)
        self.assertTrue(stops[0].startswith("LMTimeoutError"), stops[0])
        self.assertEqual(len(requests), 4)  # one call plus DSPy's three retries


class ModelMistakesAreRecordedTest(unittest.TestCase):
    def test_bad_format_and_truncation_are_scored_not_stopped(self):
        for name, payload in (
            ("unparseable", chat("I think this is IT Support.")),
            ("invalid label", chat(GOOD.replace("IT Support", "it support"))),
            ("truncated", chat("[[ ## queue ## ]]\nIT Sup", finish="length")),
        ):
            with self.subTest(name):
                record, stops, requests = classify_once(lambda _, p=payload: (200, p))
                self.assertEqual(stops, [])
                self.assertIsNone(record["predicted"])
                self.assertEqual(record["gepa_score"], 0.0)
                self.assertIsNotNone(record["error"])
                # One call each: DSPy's JSON-mode retry is off.
                self.assertEqual(len(requests), 1)


class StopAndResumeTest(unittest.TestCase):
    """Runs the real script in a subprocess: stopping exits the process."""

    def run_script(self, out: Path, fail_at: int, *args: str):
        return subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "fake_dspy_run.py"),
                str(out),
                str(fail_at),
                *args,
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=300,
        )

    def test_stopped_run_resumes_without_zero_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            # 50 pareto calls to score the seed, then cycles of 5 + reflection + 5:
            # request 63 is mid-way through the second cycle's parent minibatch.
            args = ["--budget", "70", "--eval-split", "label_key", "--workers", "4"]
            stopped = self.run_script(out, 63, *args)
            self.assertEqual(stopped.returncode, 1, stopped.stderr[-2000:])
            [run_dir] = (out / "gepa").iterdir()
            self.assertIn("LMAuthError", (run_dir / "stops.log").read_text())
            self.assertTrue((run_dir / "dspy_state" / "gepa_state.bin").exists())
            self.assertFalse((run_dir / "candidates.json").exists())

            resumed = self.run_script(out, 0, "--resume", str(run_dir))
            self.assertEqual(resumed.returncode, 0, resumed.stderr[-2000:])

            # Every honest score is 0.75; a 0 anywhere means a failed call leaked in.
            log = json.loads((run_dir / "candidates.json").read_text())
            for candidate in log["candidates"]:
                self.assertNotIn(0.0, candidate["pareto_scores"].values())
            for cycle in json.loads(
                (run_dir / "dspy_state" / "run_log.json").read_text()
            ):
                self.assertNotIn(0.0, cycle["subsample_scores"])
                self.assertNotIn(0.0, cycle.get("new_subsample_scores") or [])
            [run_file] = out.glob("*_label_key.json")
            run = json.loads(run_file.read_text())
            self.assertEqual(run["results"]["errors"], 0)
            self.assertEqual(run["metadata"]["args"]["budget"], 70)
            for usage in ("reflection_usage", "task_usage"):
                self.assertEqual(
                    run["metadata"][usage]["resolved_models"], ["fake/resolved-model"]
                )
            self.assertGreaterEqual(run["metadata"]["task_usage"]["calls"], 50)


if __name__ == "__main__":
    unittest.main()
