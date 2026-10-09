import json
import tempfile
import unittest
from itertools import cycle
from pathlib import Path
from unittest.mock import patch

import httpx2
import openai

from gepa_toy.config import LABELS as ALL_LABELS
from gepa_toy.config import SPLITS_DIR
from gepa_toy.data_processing.tsv import read_tsv
from gepa_toy.evaluate import (
    evaluate,
    evaluate_rows,
    gepa_score,
    labels_of,
    queue_confusion,
    save_run,
    summarise,
    tag_scores,
)
from gepa_toy.model import ModelConfig, ResponseShapeError
from gepa_toy.zero_shot import build_prompt

TRAIN = SPLITS_DIR / "train.tsv"
IT_SUPPORT = (
    '{"queue": "IT Support", "type": "Incident", "priority": "high", "tags": ["HR"]}'
)
BILLING = (
    '{"queue": "Billing and Payments", "type": "Request", "priority": "low",'
    ' "tags": ["Operations", "HR"]}'
)
LABELS = {
    "queue": "IT Support",
    "type": "Incident",
    "priority": "high",
    "tags": ["HR", "Integration"],
}


def status_error(cls, code):
    request = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx2.Response(code, request=request)
    return cls(f"Error code: {code}", response=response, body=None)


def record(predicted, labels, error=None):
    return {
        "predicted": predicted,
        "labels": labels,
        "gepa_score": gepa_score(predicted, labels),
        "error": error,
    }


class TagScoresTest(unittest.TestCase):
    def test_partial_overlap(self):
        scores = tag_scores(["HR", "Operations"], ["HR", "Integration", "Security"])
        self.assertAlmostEqual(scores["precision"], 1 / 2)
        self.assertAlmostEqual(scores["recall"], 1 / 3)
        self.assertAlmostEqual(scores["f1"], 0.4)

    def test_no_prediction_scores_zero(self):
        self.assertEqual(
            tag_scores([], ["HR"]), {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        )


class GepaScoreTest(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(gepa_score(LABELS, LABELS), 1.0)

    def test_failed_prediction_scores_zero(self):
        self.assertEqual(gepa_score(None, LABELS), 0.0)

    def test_averages_three_accuracies_and_tag_f1(self):
        predicted = {**LABELS, "priority": "low", "tags": ["HR"]}
        # queue and type right, priority wrong, tag F1 = 2/3.
        self.assertAlmostEqual(gepa_score(predicted, LABELS), (1 + 1 + 0 + 2 / 3) / 4)


class SummariseTest(unittest.TestCase):
    def test_scores_each_field(self):
        other = {
            "queue": "Billing and Payments",
            "type": "Request",
            "priority": "low",
            "tags": ["Operations"],
        }
        results = summarise(
            [record(LABELS, LABELS), record(LABELS, other), record(None, other, "bad")]
        )
        self.assertEqual(results["n"], 3)
        self.assertEqual(results["errors"], 1)
        self.assertAlmostEqual(results["queue"]["accuracy"], 1 / 3)
        # IT Support: P=1/2 R=1; Billing: P=0 R=0. Macro over the two labelled classes.
        self.assertAlmostEqual(results["queue"]["precision"], 1 / 4)
        self.assertAlmostEqual(results["queue"]["recall"], 1 / 2)
        self.assertAlmostEqual(results["tags"]["recall"], 1 / 3)
        self.assertAlmostEqual(results["gepa_score"], 1 / 3)
        self.assertEqual(set(results["queue_confusion"]), {"rows", "columns", "counts"})

    def test_queue_confusion_counts(self):
        billing = {**LABELS, "queue": "Billing and Payments"}
        predictions = [
            record(LABELS, LABELS),  # IT Support -> IT Support
            record(billing, LABELS),  # IT Support -> Billing
            record(None, LABELS, "bad"),  # IT Support -> (none)
            record(billing, billing),  # Billing -> Billing
        ]
        matrix = queue_confusion(predictions)
        queues = ALL_LABELS["queue"]
        self.assertEqual(matrix["rows"], queues)
        self.assertEqual(matrix["columns"], queues + ["(none)"])
        it, billing_i = queues.index("IT Support"), queues.index("Billing and Payments")
        expected = [[0] * (len(queues) + 1) for _ in queues]
        expected[it][it] = 1
        expected[it][billing_i] = 1
        expected[it][-1] = 1
        expected[billing_i][billing_i] = 1
        self.assertEqual(matrix["counts"], expected)
        for queue, row in zip(queues, matrix["counts"], strict=True):
            self.assertEqual(
                sum(row), sum(p["labels"]["queue"] == queue for p in predictions)
            )


class EvaluateTest(unittest.TestCase):
    @patch("gepa_toy.model.complete", return_value=IT_SUPPORT)
    def test_run_has_metadata_predictions_and_results(self, _):
        run = evaluate(TRAIN, build_prompt, ModelConfig())
        self.assertEqual(set(run), {"metadata", "predictions", "results"})
        self.assertEqual(
            run["metadata"]["data_file"],
            "data/customer_support_tickets/splits/train.tsv",
        )
        self.assertEqual(len(run["metadata"]["git"]["commit"]), 40)
        self.assertIn("Ticket subject: {subject}", run["metadata"]["prompt"])
        first = run["predictions"][0]
        self.assertEqual(
            set(first), {"id", "predicted", "labels", "gepa_score", "error", "raw"}
        )
        self.assertEqual(set(first["labels"]), {"queue", "type", "priority", "tags"})
        self.assertEqual(run["results"]["n"], 100)
        # Train has 10 tickets per queue and every prediction is IT Support.
        self.assertAlmostEqual(run["results"]["queue"]["accuracy"], 0.1)

    @patch("gepa_toy.model.complete")
    def test_transient_errors_are_recorded(self, complete):
        for error in (
            openai.APIConnectionError(request=None),
            openai.APITimeoutError(request=None),
            status_error(openai.RateLimitError, 429),
            status_error(openai.InternalServerError, 502),
        ):
            complete.side_effect = error
            first = evaluate(TRAIN, build_prompt, ModelConfig())["predictions"][0]
            self.assertIsNone(first["predicted"])
            self.assertEqual(first["gepa_score"], 0.0)
            self.assertTrue(first["error"].startswith("API error"))

    @patch("gepa_toy.model.complete")
    def test_bad_response_shape_is_a_failed_prediction(self, complete):
        complete.side_effect = ResponseShapeError(
            "unexpected response shape: <html>Bad gateway</html>"
        )
        first = evaluate(TRAIN, build_prompt, ModelConfig())["predictions"][0]
        self.assertIsNone(first["predicted"])
        self.assertEqual(first["gepa_score"], 0.0)
        self.assertEqual(
            first["error"],
            "API error: unexpected response shape: <html>Bad gateway</html>",
        )
        self.assertEqual(first["raw"], "")

    @patch("gepa_toy.model.complete")
    def test_results_match_the_pre_refactor_implementation(self, complete):
        # Values captured from evaluate() before evaluate_rows and model.complete
        # were split out, with the same replies cycled over train in row order.
        complete.side_effect = cycle([IT_SUPPORT, BILLING, "nope"])
        run = evaluate(TRAIN, build_prompt, ModelConfig(), workers=1)
        results = {k: v for k, v in run["results"].items() if k != "queue_confusion"}
        self.assertEqual(
            results,
            {
                "n": 100,
                "errors": 33,
                "gepa_score": 0.13529761904761906,
                "queue": {
                    "accuracy": 0.06,
                    "f1": 0.02758985200845666,
                    "precision": 0.017914438502673796,
                    "recall": 0.06,
                },
                "type": {
                    "accuracy": 0.25,
                    "f1": 0.1886384335154827,
                    "precision": 0.18649732620320855,
                    "recall": 0.19266917293233082,
                },
                "priority": {
                    "accuracy": 0.22,
                    "f1": 0.2242452536570184,
                    "precision": 0.21895424836601307,
                    "recall": 0.23119868637110016,
                },
                "tags": {
                    "f1": 0.01119047619047619,
                    "precision": 0.02,
                    "recall": 0.007833333333333333,
                },
            },
        )
        third = run["predictions"][2]
        self.assertEqual(third["id"], "5737")
        self.assertIsNone(third["predicted"])
        self.assertEqual(third["error"], "no JSON object in response")
        self.assertEqual(third["raw"], "nope")

    @patch("gepa_toy.model.complete", return_value=IT_SUPPORT)
    def test_evaluate_rows_returns_one_record_per_row_in_order(self, _):
        rows = read_tsv(TRAIN)[:7]
        records = evaluate_rows(rows, build_prompt, ModelConfig(), workers=4)
        self.assertEqual(len(records), 7)
        for record, row in zip(records, rows, strict=True):
            self.assertEqual(record["id"], row["id"])
            self.assertEqual(record["labels"], labels_of(row))
        self.assertEqual(
            set(records[0]), {"id", "predicted", "labels", "gepa_score", "error", "raw"}
        )

    @patch("gepa_toy.model.complete")
    def test_setup_errors_stop_the_run(self, complete):
        for error in (
            KeyError("OPENROUTER_API_KEY"),
            status_error(openai.BadRequestError, 400),
            status_error(openai.AuthenticationError, 401),
            status_error(openai.PermissionDeniedError, 403),
            status_error(openai.NotFoundError, 404),
        ):
            complete.side_effect = error
            with self.assertRaises(type(error)):
                evaluate(TRAIN, build_prompt, ModelConfig())

    @patch("gepa_toy.model.complete")
    @patch("gepa_toy.evaluate.git_version", side_effect=RuntimeError("no git"))
    def test_metadata_failure_happens_before_any_model_call(self, _, complete):
        with self.assertRaises(RuntimeError):
            evaluate(TRAIN, build_prompt, ModelConfig())
        complete.assert_not_called()

    def test_labels_split_tag_groups(self):
        row = {
            "queue": "IT Support",
            "type": "Incident",
            "priority": "high",
            "tag_groups": "Security,General Support",
        }
        self.assertEqual(labels_of(row)["tags"], ["Security", "General Support"])

    @patch("gepa_toy.model.complete", return_value=IT_SUPPORT)
    def test_runs_in_the_same_second_get_separate_files(self, _):
        run = evaluate(TRAIN, build_prompt, ModelConfig())
        other = {**run, "metadata": {**run["metadata"]}}
        other["metadata"]["started_at"] = run["metadata"]["started_at"].replace(
            run["metadata"]["started_at"][20:26], "999999"
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.assertNotEqual(save_run(Path(tmp), run), save_run(Path(tmp), other))
            self.assertEqual(len(list(Path(tmp).iterdir())), 2)

    @patch("gepa_toy.model.complete", return_value=IT_SUPPORT)
    def test_save_run_round_trips(self, _):
        run = evaluate(TRAIN, build_prompt, ModelConfig())
        with tempfile.TemporaryDirectory() as tmp:
            path = save_run(Path(tmp), run)
            self.assertTrue(path.name.endswith("_train.json"))
            self.assertEqual(json.loads(path.read_text()), run)


if __name__ == "__main__":
    unittest.main()
