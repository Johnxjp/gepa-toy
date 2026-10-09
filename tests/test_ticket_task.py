import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import openai

from gepa_toy.cache import Cache
from gepa_toy.config import LABELS as ALL_LABELS
from gepa_toy.evaluate import gepa_score, labels_of
from gepa_toy.gepa import Rollout
from gepa_toy.model import ModelConfig, parse_prediction
from gepa_toy.ticket_task import TicketTask, feedback
from gepa_toy.zero_shot import build

LABELS = {
    "queue": "Technical Support",
    "type": "Incident",
    "priority": "high",
    "tags": ["Security", "Outages & Recovery"],
}
CORRECT = (
    "Correct:   queue Technical Support, type Incident, priority high,"
    " tags [Security, Outages & Recovery]"
)
# (name, predicted, error, expected written feedback)
CASES = [
    (
        "all right",
        LABELS,
        None,
        "Score: 1.00 (queue 1, type 1, priority 1, tags F1 1.00)\n"
        "Predicted: queue Technical Support, type Incident, priority high,"
        " tags [Security, Outages & Recovery]\n" + CORRECT,
    ),
    (
        "queue wrong",
        {**LABELS, "queue": "IT Support"},
        None,
        "Score: 0.75 (queue 0, type 1, priority 1, tags F1 1.00)\n"
        "Predicted: queue IT Support, type Incident, priority high,"
        " tags [Security, Outages & Recovery]\n" + CORRECT,
    ),
    (
        "type wrong",
        {**LABELS, "type": "Problem"},
        None,
        "Score: 0.75 (queue 1, type 0, priority 1, tags F1 1.00)\n"
        "Predicted: queue Technical Support, type Problem, priority high,"
        " tags [Security, Outages & Recovery]\n" + CORRECT,
    ),
    (
        "priority wrong",
        {**LABELS, "priority": "low"},
        None,
        "Score: 0.75 (queue 1, type 1, priority 0, tags F1 1.00)\n"
        "Predicted: queue Technical Support, type Incident, priority low,"
        " tags [Security, Outages & Recovery]\n" + CORRECT,
    ),
    (
        "tags missed",
        {**LABELS, "tags": ["Security"]},
        None,
        "Score: 0.92 (queue 1, type 1, priority 1, tags F1 0.67)\n"
        "Predicted: queue Technical Support, type Incident, priority high,"
        " tags [Security]\n" + CORRECT,
    ),
    (
        "tags extra",
        {**LABELS, "tags": ["Security", "Outages & Recovery", "HR"]},
        None,
        "Score: 0.95 (queue 1, type 1, priority 1, tags F1 0.80)\n"
        "Predicted: queue Technical Support, type Incident, priority high,"
        " tags [Security, Outages & Recovery, HR]\n" + CORRECT,
    ),
    (
        "parse failure",
        None,
        "no JSON object in response",
        "Score: 0.00 (queue 0, type 0, priority 0, tags F1 0.00)\n"
        "Reply was not valid: no JSON object in response\n" + CORRECT,
    ),
]

ROWS = [
    {
        "id": "1",
        "subject": "Server down",
        "body": "The main server stopped responding.",
        "queue": "Technical Support",
        "type": "Incident",
        "priority": "high",
        "tag_groups": "Security,Outages & Recovery",
    },
    {
        "id": "2",
        "subject": "",
        "body": "Please refund my last invoice.",
        "queue": "Billing and Payments",
        "type": "Request",
        "priority": "low",
        "tag_groups": "Billing & Returns",
    },
    {
        "id": "3",
        "subject": "New laptop",
        "body": "I need a laptop for the new starter.",
        "queue": "IT Support",
        "type": "Request",
        "priority": "medium",
        "tag_groups": "Hardware & Infrastructure",
    },
]
REPLY = (
    '{"queue": "Technical Support", "type": "Incident", "priority": "high",'
    ' "tags": ["Security", "Outages & Recovery"]}'
)
CANDIDATE = "Classify carefully. Outages are always high priority."


class FeedbackTest(unittest.TestCase):
    def test_written_cases(self):
        for name, predicted, error, expected in CASES:
            with self.subTest(name):
                self.assertEqual(
                    feedback(predicted, LABELS, error, "written"), expected
                )

    def test_written_score_line_matches_gepa_score(self):
        for name, predicted, error, _ in CASES:
            with self.subTest(name):
                first = feedback(predicted, LABELS, error, "written").splitlines()[0]
                self.assertTrue(
                    first.startswith(f"Score: {gepa_score(predicted, LABELS):.2f} (")
                )

    def test_score_mode_is_the_score_alone(self):
        words = {"Predicted", "Correct", *(v for vs in ALL_LABELS.values() for v in vs)}
        for name, predicted, error, _ in CASES:
            with self.subTest(name):
                text = feedback(predicted, LABELS, error, "score")
                self.assertEqual(text, f"Score: {gepa_score(predicted, LABELS):.2f}")
                for word in words:
                    self.assertNotIn(word, text)

    def test_multiline_validation_error_becomes_one_line(self):
        error = parse_prediction(REPLY.replace("Technical Support", "Tech")).error
        self.assertGreater(len(error.splitlines()), 1)
        text = feedback(None, LABELS, error, "written")
        self.assertEqual(len(text.splitlines()), 3)
        self.assertIn("Reply was not valid: 1 validation error", text)
        self.assertIn("input_value='Tech'", text)


class TicketTaskTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Cache(Path(self.tmp.name) / "cache.sqlite")
        self.config = ModelConfig(model="test/model", temperature=0.5)
        self.task = TicketTask(self.config, "written", self.cache, workers=2)

    def tearDown(self):
        self.tmp.cleanup()

    @patch("gepa_toy.model.complete", return_value=REPLY)
    def test_one_rollout_per_row_in_order_from_the_candidate(self, complete):
        rollouts = self.task.evaluate(CANDIDATE, ROWS)
        self.assertEqual(len(rollouts), 3)
        # call_args_list is appended atomically; call_count is not, across worker threads.
        self.assertEqual(
            sorted(call.args for call in complete.call_args_list),
            sorted(
                (build(CANDIDATE, r["subject"], r["body"]), self.config) for r in ROWS
            ),
        )
        for rollout, row in zip(rollouts, ROWS, strict=True):
            self.assertIsInstance(rollout, Rollout)
            self.assertEqual(rollout.record["id"], row["id"])
            self.assertEqual(rollout.record["labels"], labels_of(row))
            self.assertEqual(rollout.score, rollout.record["gepa_score"])
        self.assertEqual(
            set(rollouts[0].record),
            {"id", "predicted", "labels", "gepa_score", "error", "raw"},
        )
        self.assertEqual(rollouts[0].score, 1.0)
        self.assertEqual(rollouts[1].score, 0.0)
        self.assertEqual(rollouts[0].feedback, CASES[0][3])
        self.assertEqual(
            rollouts[0].trace,
            "Subject: Server down\n"
            "Body:\n"
            "The main server stopped responding.\n"
            "Predicted: queue Technical Support, type Incident, priority high,"
            " tags [Security, Outages & Recovery]",
        )

    @patch("gepa_toy.model.complete", return_value=REPLY)
    def test_same_candidate_again_makes_no_model_calls(self, complete):
        first = self.task.evaluate(CANDIDATE, ROWS)
        second = self.task.evaluate(CANDIDATE, ROWS)
        self.assertEqual(len(complete.call_args_list), 3)
        self.assertEqual(first, second)
        self.task.evaluate("A different instruction.", ROWS)
        self.assertEqual(len(complete.call_args_list), 6)

    @patch("gepa_toy.model.complete")
    def test_transient_error_scores_zero_and_is_retried(self, complete):
        complete.side_effect = openai.APIConnectionError(request=None)
        [rollout] = self.task.evaluate(CANDIDATE, ROWS[:1])
        self.assertEqual(rollout.score, 0.0)
        self.assertIsNone(rollout.record["predicted"])
        self.assertIn("Reply was not valid: API error", rollout.feedback)
        self.assertEqual(len(rollout.feedback.splitlines()), 3)
        self.assertIn("\nReply was not valid: API error", rollout.trace)
        complete.side_effect = None
        complete.return_value = REPLY
        [retried] = self.task.evaluate(CANDIDATE, ROWS[:1])
        self.assertEqual(len(complete.call_args_list), 2)
        self.assertEqual(retried.score, 1.0)

    @patch("gepa_toy.model.complete", return_value="not json")
    def test_unparseable_reply_is_cached_and_scored_zero(self, complete):
        [rollout] = self.task.evaluate(CANDIDATE, ROWS[:1])
        [again] = self.task.evaluate(CANDIDATE, ROWS[:1])
        self.assertEqual(len(complete.call_args_list), 1)
        self.assertEqual(rollout.score, 0.0)
        self.assertEqual(rollout.record["raw"], "not json")
        self.assertEqual(rollout, again)

    @patch("gepa_toy.model.complete", return_value=REPLY)
    def test_score_mode_feedback(self, _):
        task = TicketTask(ModelConfig(), "score", self.cache)
        rollouts = task.evaluate(CANDIDATE, ROWS[:2])
        self.assertEqual([r.feedback for r in rollouts], ["Score: 1.00", "Score: 0.00"])
        # The trace still carries the prediction, so the reflector has something to read.
        self.assertIn("Predicted: queue Technical Support", rollouts[0].trace)
        self.assertTrue(rollouts[1].trace.startswith("Subject: (no subject)\nBody:\n"))

    def test_describe(self):
        self.assertEqual(
            self.task.describe(),
            {"model": self.config.model_dump(), "feedback": "written"},
        )
        self.assertEqual(self.task.describe()["model"]["model"], "test/model")
        self.assertEqual(
            TicketTask(ModelConfig(), "score", self.cache).describe()["feedback"],
            "score",
        )


if __name__ == "__main__":
    unittest.main()
