import unittest

from gepa_toy.config import LABELS
from gepa_toy.feedback import score_feedback, written_feedback

LABELS_ = {
    "queue": "Technical Support",
    "type": "Incident",
    "priority": "high",
    "tags": ["Network", "Outage"],
}
CORRECT_LINE = (
    "Correct:   queue Technical Support, type Incident, priority high, "
    "tags [Network, Outage]"
)


class WrittenFeedbackTest(unittest.TestCase):
    def test_all_right(self):
        self.assertEqual(
            written_feedback(dict(LABELS_), LABELS_, None),
            "Score: 1.00 (queue 1, type 1, priority 1, tags F1 1.00)\n"
            "Predicted: queue Technical Support, type Incident, priority high, "
            "tags [Network, Outage]\n" + CORRECT_LINE,
        )

    def test_wrong_priority_and_partial_tags(self):
        predicted = {**LABELS_, "priority": "low", "tags": ["Network", "Billing"]}
        self.assertEqual(
            written_feedback(predicted, LABELS_, None),
            "Score: 0.62 (queue 1, type 1, priority 0, tags F1 0.50)\n"
            "Predicted: queue Technical Support, type Incident, priority low, "
            "tags [Network, Billing]\n" + CORRECT_LINE,
        )

    def test_each_wrong_field_scores_zero(self):
        for field, wrong in (
            ("queue", "Billing and Payments"),
            ("type", "Request"),
            ("priority", "low"),
        ):
            feedback = written_feedback({**LABELS_, field: wrong}, LABELS_, None)
            self.assertIn(f"{field} 0", feedback.splitlines()[0])
            self.assertIn(f"{field} {wrong}", feedback.splitlines()[1])

    def test_parse_failure_is_one_line_without_docs_links(self):
        error = (
            "1 validation error for Prediction\n"
            "queue\n"
            "  Input should be 'Technical Support' or 'IT Support' "
            "[type=literal_error, input_value='<queue>', input_type=str]\n"
            "    For further information visit https://errors.pydantic.dev/2.14/v/literal_error"
        )
        lines = written_feedback(None, LABELS_, error).splitlines()
        self.assertEqual(lines[0], "Score: 0.00 (no valid reply)")
        self.assertTrue(lines[1].startswith("Reply was not valid: 1 validation error"))
        self.assertIn("input_value='<queue>'", lines[1])
        self.assertNotIn("pydantic.dev", lines[1])
        self.assertEqual(lines[2], CORRECT_LINE)


class ScoreFeedbackTest(unittest.TestCase):
    def test_score_only_never_names_a_label(self):
        predicted = {**LABELS_, "priority": "low", "tags": ["Network", "Billing"]}
        feedback = score_feedback(predicted, LABELS_)
        self.assertEqual(feedback, "Score: 0.62")
        for values in LABELS.values():
            for value in values:
                self.assertNotIn(value, feedback)

    def test_failed_prediction_scores_zero(self):
        self.assertEqual(score_feedback(None, LABELS_), "Score: 0.00")


if __name__ == "__main__":
    unittest.main()
