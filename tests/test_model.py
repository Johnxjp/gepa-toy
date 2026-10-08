import unittest

from gepa_toy.model import parse_prediction

VALID = (
    '{"queue": "IT Support", "type": "Incident", "priority": "high", "tags": ["HR"]}'
)


class ParsePredictionTest(unittest.TestCase):
    def test_plain_json(self):
        result = parse_prediction(VALID)
        self.assertIsNone(result.error)
        self.assertEqual(result.prediction.queue, "IT Support")
        self.assertEqual(result.prediction.tags, ["HR"])

    def test_json_in_code_fence_with_chatter(self):
        result = parse_prediction(f"Sure! Here you go:\n```json\n{VALID}\n```")
        self.assertEqual(result.prediction.queue, "IT Support")

    def test_no_json(self):
        result = parse_prediction("I cannot classify this ticket.")
        self.assertIsNone(result.prediction)
        self.assertEqual(result.error, "no JSON object in response")

    def test_malformed_json(self):
        result = parse_prediction('{"queue": "IT Support", "type": ')
        self.assertIsNone(result.prediction)
        self.assertIsNotNone(result.error)

    def test_unknown_label(self):
        result = parse_prediction(VALID.replace("IT Support", "Accounts"))
        self.assertIsNone(result.prediction)
        self.assertIn("queue", result.error)

    def test_missing_field(self):
        result = parse_prediction('{"queue": "IT Support"}')
        self.assertIsNone(result.prediction)

    def test_raw_text_is_kept(self):
        self.assertEqual(parse_prediction("nope").raw, "nope")


if __name__ == "__main__":
    unittest.main()
