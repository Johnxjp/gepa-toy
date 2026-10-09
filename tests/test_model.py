import os
import unittest
from unittest.mock import patch

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from gepa_toy.model import (
    OPENROUTER_BASE_URL,
    ModelConfig,
    ResponseShapeError,
    classify,
    complete,
    parse_prediction,
)

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


def completion(choices):
    return ChatCompletion.construct(
        id="x", choices=choices, created=0, model="m", object="chat.completion"
    )


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"})
@patch("gepa_toy.model.OpenAI")
class ClassifyTest(unittest.TestCase):
    def reply(self, openai, response):
        openai.return_value.chat.completions.create.return_value = response
        return classify("prompt", ModelConfig())

    def test_valid_reply(self, openai):
        message = ChatCompletionMessage(role="assistant", content=VALID)
        choice = Choice(index=0, finish_reason="stop", message=message)
        result = self.reply(openai, completion([choice]))
        self.assertEqual(result.prediction.queue, "IT Support")
        self.assertEqual(result.raw, VALID)

    def test_complete_returns_the_reply_text(self, openai):
        message = ChatCompletionMessage(role="assistant", content="any text")
        choice = Choice(index=0, finish_reason="stop", message=message)
        openai.return_value.chat.completions.create.return_value = completion([choice])
        self.assertEqual(complete("prompt", ModelConfig()), "any text")

    def test_sends_config_to_openrouter(self, openai):
        model_config = ModelConfig(
            model="some/model",
            temperature=0.7,
            max_tokens=123,
            timeout=9.0,
            reasoning=True,
        )
        create = openai.return_value.chat.completions.create
        message = ChatCompletionMessage(role="assistant", content="")
        create.return_value = completion(
            [Choice(index=0, finish_reason="stop", message=message)]
        )
        classify("the prompt", model_config)
        openai.assert_called_once_with(
            base_url=OPENROUTER_BASE_URL, api_key="test", timeout=9.0
        )
        create.assert_called_once_with(
            model="some/model",
            temperature=0.7,
            max_tokens=123,
            messages=[{"role": "user", "content": "the prompt"}],
            extra_body={"reasoning": {"enabled": True}},
        )

    def test_empty_content_is_an_error(self, openai):
        message = ChatCompletionMessage(role="assistant", content=None)
        choice = Choice(index=0, finish_reason="length", message=message)
        result = self.reply(openai, completion([choice]))
        self.assertIsNone(result.prediction)
        self.assertEqual(result.error, "no JSON object in response")

    def test_no_choices_raises(self, openai):
        for choices in ([], None):
            with self.assertRaises(ResponseShapeError) as raised:
                self.reply(openai, completion(choices))
            self.assertIn("unexpected response shape", str(raised.exception))
            self.assertIn('"id":"x"', str(raised.exception))

    def test_choice_without_message_raises(self, openai):
        # The SDK builds the response without validating it, so a choice can carry no message.
        for choices in (
            [Choice.construct(index=0, finish_reason="stop", message=None)],
            [Choice.construct(index=0, finish_reason="stop")],
            "oops",
        ):
            with self.assertRaises(ResponseShapeError) as raised:
                self.reply(openai, completion(choices))
            self.assertIn("unexpected response shape", str(raised.exception))

    def test_non_json_or_null_body_raises(self, openai):
        # The SDK returns the body as a str for non-JSON content types, and None for `null`.
        for response, raw in (
            ("<html>Bad gateway</html>", "<html>Bad gateway</html>"),
            (None, "None"),
        ):
            with self.assertRaises(ResponseShapeError) as raised:
                self.reply(openai, response)
            self.assertEqual(str(raised.exception), f"unexpected response shape: {raw}")


if __name__ == "__main__":
    unittest.main()
