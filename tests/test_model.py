import os
import unittest
from unittest.mock import patch

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from gepa_toy.model import (
    OPENROUTER_BASE_URL,
    Model,
    UnexpectedResponseError,
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
class CompleteTest(unittest.TestCase):
    def reply(self, openai, response):
        openai.return_value.chat.completions.create.return_value = response
        return parse_prediction(Model(model="m").complete("prompt"))

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
        self.assertEqual(Model(model="m").complete("prompt"), "any text")

    def test_sends_config_to_openrouter(self, openai):
        model = Model(
            model="some/model",
            temperature=0.7,
            max_tokens=123,
            timeout=9.0,
            reasoning=True,
        )
        create = openai.return_value.chat.completions.create
        message = ChatCompletionMessage(role="assistant", content=VALID)
        create.return_value = completion(
            [Choice(index=0, finish_reason="stop", message=message)]
        )
        model.complete("the prompt")
        openai.assert_called_once()
        kwargs = openai.call_args.kwargs
        self.assertEqual(kwargs["base_url"], OPENROUTER_BASE_URL)
        self.assertEqual(kwargs["api_key"](), "test")
        create.assert_called_once_with(
            model="some/model",
            temperature=0.7,
            max_tokens=123,
            messages=[{"role": "user", "content": "the prompt"}],
            extra_body={"reasoning": {"enabled": True}},
            timeout=9.0,
        )

    def test_empty_or_errored_reply_stops_the_run(self, openai):
        # OpenRouter reports a provider failing after the request was accepted as a 200
        # with finish_reason "error"; a reply with no text is treated the same way.
        for content, finish in ((None, "length"), ("", "stop"), (VALID, "error")):
            message = ChatCompletionMessage(role="assistant", content=content)
            choice = Choice.construct(index=0, finish_reason=finish, message=message)
            with self.assertRaises(UnexpectedResponseError):
                self.reply(openai, completion([choice]))

    def test_reply_cut_off_part_way_is_a_model_mistake(self, openai):
        message = ChatCompletionMessage(role="assistant", content='{"queue": "IT Sup')
        choice = Choice(index=0, finish_reason="length", message=message)
        result = self.reply(openai, completion([choice]))
        self.assertIsNone(result.prediction)
        self.assertEqual(result.error, "no JSON object in response")

    def test_no_choices_stops_the_run(self, openai):
        for choices in ([], None):
            with self.assertRaises(UnexpectedResponseError) as caught:
                self.reply(openai, completion(choices))
            self.assertIn('"id":"x"', str(caught.exception))

    def test_choice_without_message_stops_the_run(self, openai):
        # The SDK builds the response without validating it, so a choice can carry no message.
        for choices in (
            [Choice.construct(index=0, finish_reason="stop", message=None)],
            [Choice.construct(index=0, finish_reason="stop")],
            "oops",
        ):
            with self.assertRaises(UnexpectedResponseError):
                self.reply(openai, completion(choices))

    def test_non_json_or_null_body_stops_the_run(self, openai):
        # The SDK returns the body as a str for non-JSON content types, and None for `null`.
        for response, raw in (
            ("<html>Bad gateway</html>", "<html>Bad gateway</html>"),
            (None, "None"),
        ):
            with self.assertRaises(UnexpectedResponseError) as caught:
                self.reply(openai, response)
            self.assertEqual(str(caught.exception), raw)


class ModelTest(unittest.TestCase):
    def test_model_builds_without_a_key_and_the_first_call_fails(self):
        with patch.dict(os.environ, clear=True):
            model = Model(model="m")
            with self.assertRaises(KeyError):
                model.complete("prompt")

    def test_config_has_the_logged_shape(self):
        # Every run's metadata logs this dict; earlier runs used these exact keys.
        self.assertEqual(
            Model(model="m").config,
            {
                "model": "m",
                "temperature": 0.0,
                "max_tokens": 1000,
                "reasoning": False,
                "timeout": 60.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
