import os
import re
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from gepa_toy import config
from gepa_toy.config import LABELS

load_dotenv(config.PROJECT_ROOT / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ModelConfig(BaseModel):
    model: str
    temperature: float = 0.0
    max_tokens: int = 1000
    # Off for the task model: on, Nemotron spends ~1,300 tokens and ~40s per ticket thinking.
    reasoning: bool = False
    # The SDK default is 600s, retried twice: one hung call could stall a worker for 30 minutes.
    timeout: float = 60.0


def reflection_config(model: str = config.REFLECTION_MODEL) -> ModelConfig:
    # Reasoning used 4,800 to 7,300 tokens on a real reflection prompt; 16,000 leaves room
    # for the instruction itself. Temperature 1 so seeds explore different rewrites.
    return ModelConfig(
        model=model, temperature=1.0, max_tokens=16000, reasoning=True, timeout=300.0
    )


class UnexpectedResponseError(Exception):
    """A 200 reply with no usable answer: an API failure, not a model mistake."""


class Prediction(BaseModel):
    queue: Literal[tuple(LABELS["queue"])]
    type: Literal[tuple(LABELS["type"])]
    priority: Literal[tuple(LABELS["priority"])]
    tags: list[Literal[tuple(LABELS["tags"])]]


class Result(BaseModel):
    prediction: Prediction | None
    raw: str
    error: str | None = None


def parse_prediction(raw: str) -> Result:
    # Models often wrap JSON in ```json fences or add a sentence around it.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return Result(prediction=None, raw=raw, error="no JSON object in response")
    try:
        prediction = Prediction.model_validate_json(match.group())
    except ValidationError as e:
        return Result(prediction=None, raw=raw, error=str(e))
    return Result(prediction=prediction, raw=raw)


def complete(prompt: str, model_config: ModelConfig) -> str:
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=model_config.timeout,
    )
    response = client.chat.completions.create(
        model=model_config.model,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"reasoning": {"enabled": model_config.reasoning}},
    )
    # OpenRouter can return 200 with an error body and no choices when the upstream provider
    # fails, or a choice with finish_reason "error" and no content. The SDK hands back a
    # plain str or None for non-JSON or null bodies. An empty reply counts as one of these:
    # a reply cut off part-way still has text and is scored as a model mistake.
    # The SDK doesn't validate the body either, so a choice can arrive with no message.
    choices = response.choices if isinstance(response, ChatCompletion) else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    content = getattr(getattr(choice, "message", None), "content", None)
    if choice is None or choice.finish_reason == "error" or not content:
        raw = (
            response.model_dump_json()
            if isinstance(response, ChatCompletion)
            else str(response)
        )
        raise UnexpectedResponseError(raw)
    return content


def classify(prompt: str, model_config: ModelConfig) -> Result:
    return parse_prediction(complete(prompt, model_config))
