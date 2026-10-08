import os
import re
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from gepa_toy import config
from gepa_toy.config import LABELS

load_dotenv(config.PROJECT_ROOT / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ModelConfig(BaseModel):
    model: str = "openai/gpt-4.1-mini"
    temperature: float = 0.0
    max_tokens: int = 300


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


def classify(prompt: str, model_config: ModelConfig) -> Result:
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"]
    )
    response = client.chat.completions.create(
        model=model_config.model,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return parse_prediction(response.choices[0].message.content or "")
