# Model refactor

Status: notes, not yet designed. To be reviewed after the GEPA class refactor lands.

## Why

The GEPA class refactor needs the loop to call the reflection model. That raised a wider question: the code has no single object for "a model". A model is a `ModelConfig` (settings) plus the function `model.complete(prompt, config)`, and the two travel separately through the code. Callers that need caching or a test fake pass a third thing, a `complete` callback.

The aim is one `Model` object that holds its settings and can be called, so it can be passed in wherever a model is needed and replaced with a fake in tests.

## What exists today

`gepa_toy/model.py`:

- `ModelConfig`, a pydantic model: `model`, `temperature`, `max_tokens`, `reasoning`, `timeout`.
- `reflection_config(model)`, which builds the settings for the reflection model.
- `complete(prompt, model_config) -> str`, which makes the OpenRouter call and raises `UnexpectedResponseError` on a 200 reply with no usable answer. It builds a new `OpenAI` client on every call.
- `classify(prompt, model_config) -> Result`, which is `parse_prediction(complete(...))`.

Where the config and the call are used:

| Place | What it does with the model |
| --- | --- |
| `gepa_toy/cache.py` | `Cache.key` hashes the config fields (not `timeout`). `Cache.complete(prompt, config, complete)` takes the real call as a callback |
| `gepa_toy/evaluate.py` | `predict`, `evaluate_rows` and `evaluate` take `model_config` plus an optional `complete: Callable[[str, ModelConfig], str]`, falling back to `model.complete` |
| `gepa_toy/ticket_task.py` | Holds a `ModelConfig` and builds a lambda that routes calls through the cache. Looks up `model.complete` per call so tests can patch it. `describe()` logs `model_config.model_dump()` |
| `gepa_toy/dspy_task.py` | `make_lm(model_config, callbacks)` reads the config fields to build DSPy's LM. It never calls `model.complete` |
| `gepa_toy/gepa.py` | `GEPA._propose` builds the reflection prompt and calls `model.complete(prompt, self.reflection_model)`, looked up at call time so tests can patch it |
| `scripts/run_gepa.py` | Builds task and reflection configs, runs a flight test with `model.complete("Reply with OK.", ...)`, wraps `model.complete` in a lambda for the cache, logs `reflection_config.model_dump()` |
| `scripts/run_dspy.py`, `scripts/evaluate.py` | Build `ModelConfig` from command-line arguments |

Tests patch `gepa_toy.model.complete` in about 25 places (`test_evaluate.py`, `test_ticket_task.py`, `test_run_gepa.py`). `test_cache.py` passes fake `complete` callbacks to `Cache.complete`, including concurrency and "first stored reply wins" cases.

## Proposed shape

`ModelConfig` and `Model` merge into one class. The settings become fields of `Model`, so there is no separate config object to keep in step:

```python
class Model(BaseModel):
    model: str
    temperature: float = 0.0
    max_tokens: int = 1000
    reasoning: bool = False
    timeout: float = 60.0

    def complete(self, prompt: str) -> str: ...
    def classify(self, prompt: str) -> Result: ...
```

Everything that read the config reads the same fields from `Model`: `Cache.key`, `make_lm`, and the `model_dump()` calls that log settings. The logged JSON keeps the same shape.

The alternative is a `Model` that wraps a `ModelConfig` and exposes it as `.config`. That keeps two classes for one idea, and every reader of the settings writes `model.config.temperature`. Merging is simpler.

## Benefits

- One object to pass instead of a config and a callback. The `complete` parameters on `predict`, `evaluate_rows` and `evaluate`, and the lambda in `TicketTask`, go away.
- `GEPA` can take the reflection model as a parameter and call `reflection_model.complete(prompt)` without importing `model.py`. Tests pass a small fake instead of patching.
- The task model and the reflection model are built and passed the same way.
- The OpenAI client can be built once per model instead of once per call.

## Costs

- A wide diff: `model.py`, `cache.py`, `evaluate.py`, `ticket_task.py`, `dspy_task.py`, `reflect.py` (or its replacement in the GEPA refactor), all three scripts, and most of the tests listed above.
- Without the client and cache living inside it, `Model` is settings plus one method, which `functools.partial(model.complete, model_config=config)` already gives.

## Open questions

- **Where the cache goes.** Today only task-model calls are cached; reflection calls are not. Options: `Model` takes an optional `Cache`, or a separate caching wrapper with the same `complete(prompt)` method. `Cache.complete` currently takes the real call as a callback, and its tests depend on that, so its shape changes either way.
- **Holding the client on a pydantic model.** The `OpenAI` client would need to be a private attribute so that `model_dump()`, the cache key and equality ignore it. It must also be safe to share across the worker threads in `evaluate_rows`.
- **Naming.** `Model` is a pydantic `BaseModel` with a field called `model` and methods prefixed `model_`, inside a module called `model`. Something like `LLM` may read better.
- **`reflection_config()`.** Becomes a function or class method that returns a `Model` with the reflection settings.
- **Tests.** Whether tests switch to passing fake models everywhere, or keep patching (`patch.object(Model, "complete")`) where a fake would mean threading a new parameter through.
- **Error types.** `Model.complete` should turn provider errors into project error types, at least "temporary" (rate limit, timeout, connection, 5xx) and "unusable reply" (empty reply), so callers such as `GEPA` can decide what to skip without knowing OpenAI's exception classes. See the follow-up step in [gepa_class_refactor.md](gepa_class_refactor.md).
- **Order.** Planned after the GEPA class refactor. That refactor has `GEPA` take `reflection_model: ModelConfig` and call `model.complete`, which becomes `reflection_model.complete(prompt)` here.
