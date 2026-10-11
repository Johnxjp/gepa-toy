# GEPA class refactor

Status: agreed design, not yet implemented. Reviewed adversarially once; the fixes are folded in.

## Why

`gepa_toy/gepa.py` runs GEPA as one 130-line function, `optimise`. Its helpers are closures that share `calls_used`, `log` and `candidates` through `nonlocal`, candidates are untyped dicts, module functions such as `add_candidate` and `update_wins` change the list passed to them, and the reflector hands its prompt and reply back by setting attributes on itself (`propose.last_prompt`), which the loop reads with `getattr`. The refactor turns the run into a class, gives candidates a type, and moves the reflection step into the loop that owns it.

The behaviour of the algorithm does not change. A run with the same inputs and seed must produce the same candidates, iterations and log as today, apart from the metadata changes listed below.

## Design

```python
"""GEPA core: candidates are strings, examples are opaque."""

from gepa_toy import model
from gepa_toy.model import ModelConfig


@dataclass
class Rollout:
    score: float  # 0 to 1
    trace: str  # what the reflector sees: inputs and output, as text
    feedback: str  # the outcome as text, in the task's chosen feedback mode
    record: dict  # task-specific detail for the log (id, prediction, raw reply)


class Task(Protocol):
    def evaluate(self, candidate: str, examples: list) -> list[Rollout]: ...
    def describe(self) -> dict: ...


@dataclass
class Candidate:
    id: int
    text: str
    parent: int | None  # None for the seed
    iteration: int | None  # iteration that created it; None for the seed
    pareto_scores: list[float]  # one per pareto example, same order as the pareto list
    mean_pareto_score: float = field(init=False)  # computed in __post_init__; a field, so asdict logs it

    def __post_init__(self) -> None: ...


class CandidatePool:
    """Accepted candidates and their pareto scores. No model calls, budget or files."""

    def __init__(self) -> None: ...
    def add(
        self,
        text: str,
        parent: int | None,
        iteration: int | None,
        pareto_scores: list[float],
    ) -> Candidate: ...  # appends only; changes no other candidate
    def best(self) -> Candidate: ...
    def pareto_front(self) -> list[Candidate]: ...
    def wins(self) -> dict[int, set[int]]: ...  # per candidate id, the pareto examples it tops; recomputed each call
    def select_parent(
        self, selection: Literal["pareto", "best"], rng: random.Random
    ) -> Candidate: ...


class GEPA:
    def __init__(
        self,
        task: Task,
        reflection_model: ModelConfig,
        reflection_prompt: str,  # template with {instruction} and {rollouts}
        log_path: Path,
        *,
        budget: int,  # task-model calls, counted as examples evaluated
        minibatch_size: int = 5,
        selection: Literal["pareto", "best"] = "pareto",
        seed: int = 0,
    ): ...

    def run(self, seed_candidate: str, train: list, pareto: list) -> Candidate: ...

    def _step(self, parent: Candidate, batch_ids: list[int], train: list, pareto: list) -> dict: ...
    def _propose(
        self, parent: Candidate, rollouts: list[Rollout], iteration: dict
    ) -> str: ...
    def _evaluate(self, candidate: str, examples: list) -> list[Rollout]: ...
    def _budget_left(self) -> int: ...
    def _write_log(self) -> None: ...


def build_reflection_prompt(
    template: str, instruction: str, rollouts: list[Rollout]
) -> str: ...
def extract_instruction(reply: str) -> str: ...
def shuffled_positions(n: int, rng: random.Random) -> Iterator[int]: ...
```

Every non-trivial function and method gets a docstring: one line on what it does, then its parameters and return value, describing what the signature cannot say.

### Decisions

- **One class per run.** `__init__` only stores the settings. `run()` checks its inputs (the same checks as today: non-empty train and pareto, `minibatch_size >= 1`, `budget >= len(pareto)`) and builds all run state fresh: `calls_used = 0`, a new `CandidatePool`, the start time and the log. `rng` and the minibatch positions are locals of `run()`, so they cannot be reached before a run starts. Calling `run()` again starts a new run; it never continues the last one. Resuming can be added later if needed.
- **`run()` returns the best `Candidate`.** The finished run stays on the instance: `gepa.log` is the log dict (the same JSON-shaped data written to `log_path`, with candidates converted by `asdict`) and `gepa.pool` holds the candidates. Both are replaced by the next `run()`.
- **The log path belongs to the caller.** `log_path` is set in `__init__`, and every `run()` writes to it, so a second `run()` on the same object overwrites the first run's log file. To keep both, set `gepa.log_path` to a new path before calling `run()` again, or build a new `GEPA`. `run_gepa.py` builds one `GEPA` per run with a timestamped path, so it is not affected.
- **Attributes are set per instance in `__init__` or `run()`.** No class-level attributes, matching `TicketTask` and `Cache`.
- **`CandidatePool` stays separate** so the selection logic can be tested without a task, a model, a budget or a log file. Its methods replace the module functions that changed the candidate list passed to them.
- **Wins are computed when selecting, not stored on candidates.** A candidate's wins depend on every other candidate, so storing them meant rewriting every candidate on each `add`. `wins()` computes them from the scores, following the paper's selection: per pareto example find the top scorers (ties included), drop dominated candidates (`pareto_front()`), then sample weighted by win count (`select_parent()`). The front keeps candidate id order, so `rng.choices` draws exactly as before.
- **`Candidate` is a dataclass.** `asdict` gives the same keys in the same order as today's dicts, so the log JSON is unchanged. `mean_pareto_score` is computed from `pareto_scores` in `__post_init__` and cannot be passed in, so it can never disagree with the scores. It is a field with `init=False` rather than a `@property` because `asdict` skips properties and the log would lose it.
- **`Settings` is removed.** Its fields become keyword-only constructor arguments. The log keeps the `settings` block with the same keys.
- **`Propose` and `make_reflector` are removed.** Reflection is part of the algorithm, and only the model and the template vary between runs, so `GEPA` takes those two values and does the reflection itself in `_propose`:
  1. build the prompt with `build_reflection_prompt(self.reflection_prompt, parent.text, rollouts)` and write it to `iteration["reflection_prompt"]`
  2. call `model.complete(prompt, self.reflection_model)` and write the reply to `iteration["reflection_reply"]`
  3. return `extract_instruction(reply)`

  `_step` catches any exception from `_propose`, records "proposal failed" and the error, and carries on. Today's `propose` already ran the prompt build and the model call inside that same `except Exception`, so nothing new is swallowed. Writing the prompt and reply as they are produced means a failed proposal logs whatever it got: the prompt and `None` if the call fails, both if extraction fails. Each iteration starts with both set to `None`, so a value from an earlier iteration can never be logged against a later one.
- **Reflection errors keep today's handling in this refactor.** A follow-up step narrows which errors count as a failed proposal. See [Reflection model check](#reflection-model-check).
- **`model.complete` is looked up at call time** (`model.complete(...)`, not `from gepa_toy.model import complete`), so tests can patch `gepa_toy.model.complete`. Same pattern as `ticket_task.py`.
- **`build_reflection_prompt` and `extract_instruction` move into `gepa.py`.** They hold nothing ticket-specific, and `reflect.py` imports `Rollout` from `gepa.py`, so leaving them there would make a circular import. `reflect.py` keeps only the two templates, `REFLECTION_PROMPT` and `DSPY_REFLECTION_PROMPT`, because they mention support tickets. `model.py` imports only `config`, so `gepa.py` importing `model` makes no cycle.
- **`run()` does the sampling, `_step` does the work.** `run()` selects the parent and draws the minibatch positions, then `_step(parent, batch_ids, train, pareto)` evaluates, reflects and accepts or rejects. `_step` uses no randomness, and it looks the examples up from `batch_ids` itself, so the logged positions and the evaluated examples cannot disagree.
- **The random number generator is used in the same order as today.** In each iteration the parent is selected before the minibatch is drawn, on adjacent lines in `run()`. In pareto mode `rng.choices` draws even when the front has one candidate, and `shuffled_positions` is lazy, so swapping those two calls changes every minibatch.
- **The `metadata` parameter is removed.** `GEPA` logs everything it needs itself. The stage name and template name are our experiment's labels; the stage stays in the log filename and in the test-set evaluation file that `summarise_runs.py` reads.
- **Iterations stay plain dicts** with the same keys as today.
- **`ModelConfig` stays for now.** The planned `Model` class is a separate change, see [model_refactor.md](model_refactor.md). After it, `reflection_model` becomes a `Model` and `_propose` calls `self.reflection_model.complete(prompt)`.

### Reflection model check

`_step` treats every exception from `_propose` as a failed proposal and moves on. That is deliberate: a reflection model sometimes returns an empty reply or one with no fenced block, and one bad reply should not end a long run.

The same rule also catches configuration errors: a missing `OPENROUTER_API_KEY`, a wrong key (401), an unknown model name (404). With `GEPA` itself not checking the model, a misconfigured reflection model leads to this:

1. Every iteration selects a parent, evaluates it on a minibatch (`minibatch_size` task-model calls, counted against the budget and paid for unless cached), then fails to propose and logs "proposal failed".
2. No child is ever created, so the loop runs until the budget is spent on parent evaluations alone.
3. `run()` returns the seed as the best candidate. The log is complete and well formed, so the run looks like a real one in which GEPA found nothing better. Only the `error` field on each iteration shows otherwise.

Today `scripts/run_gepa.py` guards against this with a flight test before the run: `model.complete("Reply with OK.", reflection_config)`, which raises on a bad key or model. It only checks the state at the start, so running out of credits or a revoked key mid-run still falls into the silent loop.

This refactor keeps today's behaviour, so the old logs replay identically. The fix is the follow-up step below.

### Follow-up: classify reflection errors

The broad `except` treats three kinds of failure the same way. Each needs its own response:

| Kind | Examples | Response |
| --- | --- | --- |
| Unusable reply | no fenced block or an empty one (`ValueError` from `extract_instruction`), empty reply (`UnexpectedResponseError`, common when a reasoning model spends its whole token limit thinking) | record "proposal failed" and carry on |
| Temporary API trouble, after the SDK's own retries | `RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError` | record "proposal failed" and carry on |
| Broken setup or a bug | missing `OPENROUTER_API_KEY`, 401/403, unknown model (404), bad request (400), out of credits (402), `KeyError` or `TypeError` in our own code | crash |

The step:

1. `_step` catches only the first two kinds. Everything else propagates and ends the run, matching the task side, which already stops on any API error. A misconfigured reflection model then fails on the first reflection call, after the seed is scored on the pareto set (at most `len(pareto)` task calls, mostly cached).
2. The flight test stays in `run_gepa.py` as a cheap guard that fails before anything is spent. `GEPA` no longer depends on it to be correct.
3. No limit on consecutive failures, unless a real run shows a failure that gets past step 1. Real runs hit bad replies often (9 of 25 iterations in the latest stage 3 run), so a limit is hard to set safely.
4. Each kind is tested by sending a real error through the real call path with `tests/fake_openrouter.py`, not by raising the exception type we expect.
5. When the [`Model` refactor](model_refactor.md) happens, the list of OpenAI exception types moves out of `GEPA` into `Model`, which raises project error types for "temporary" and "unusable reply". `GEPA` then catches only those.

This step changes behaviour on purpose: a run that today ends with every proposal failed will instead crash. It is done after the class refactor, so the refactor can be verified by exact replay first.

### Log format

Iterations and `best` keep the same keys and values. Candidates keep theirs apart from `pareto_wins`, which is no longer logged: it was only the final state, and it can be recomputed from `pareto_scores`. `metadata` changes:

| Key | Before | After |
| --- | --- | --- |
| `git`, `settings`, `task`, `train_size`, `pareto_size`, `started`, `runtime_seconds` | logged by `optimise` | unchanged |
| `reflection_model` | passed in by `run_gepa.py` | logged by `GEPA` from `reflection_model.model_dump()`, same value |
| `reflection_prompt` | template name, `"ours"` or `"dspy"` | full template text |
| `seed` | passed in by `run_gepa.py` (duplicate of `settings.seed`) | removed; still in `settings` |
| `stage` | passed in by `run_gepa.py` | removed |

Nothing in the repository reads these keys back from GEPA logs apart from `tests/test_run_gepa.py`. Old logs keep their old metadata.

## Files that change

| File | Change |
| --- | --- |
| `gepa_toy/gepa.py` | The design above |
| `gepa_toy/reflect.py` | Keeps the two templates; `make_reflector`, `build_reflection_prompt`, `extract_instruction` and `FENCED_BLOCK` leave |
| `scripts/run_gepa.py` | Builds `GEPA(task, reflection_config, REFLECTION_PROMPTS[...], log_path, budget=..., minibatch_size=5, selection=..., seed=...)`, calls `best = gepa.run(SEED_INSTRUCTION, train, pareto)` and uses `best.text` and `best.id` (replacing `log["best"]` and `log["candidates"][best_id]["text"]` at lines 86-87). Keeps the flight test. Drops `metadata` |
| `tests/test_gepa.py` | `optimise(...)` becomes `GEPA(...).run(...)`, and tests read the log from `gepa.log`. Fake proposers become a patched `gepa_toy.model.complete` (see [Fakes](#fakes)). Selection tests use `CandidatePool`. Candidates are read as attributes |
| `tests/test_reflect.py` | `PromptTest` and `ExtractionTest` are rewritten against `build_reflection_prompt` and `extract_instruction` directly (some go through `make_reflector` today). `RecordingTest` is replaced by GEPA tests of the logged prompt and reply |
| `tests/test_run_gepa.py` | `patch.object(run_gepa, "optimise")` (line 227) becomes `run_gepa.GEPA`. Assertions on the GEPA log's `seed` (line 159) and template name (lines 163, 203) change to `settings.seed` and the full template text. Line 148 reads the test-set run file, which keeps `stage` |
| `AGENTS.md` | Add rows for `gepa_toy/gepa.py` and `gepa_toy/reflect.py` to the structure table (neither has one today) |
| `docs/model_refactor.md` | Its table still describes `make_reflector`; update it to the new `_propose` |

## How we verify it

### Fakes

The fake reflection model replaces today's fake proposers, which take `(parent, rollouts)`. Several of them use the rollouts, for example `add_keyword` picks its keyword from the first failing rollout's feedback, so the fake must see the rollouts, not only the parent text.

Tests use a template with both placeholders, such as `"{instruction}\n---\n{rollouts}"`. A helper parses the prompt back into the parent text and the `Example i\n{trace}\n{feedback}` blocks, calls the existing proposer function, and returns its result in a fenced block. Patched in as `gepa_toy.model.complete`, it gets `(prompt, model_config)`.

The same fake works with today's code through `make_reflector(lambda p: fake(p, cfg), template)`. Both versions then build the prompt and extract the reply with the same functions, so `reflection_prompt` and `reflection_reply` can be compared exactly between the old and new logs.

### 1. Golden logs from the current code

Before changing anything, a scratch script runs today's `optimise` (with `make_reflector` and the fake above) on a set of scenarios and saves each log as JSON. The scenarios cover every outcome and both selection modes:

- pareto selection over several iterations with accepted, rejected and skipped steps
- best selection
- a reflection call that raises, and a reply with no fenced block
- a run that ends "out of budget", and one that ends on the minibatch budget check

After the refactor the same scenarios run through `GEPA(...).run(...)`, and `gepa.log` is compared with the saved log. Candidates, iterations and `best` must match exactly, including `reflection_prompt` and `reflection_reply`. `metadata` must match apart from `started`, `runtime_seconds`, `git` and the changes in the log format table.

### 2. Replay of real runs, with no API calls

Real logs are replayed through the new code:

- `results/gepa/20261011-102602-489455_stage3_seed0.json`: pareto selection, clean tree at commit `aeed4b1`, 25 iterations with 14 accepted, 9 failed proposals (including empty replies and replies with no fenced block), 1 rejected and 1 out of budget
- `results/gepa/20261011-102542-928670_stage1_seed0.json`: best selection and score feedback. The run was stopped by hand, so the logged iterations are compared as a prefix and the replay ends at the first cache miss

How the replay runs:

- task-model calls go through the real `TicketTask` and are served from a **copy** of `results/cache.sqlite`, so a fault in the replay can never write into the experiment's cache
- `gepa_toy.model.complete` is patched so that a reflection call returns the reply logged for that prompt (no logged prompt appears twice), and any call that is neither in the log nor in the cache copy raises instead of reaching the API
- the reflection config is `model.reflection_config(REFLECTION_MODEL)`, which matches what both logs record

Before the refactor, the same replay runs through today's `optimise` and must reproduce both logs. The review did this: 0 misses on the stage 3 log with all 25 iterations equal, and every logged iteration equal on the stage 1 log. After the refactor, the replayed candidates and iterations must match the old logs exactly. This checks the real `TicketTask`, the real templates and real reflection replies, which the fakes cannot.

### 3. Unit tests

- All existing tests in `test_gepa.py` are ported, not dropped. Each still tests the same behaviour through the new API.
- New tests:
  - `run()` returns the candidate with the highest mean pareto score, the same one `gepa.log["best"]` names
  - `run()` called twice on one `GEPA` gives two runs whose logs match apart from `started` and `runtime_seconds`, and the second overwrites the first run's log file
  - a reflection call that raises logs the prompt and a `None` reply, and the loop continues
  - a reply with no fenced block logs both the prompt and the reply
  - the log metadata contains `reflection_model` and the full `reflection_prompt` text
  - the file at `log_path` equals `gepa.log`
- `CandidatePool` tests cover what the `pareto_front`, `select_parent`, `best_candidate` and `add_candidate` tests cover today.

### 4. Validation

- `uvx ruff format` and `uvx ruff check --fix`, with nothing left over.
- The full suite passes: `uv run python -m unittest discover -s tests` (166 tests pass before the change).

No script runs are part of this check. `run_gepa.py --stage 0` never builds a `GEPA`, and every run of it writes another test-set result that `summarise_runs.py` counts. A paid end-to-end run is not needed either: the golden logs and the replays cover the loop without API calls.
