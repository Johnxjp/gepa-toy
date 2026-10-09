# GEPA design

Status: agreed, ready for implementation. Decisions D1 to D6 are listed in the appendix.

## 1. Context and problem

The project builds GEPA (reflective prompt optimisation) from scratch on a support-ticket classifier, then compares it with DSPy's `dspy.GEPA` on the same data and budget. The task, stages, rules and "done" criteria are in the [README](../README.md); this document does not restate them.

Today the repository can evaluate one fixed prompt on one split and save the run. It cannot swap the instruction text inside the prompt, score a handful of tickets without writing a file, cache model calls, count a budget, write feedback, call a reflection model, or keep a pool of candidates. Everything GEPA needs beyond "run a prompt and score it" is missing.

This matters to anyone implementing stages 1 to 4 or reading their results. The design fixes the interfaces so that stages 1, 2 and 3 differ only in two switches, and so that stage 4 (DSPy) is a fair comparison rather than a different experiment.

## 2. Goals and non-goals

**Goals**

- A GEPA core that knows only candidate strings, scores and feedback, with the ticket task plugged in behind a two-method interface.
- Stages 1, 2 and 3 run from one script, differing only in `feedback` and `selection`.
- Stage 4 runs DSPy's GEPA with the same seed instruction, splits, budget accounting, minibatch size, feedback text and reflection model.
- A way to tell whether a gap between stage 3 and stage 4 comes from the loop or from the reflection prompt wording.
- Every candidate logged with its parent, scores and wins, and the log survives a crash.

**Non-goals**

- Merging candidates from different lineages. Off in both our loop and DSPy (D3). A later experiment at most.
- LLM-written feedback. Templated feedback only (D6). See alternatives.
- Rewriting anything other than the instruction text. Label lists, ticket layout and JSON format stay fixed (D1).
- Running the DSPy-optimised instruction through our own template. Stage 4 is scored by running the DSPy program itself (D2).
- Changing the metric. `gepa_score` is defined in [ticket_scoring.md](ticket_scoring.md) and is not revisited here.

## 3. Current state

The evaluation path today:

```
scripts/evaluate.py --split X
        │
        ▼
evaluate.evaluate()            loads the split, builds a prompt per ticket with
        │                       zero_shot.build_prompt (instruction fixed in template)
        ▼
evaluate.predict()  ──► model.classify() ──► OpenRouter ──► model.parse_prediction()
        │                 (thread pool, timeout 60s, reasoning off, strict JSON parse)
        ▼
evaluate.gepa_score() per ticket, evaluate.summarise() per run
        │
        ▼
evaluate.save_run()            one JSON file under results/ with git commit,
                                dirty flag, model config and prompt template
```

| Component | Where | Notes |
| --- | --- | --- |
| Splits: train 100, pareto 50, test 300, label 50 | `data/.../splits/`, `load_split` | Fixed seed, saved to disk |
| Task-model call | `model.classify` | OpenRouter, timeout 60s, reasoning off |
| Strict reply parsing | `model.parse_prediction` | Failure keeps `error` and `raw` |
| Per-ticket score | `evaluate.gepa_score` | 0 to 1, mean of queue, type, priority, tag F1 |
| Transient vs fatal API errors | `evaluate.predict` | Transient errors score 0, setup errors crash |
| Run summary | `evaluate.summarise` | Accuracy and macro P/R/F1 per field |
| Run file with provenance | `evaluate.save_run` | Git commit, dirty flag, model, prompt template |
| Baseline prompts | `zero_shot.build_prompt`, `build_expanded_prompt` | Instruction text is fixed inside the template |

Two properties of this path shape the design. The instruction is baked into the template, so there is no way to run "this text" as a candidate. And `evaluate()` goes from split name to saved file in one step, so there is no way to score five tickets and get the results back in memory.

## 4. Proposed design

### 4.1 Architecture

```
gepa_toy/gepa.py         GEPA core. Knows candidates (strings), scores, feedback. No tickets.
gepa_toy/reflect.py      Reflector: generic reflection prompt + call to the reflection model.
gepa_toy/ticket_task.py  Ticket task: builds prompts, runs the model, scores, writes feedback.
gepa_toy/cache.py        SQLite cache of raw model replies keyed by input hash.
gepa_toy/dspy_task.py    Signature, metric wrapper, conversions between our rows and dspy.Example.
scripts/run_gepa.py      Runs one stage with one seed, saves the log, scores the best prompt on test.
scripts/run_dspy.py      Runs stage 4 with one seed, saves a log in our format, scores on test.
scripts/summarise_runs.py  Results table across runs.
```

```
                 ┌──────────────────────────────────────────────┐
                 │ gepa.optimise (core)                         │
                 │  pool · minibatches · accept · pareto · log  │
                 └───────┬──────────────────────────┬───────────┘
          task.evaluate  │                          │  propose
                         ▼                          ▼
              ┌─────────────────────┐     ┌──────────────────────┐
              │ TicketTask          │     │ reflect.make_reflector│
              │ build → complete →  │     │ prompt → complete →   │
              │ parse → score →     │     │ extract fenced block  │
              │ feedback            │     └──────────┬───────────┘
              └──────────┬──────────┘                │
                         ▼                           ▼
              ┌─────────────────────┐     ┌──────────────────────┐
              │ cache.Cache (SQLite)│     │ reflection model     │
              └──────────┬──────────┘     │ (not cached)         │
                         ▼                └──────────────────────┘
              ┌─────────────────────┐
              │ task model          │
              └─────────────────────┘
```

The core sees only strings and scores. Examples are opaque to it; it refers to them by position in `train` and `pareto`. To plug in a new kind of data later, write a new task with the same two methods. Nothing in `gepa.py` or `reflect.py` changes.

Stage 4 bypasses `gepa.py` and `reflect.py` entirely: DSPy supplies the loop and the reflector, and `dspy_task.py` supplies our signature, our metric and our feedback text.

### 4.2 Data model

**Rollout.** One per example evaluated:

```python
@dataclass
class Rollout:
    score: float      # 0 to 1
    trace: str        # what the reflector sees: inputs and output, as text
    feedback: str     # one line on the outcome, in the task's chosen feedback mode
    record: dict      # task-specific detail for the log (id, prediction, raw reply)
```

**RunLog.** Rewritten to `log_path` after each iteration, so a crash keeps everything up to that point:

- `metadata`: git version, settings, task description (`task.describe()`, a dict with model configs and feedback mode), split names, start time, runtime.
- `candidates`: id, text, parent id, iteration created, per-example pareto scores, mean pareto score, examples won.
- `iterations`: parent id, minibatch positions, parent and child scores and rollouts, child text, accepted or why not, calls used so far, reflection prompt and reply.
- `best`: id of the returned candidate.

Stage 4 converts DSPy's `detailed_results` (`candidates`, `parents`, `val_subscores`, `best_idx`) into the same `candidates` shape so lineage reads the same way.

**Cache.** One SQLite file at `config.CACHE_PATH` (under `results/`, gitignored):

- Key: SHA-256 of the full prompt text plus the model config fields that change the reply (model, temperature, max tokens, reasoning). Timeout is left out.
- Value: the raw reply text. Parsing and scoring rerun on every hit, so a change to the parser or score is picked up.
- Store only replies that came back from the API, including ones that fail to parse: at temperature 0 that is the model's real answer. Never store transient API errors.
- One connection guarded by a lock for the worker threads.
- The reflector is not cached. With a cache, three runs with the same reflection input would propose the same prompts and the spread would shrink to nothing.

**Test run metadata.** The test run saved by `run_gepa.py` records the stage, seed and log path.

### 4.3 Interfaces

Core (`gepa_toy/gepa.py`):

```python
class Task(Protocol):
    def evaluate(self, candidate: str, examples: list) -> list[Rollout]:
        """Run the candidate on each example. One rollout per example, same order."""


Propose = Callable[[str, list[Rollout]], str]  # parent candidate + its rollouts -> child


@dataclass
class Settings:
    budget: int                       # task-model calls, counted as examples evaluated
    minibatch_size: int = 5          # tickets are short, so 5 fit easily in the reflection prompt
    selection: Literal["pareto", "best"] = "pareto"
    seed: int = 0


def optimise(
    seed_candidate: str, train: list, pareto: list,
    task: Task, propose: Propose, settings: Settings,
    log_path: Path,
) -> RunLog: ...
```

Reflector (`gepa_toy/reflect.py`):

```python
def make_reflector(complete: Callable[[str], str]) -> Propose: ...
```

Ticket task (`gepa_toy/ticket_task.py`):

```python
class TicketTask:
    def __init__(self, model_config: ModelConfig, feedback: Literal["written", "score"], cache: Cache): ...
    def evaluate(self, candidate: str, rows: list[dict]) -> list[Rollout]: ...
    def describe(self) -> dict: ...
```

Changes to existing modules:

- `zero_shot.build(instruction, subject, body)`: the instruction goes at the top; the label lists, the ticket and the JSON format stay fixed below it. Insert the instruction by concatenation, not `str.format` or an f-string field, so braces written by the reflector can't break the template. `build_prompt` becomes `build(PLAIN_INSTRUCTION, ...)` and must return the same string as today. `SEED_INSTRUCTION` is the D5 seed.
- `evaluate.evaluate_rows(rows, build, model_config, workers) -> list[dict]` holds the thread pool. The file-based `evaluate()` calls it, so `scripts/evaluate.py` keeps working.
- `model.complete(prompt, model_config) -> str` is the API call and response-shape check split out of `classify`. The reflector uses it with its own `ModelConfig`. A bad response shape raises a transient error so the task records it as a failed prediction, as now.
- `evaluate.summarise` gains `queue_confusion`: rows are true queues, columns are predicted queues plus `(none)`, labels in `LABELS["queue"]` order.
- `scripts/run_gepa.py --stage {1,2,3,3c} --seed N` maps the stage to `feedback`, `selection` and reflection prompt, runs GEPA, saves the log under `config.GEPA_RESULTS_DIR`, then scores the best candidate on test with the existing evaluate and `save_run`.
- `scripts/summarise_runs.py` groups test runs by stage and prints mean and sample standard deviation of queue accuracy and `gepa_score`, with n per stage.

DSPy (`gepa_toy/dspy_task.py`, `scripts/run_dspy.py`), settings from the [DSPy GEPA API page](https://dspy.ai/current/api/optimizers/GEPA/overview/):

- Signature: inputs `subject`, `body`; outputs `queue`, `type`, `priority` as `Literal` of the label lists and `tags` as `list[Literal[...]]`. The signature's instructions are the seed instruction, word for word.
- Metric: wraps `gepa_score` and the written feedback function, returning `{"score": ..., "feedback": ...}`.
- Optimiser: `dspy.GEPA(metric, max_metric_calls=<same budget>, reflection_lm=<same reflection model>, reflection_minibatch_size=5, candidate_selection_strategy="pareto", skip_perfect_score=True, use_merge=False, track_stats=True, seed=N, log_dir=...)`, then `compile(program, trainset=train, valset=pareto)`.
- Stage 4 is scored by running the program `compile` returns on test with our metric (D2).

### 4.4 Key flows

**The loop** (`optimise`):

```
score seed on pareto                                  (costs len(pareto))
while budget left >= 2 * minibatch_size:
    parent  = select(pool)                            pareto: drop dominated, weight by wins; best: top mean
    batch   = next minibatch from train               epoch-shuffled, seeded
    parent_rollouts = task.evaluate(parent, batch)
    if every parent score is 1: log "skipped", continue
    child   = propose(parent, parent_rollouts)
    child_rollouts  = task.evaluate(child, batch)
    if sum(child scores) > sum(parent scores):        strictly greater
        if budget left < len(pareto): log "out of budget", stop
        score child on pareto, add to pool
    write log
return candidate with the highest mean pareto score   ties: earliest
```

**Pareto selection.** For each pareto example, the winners are every candidate with the top score on it (ties all win). Candidates that win nothing are out. Then drop dominated candidates: a candidate is dominated when another candidate still in the running wins every example it wins. Check candidates from lowest mean pareto score upwards, so of two with the same win set the weaker one goes. A surviving candidate's weight is the number of examples it wins. Sample a parent in proportion to weight with the seeded RNG. Dropping affects parent selection only: dominated candidates stay in the log and in the final pick, which is over all candidates. This follows the reference implementation's `remove_dominated_programs`.

**Greedy selection** (stages 1 and 2). The parent is always the candidate with the highest mean pareto score. Accepted children are still scored on pareto, so the final pick works the same way in every stage.

**Minibatches.** Shuffle train with the seed, take minibatches in order, reshuffle when exhausted. Every ticket is seen before any repeats, as in the reference implementation.

**Budget.** The core counts `len(examples)` on every `task.evaluate` call, whether or not the cache answers it. It never starts an evaluation that would go over the budget. Reflection calls are not counted. This matches DSPy's `max_metric_calls`, which counts one metric call per example evaluated, including the first scoring of the seed on the valset. With 1,000 calls: 50 to score the seed, 10 per iteration (parent and child on 5 tickets), and 50 more for each accepted child. With 10 accepted children that leaves room for about 45 iterations.

**One ticket through `TicketTask.evaluate`:**

```
row ──► zero_shot.build(candidate, subject, body)
    ──► cache lookup (hash of prompt + model config)
          miss ──► model.complete ──► store raw reply
          hit  ──► raw reply
    ──► model.parse_prediction ──► evaluate.gepa_score
    ──► Rollout(score, trace, feedback, record)
```

Trace: subject, body, and the model's prediction (or the one-line parse error).

**Feedback, written mode** (D6). A template filled from the score, the prediction and the true labels. No model call. In stage 4 DSPy calls our metric, which returns the same text, so stages 2 and 4 see identical feedback:

```
Score: 0.67 (queue 1, type 1, priority 0, tags F1 0.67)
Predicted: queue Technical Support, type Incident, priority low, tags [Network, Billing]
Correct:   queue Technical Support, type Incident, priority high, tags [Network, Outage]
```

A parse failure replaces the `Predicted` line with one line, such as `Reply was not valid: queue '<queue>' is not an allowed value.` The reflector also sees the ticket text, so it can work out why a label was wrong.

**Feedback, score mode.** `Score: 0.67` only, with no breakdown and no labels. This is close to DSPy's default when a metric gives no feedback.

**Reflection.** `make_reflector` builds one prompt: the current instruction, then for each rollout its `trace` and `feedback`, then a request to write an improved instruction inside a fenced block. It returns the text inside the last fenced block. We write our own reflection prompt (D4); stage 3c swaps in DSPy's default, copied.

**Seed instruction** (D5). The opening paragraph plus the README descriptions of each queue, type and priority (today's `build_expanded_prompt`). The descriptions live in the instruction, so GEPA can rewrite them; the fixed label lists below carry names only. Stage 0 is this seed, unchanged.

### 4.5 Failure modes

| Dependency | How it fails | Behaviour |
| --- | --- | --- |
| Task model (OpenRouter) | Transient error, timeout, 429 that outlasts the SDK retries | Prediction recorded as failed, scores 0, not cached. Many of these in a log mean the run is thrown away, not reported. |
| Task model | Setup error (bad key, unknown model) | Crash, as today. |
| Task model | Reply parses badly | Scored 0, raw reply cached: at temperature 0 it is the model's real answer. |
| Reflection model | Reply has no fenced block | Reflector raises. Iteration logged as "proposal failed", loop continues. The parent's minibatch calls are already spent. A reflector that keeps failing shows up in the log, and the budget still ends the run. |
| Reflection model | Transient error | Same as above: logged, loop continues. |
| Budget | Lands mid-iteration or mid-pareto-scoring | The core never starts an evaluation it can't afford. An accepted child with no budget left to score it ends the run with "out of budget". |
| Parent | Scores 1 on every minibatch example | No reflection call, iteration logged as skipped. |
| Cache | Two workers write the same key | Lock on the single connection: no exception, one stored row. |
| Process | Crash mid-run | Log is rewritten after every iteration; everything up to the crash is on disk. |
| DSPy adapter | Output format failure | DSPy may retry with a different adapter, and failures score `failure_score=0.0`. Behaviour differs from our strict JSON parse; recorded as a known difference. |
| DSPy and LiteLLM | `reasoning: {enabled: false}` not forwarded | Nemotron thinks for about 40 seconds and runs out of tokens before answering ([ticket_scoring.md](ticket_scoring.md) section 8). Checked with one call before any full run. |

## 5. Alternatives considered

- **Use DSPy only, skip the from-scratch loop.** Rejected: the README's purpose is to learn GEPA's mechanics well enough to explain it from memory, which a library call does not give. Stage 4 keeps DSPy as the reference point.
- **Merge on in both loops.** Rejected (D3): one more mechanism to build and to explain, and it muddies the comparisons between stages. Possible later experiment.
- **LLM-written feedback.** An extra model call per ticket explaining why the prediction was wrong. Rejected for now (D6): the reflector already sees the ticket, the prediction and the true labels, so the feedback model would mostly repeat the reflector's work. About 6% of labels look arbitrary, and a model asked to justify one will invent a reason, which pushes GEPA towards the oddly specific rules the README warns about. It would earn its place if the feedback model sees something the reflector doesn't (label definitions, training tickets with the same label), or on tasks without a single true label, such as playbook extraction in mini-Edra.
- **Minibatch-level precision, recall and F1 as feedback.** Rejected (D6): five tickets is too few for the numbers to mean anything, and DSPy's metric sees one ticket at a time, so stages 2 and 4 would get different feedback.
- **Let GEPA rewrite the whole template.** Rejected (D1): a candidate could break the reply format, and DSPy rewrites the instruction only, so the comparison would no longer be like for like.
- **Use DSPy's reflection prompt as ours.** Rejected (D4): writing our own is part of the learning. The cost is that a 3 vs 4 gap could come from prompt wording, which stage 3c isolates.
- **Score stage 4 by putting DSPy's instruction into our template.** Rejected for stage 4 (D2): it would measure our template, not DSPy's program. Kept as a possible later check.
- **Cache the reflection model too.** Rejected: three runs with the same reflection input would propose the same prompts and the spread across seeds would collapse.

## 6. Cross-cutting concerns

**Cost and rate limits.** Each stage is 3 runs of 1,000 calls plus 300 test calls: about 16,000 task-model calls across stages 0 to 4, before cache savings. OpenRouter limits free models per minute and per day. Check the current limits for the account before the first full run, and size `workers` to stay under the per-minute limit. A 429 that outlasts the SDK retries is recorded as a failed prediction, which would quietly lower scores.

**Performance.** `evaluate_rows` keeps the existing thread pool. The cache means a parent rescored on the same tickets costs no API calls, though it still costs budget.

**Observability.** The run log is the instrument. After a run, read: acceptance rate, pool size, best pareto score against calls used, count of failed predictions and failed proposals. `summarise_runs.py` gives mean and spread per stage. Stage 3, 3c and 4 logs are compared side by side on the same quantities.

**Metric reporting.** The README headline is queue accuracy, but GEPA optimises `gepa_score` across all four fields. Report both, or GEPA may look worse on the headline number than it is on its target.

**Backwards compatibility.** `build_prompt` stays byte-for-byte the same. `build_expanded_prompt` output changes because the descriptions move from the label lists into the instruction, so any earlier "expanded" baseline must be rerun. `scripts/evaluate.py` keeps working through `evaluate_rows`.

**Dependencies.** `uv add dspy`, pinned. Must install alongside the current `openai` version.

**Determinism.** Same seed with a fake task and fake reflector gives an identical log apart from timestamps. Stage 0 is deterministic at temperature 0, so its 3 runs will match and its spread is 0. DSPy's own cache is on by default, so each of its three runs needs a different `seed`. Seeds 0, 1, 2 for every stage.

## 7. Rollout and testing

### Phases

Every phase uses the same task model, reflection model, splits and budget: 1,000 task-model calls, optimising on train (minibatches of 5) and pareto, then scoring the result once on test. Each runs 3 times with seeds 0, 1 and 2. Build order follows the table: each phase adds one switch or one component to the previous.

| Phase | What | Feedback | Parent selection | Reflection prompt | Run with |
| --- | --- | --- | --- | --- | --- |
| 0 | Seed instruction, no optimisation | none | none | none | `scripts/evaluate.py` on test |
| 1 | Our GEPA, keep single best, bare score | score | best | ours | `scripts/run_gepa.py --stage 1` |
| 2 | Stage 1 with written feedback | written | best | ours | `scripts/run_gepa.py --stage 2` |
| 3 | Stage 2 with the Pareto pool | written | pareto | ours | `scripts/run_gepa.py --stage 3` |
| 3c | Control: stage 3 with DSPy's reflection prompt (one run only) | written | pareto | DSPy's default, copied | `scripts/run_gepa.py --stage 3c` |
| 4 | `dspy.GEPA`, merge off | written | pareto | DSPy's built-in | `scripts/run_dspy.py` |

The written feedback function (M5 below) is built before stage 2 and shared with stage 4, which depends on it through the metric wrapper.

### Build list

| # | Component | Why GEPA needs it | Phase |
| --- | --- | --- | --- |
| M1 | Prompt template with a swappable instruction | A candidate is text; the evaluation must run that text | 1 |
| M2 | Evaluate a list of rows, without writing a file | GEPA scores minibatches of 5 and the pareto set many times | 1 |
| M3 | Model-call cache keyed by input hash | Parent and child are rescored on the same tickets; README rule | 1 |
| M4 | Call budget counter | Same task-model budget per stage; README rule | 1 |
| M5 | Feedback per ticket, two modes: written and score-only | Stage 1 vs stage 2 is the key comparison | 1, 2 |
| M6 | Reflector: reflection prompt, stronger model, extract new instruction | Step 3 of GEPA | 1 |
| M7 | GEPA loop: minibatch, accept, pareto scoring, parent selection | Steps 1, 4, 5 | 1, 3 |
| M8 | Candidate log (lineage) | README rule: log prompt, parent, wins, score | 1 |
| M9 | Plain text completion call for the reflector | `classify` parses ticket JSON; the reflector returns free text | 1 |
| M10 | Queue confusion matrix on test | README metric | 0 |
| M11 | Summary across runs: mean and spread over 3 runs per stage | README results table | 0 |
| M12 | DSPy program, metric wrapper and run script | Stage 4 | 4 |

### How we know it works

Comparisons: 2 vs 1 (does written feedback help), 3 vs 2 (does the pool beat greedy), 4 vs 3 (how close is ours). 3c tells us whether a 4 vs 3 gap comes from the loop or the reflection prompt.

Checking the loop separately from the prompt, in order:

1. **Fake task, fake reflector.** The unit tests below. They prove the mechanics without a model call.
2. **Planted answer.** A fake task where an example scores 1 only if the candidate contains that example's keyword, with different examples needing different keywords. The fake reflector adds one keyword named in the feedback. Checks: the best mean pareto score reaches 1 within a known number of iterations, and with `selection="pareto"` the pool keeps candidates that win different examples, while `"best"` follows one line. No model calls.
3. **Control run.** One stage 3c run (about 1,000 task calls). If it scores close to stage 4, the loops match and any gap comes from our prompt. If it still falls well short, the gap is in the loop. Also compare the logs of stage 3, 3c and 4: acceptance rate, pool size, and best pareto score against calls used.

Success for the whole project is the README's "done means" list: a results table for stages 0 to 4 with mean and spread, three sentences answering the three comparisons, and the best prompt with a paragraph on what it learned.

### Rollback

A run whose log shows many API errors is discarded and rerun, not reported. Because the log is rewritten every iteration and the cache keeps every real reply, a rerun after a crash or a rate-limit storm repeats only the calls that never completed. The old `build_expanded_prompt` baseline is rerun rather than kept.

### Unit checks, stages 0 to 3c

Tests use fakes and never call a model, apart from the smoke run.

| Component | Check |
| --- | --- |
| Prompt builder | `build_prompt` is byte-for-byte unchanged. The seed prompt contains every README description and every label name exactly once. An instruction containing `{`, `}` and `{body}` appears verbatim and the ticket fields are still filled once. |
| `evaluate_rows` | `scripts/evaluate.py` on a fixed split gives the same `results` as before (with a stubbed model). |
| Cache | Same prompt and config: one call to the stubbed API, second answer from the cache. Different model, temperature, max tokens or reasoning: a miss. Transient error: not stored. Parse failure: stored. Concurrent workers on the same key: no exception, one stored row. |
| Budget | With a fake task, total examples evaluated never exceeds the budget, for budgets that land mid-iteration and mid-pareto-scoring. Cache hits are counted. The seed's pareto scoring is counted. |
| Acceptance | Child equal to parent on the minibatch is rejected. A rejected child is never scored on pareto and never enters the pool. |
| Pareto selection | Hand-written score matrix: winners per example include ties, weights equal win counts, a candidate that wins nothing is never picked, sampling frequencies over many draws match weights. |
| Dominated candidates | A candidate whose win set is a subset of another's is never picked. Two candidates with the same win set: only the one with the higher mean can be picked. A candidate that wins one example nobody else wins survives whatever its mean. A dominated candidate with the highest mean is still the final pick. |
| Greedy selection | Always picks the top mean; never picks anything else. |
| Final pick | Highest mean pareto score; earliest on a tie. |
| Skip perfect | Parent scoring 1 on every minibatch example: no reflection call, iteration logged as skipped. |
| Determinism | Same seed with a fake task and fake reflector gives an identical log (apart from timestamps). Different seeds give different minibatch orders. |
| Reflector | Extracts the last fenced block. No block: raises, and the loop logs "proposal failed" and continues. |
| Feedback | Table of cases (all right, each field wrong, tags missed and extra, parse failure) maps to the expected strings. Score mode never mentions a label. |
| Stage switches | Each stage's saved metadata matches its row in the phases table: 1, 2 and 3 differ only in `feedback` and `selection`, and 3c differs from 3 only in the reflection prompt. |
| Log | After a fake run, every accepted candidate has a parent that exists, and pareto wins are consistent with the stored scores. |
| Confusion matrix | Hand-made predictions, including a failed one, give the expected counts; row sums equal true-label counts. |
| Smoke run (real models) | Stage 3 with budget 80 completes, the log has at least one iteration, calls used ≤ 80, the cache file has entries, and the test run file links back to the log. |

### Checks, stage 4

| Component | Check |
| --- | --- |
| Metric wrapper | For the feedback test cases above, the DSPy metric returns the same score and the same text as our task. |
| Conversions | A train row round-trips to `dspy.Example` with the same inputs and labels; `with_inputs("subject", "body")` is set. |
| Seed instruction | The compiled program before optimisation has instructions equal to our seed instruction. |
| Reasoning flag | One real call through DSPy returns in a few seconds and uses well under the token limit. |
| Budget | After a run, `detailed_results` metric-call count is at most the budget. |
| Smoke run | Budget 80, completes, log converts to our shape, test run file links back to the log. |

## 8. Open questions

- **Rate limits.** The account's current per-minute and per-day limits on OpenRouter free models are unknown, and they set `workers` and whether 16,000 calls fit in the time available.
- **Reasoning flag through LiteLLM.** Whether `reasoning: {enabled: false}` reaches Nemotron when called as `openrouter/<model>` through DSPy. If it doesn't, stage 4 needs a different route to the model or a different model.
- **Token limit for DSPy's format.** DSPy's output format is longer than our JSON. Whether `max_tokens=300` is enough is checked with one call, and the answer may force a higher limit in stage 4 only, which is itself a difference to record.

## 9. Appendix and references

### Decisions

All agreed.

- **D1** GEPA rewrites the instruction text only. Label names, ticket and JSON format stay fixed, so a candidate can't break the reply format. This is also what DSPy rewrites.
- **D2** Stage 4 is scored by running the optimised DSPy program on test with our metric. Putting its instruction into our template is a possible later check.
- **D3** Merge is off in both: we don't build it, and stage 4 sets `use_merge=False`.
- **D4** We write our own reflection prompt, with one control run (3c) using DSPy's default copied into our reflector.
- **D5** The seed instruction is the opening paragraph plus the README descriptions of each queue, type and priority. `build_prompt` (plain) stays byte-for-byte the same; `build_expanded_prompt` changes.
- **D6** Written feedback is the score breakdown plus predicted and true labels, from a template with no model call.

### Known differences from DSPy, for the write-up

- DSPy's adapter formats fields with its own markers and parses its own output format, not our strict JSON, so format failures behave differently.
- DSPy's default minibatch is 3; both sides use 5 (`reflection_minibatch_size=5` set explicitly).
- `skip_perfect_score=True` is a DSPy default and our loop copies it.
- Dominance rule: our loop drops a candidate only when one other candidate wins every example it wins (section 4.4). The reference `remove_dominated_programs` uses a coverage rule: a candidate is dropped when every example it wins is also won by some other surviving candidate, not necessarily the same one. With A winning {0}, B winning {1} and C winning {0, 1} with the lowest mean, the reference drops C and ours keeps it.
- The reflector takes the last fenced block; the reference takes everything between the first and last fence.

### Related documents

- [README](../README.md): task, stages, rules, done criteria.
- [ticket_scoring.md](ticket_scoring.md): `gepa_score`, strict parsing, failed predictions, the Nemotron reasoning problem (section 8).
- [dataset_preparation.md](dataset_preparation.md): cleaning and splits.
- [observations.md](observations.md): label overlap seen while hand-labelling.

### External references

- GEPA paper: [Agrawal et al., 2025](https://arxiv.org/abs/2507.19457).
- Reference implementation: [gepa-ai/gepa](https://github.com/gepa-ai/gepa).
- DSPy: [GEPA optimizer API](https://dspy.ai/current/api/optimizers/GEPA/overview/), [GEPA in depth](https://dspy.ai/diving-deeper/gepa-in-depth/index.md).
