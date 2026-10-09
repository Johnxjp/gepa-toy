# Ticket scoring

How we score the classifier's answer for each ticket and for a whole run, and why. The code is in `gepa_toy/model.py` (calling the model, parsing the reply) and `gepa_toy/evaluate.py` (scoring, saving).

Run an evaluation with:

```sh
uv run python -m scripts.evaluate --split train          # or pareto, test, label_key
uv run python -m scripts.evaluate --split pareto --model openai/gpt-4.1-mini --max-tokens 500
```

## 1. Score each ticket on all four fields

**Decision:** the model predicts queue, type, priority and tags for every ticket. Each ticket gets a `gepa_score` from 0 to 1: the mean of exact match on queue, type and priority, plus F1 between the predicted and true tag sets.

**Why:** GEPA works with one score per ticket. Those scores build the Pareto frontier, and a child prompt is accepted only if its scores on a minibatch beat its parent's. Scoring all four fields gives the reflector more to learn from than queue alone. The tag F1 is a fraction rather than right or wrong, so scores spread out and candidates tie less often on pareto tickets.

**Example:** queue and type right, priority wrong, and one of two true tags predicted with no extras gives (1 + 1 + 0 + 2/3) / 4 = 0.67.

## 2. Use `tag_groups` as the true tags

**Decision:** true tags come from the `tag_groups` column, split on commas. The raw `tags` column is not used for scoring.

**Why:** the prompt lists the 20 tag groups from `labels.json`, so those are the values the model can choose. Every value in the train, pareto and test splits is in `labels.json`, and every ticket has at least one group. 445 of the 450 tickets in those splits have more than one group, so the split on commas matters: a test checks it.

## 3. Parse the reply strictly

**Decision:** a reply counts only if it contains a JSON object whose four fields all use values from `labels.json`, spelled and cased exactly. Anything else makes the whole prediction fail. There is no normalisation, not even of case. Code fences or a sentence around the JSON are allowed: the parser takes the first `{...}` block.

**Why:** the prompt lists every allowed value and asks for exact spelling, so a wrong value is a real failure to follow instructions. Following the format is part of what GEPA should learn. When we checked the failures, they were real mistakes and not near-misses: tag names used as the queue, the template copied back (`{"queue": "<queue>", ...}`), or tags written without list brackets.

**Considered:**

- **Check each field separately**, so one bad tag loses only the tag score. Rejected: it hides format problems that a prompt should fix.
- **Drop unknown tags and keep the rest.** Rejected for the same reason.
- **Match labels without regard to case.** Rejected: no failure we saw was a case slip, and it adds a rule to maintain.

## 4. A failed prediction scores 0 and keeps its evidence

**Decision:** when a reply can't be used, `predicted` is `null`, the ticket's `gepa_score` is 0, and it counts as wrong on every field. The reason goes in `error` and the reply exactly as it came back goes in `raw`.

**Why:** the reflector needs to see what went wrong to propose a fix. `error` is currently pydantic's full message, which is long. The written-feedback step should turn it into one line (for example "queue '<queue>' is not an allowed value; missing type, priority, tags") and leave the full message in the saved run.

## 5. Record only temporary API failures

**Decision:** connection errors, timeouts, rate limits (429) and server errors (5xx) that still fail after the SDK's two retries are recorded as a failed prediction. Any other error stops the run, including a bad request (400), a bad key (401), a blocked model (403), an unknown model (404) and a missing API key. A reply with the wrong shape (no `choices`, an HTML page, a `null` body) is recorded as `"unexpected response shape"`.

**Why:** one flaky call shouldn't sink a run of hundreds of tickets. A setup mistake must stop the run: otherwise every ticket scores 0 and the saved run looks like a real result. A GEPA loop would then score every candidate 0 and keep going.

**Also:** each call has a 60-second timeout. The SDK default of 600 seconds, tried three times, meant one hung call could hold a worker for 30 minutes.

## 6. Summarise a run for reporting

**Decision:** the `results` section of a run holds:

| Field | Metrics |
| --- | --- |
| `queue`, `type`, `priority` | accuracy, and macro precision, recall and F1 |
| `tags` | set precision, recall and F1, averaged over tickets |
| `gepa_score` | mean of the per-ticket scores |
| `n`, `errors` | number of tickets, and how many failed |

**Why macro:** the splits are balanced across queues, so every class should count equally.

**Why only classes present in the labels:** a class missing from a small split would otherwise score 0 and pull the average down. Failed predictions are scored as a placeholder class, `(none)`, so they count as a miss for the true class.

**Why tags per ticket:** it matches how tags enter `gepa_score`, so the run's `gepa_score` equals the mean of queue, type and priority accuracy plus tags F1. We checked this on a real run.

**What GEPA uses:** only the per-ticket scores and their mean. GEPA picks the final prompt by its mean score on the pareto set. Precision, recall and F1 are for the results table and for spotting failures, such as a queue the model never predicts.

**Considered:**

- **Pooled (micro) tag scores** over all tags. Rejected: tickets with many tags would weigh more, and the run total would no longer match the per-ticket scores.
- **Weighted averaging** for queue, type and priority. Rejected because the splits are balanced.

## 7. Save each run as one traceable file

**Decision:** each run is saved as `results/<timestamp>_<split>.json` with three keys:

- `metadata`: start time, runtime, git commit, a `dirty` flag with the list of changed files, the data file, the model config and the prompt template (with `{subject}` and `{body}` placeholders).
- `predictions`: one record per ticket with `id`, `predicted`, `labels`, `gepa_score`, `error` and `raw`.
- `results`: the summary from section 6.

**Why:** any number in the results table must trace back to the code, data, model settings and prompt that produced it. Once GEPA starts rewriting prompts, a run without its prompt can't be explained.

**Details:**

- Metadata is gathered before any model call, so a git failure doesn't waste a paid run.
- `dirty` counts untracked files, so new uncommitted code isn't recorded as a clean commit. `results/` is left out because it is gitignored output.
- The timestamp includes microseconds, so parallel runs (three per stage) don't overwrite each other.

## 8. Task model: Nemotron with reasoning off

**Decision:** the default task model is `nvidia/nemotron-3.5-lightning:free`, with reasoning turned off.

**Why:** the README asks for a cheap task model. With reasoning on, Nemotron took about 40 seconds and 1,300 tokens per ticket, and ran out of the 300-token limit before writing the JSON. With reasoning off it took about 1 second and 31 tokens, and gave the same answer on our check. The reflector will use a separate, stronger model, probably with reasoning on.

**Baseline on train (zero-shot prompt):**

| Model | Queue accuracy | `gepa_score` | Failed replies |
| --- | --- | --- | --- |
| `nvidia/nemotron-3.5-lightning:free`, reasoning off | 0.17 | 0.34 | 2 |
| `openai/gpt-4.1-mini` | 0.25 | 0.41 | 0 |

**Considered:**

- **`thinkingmachines/inkling-small:free`.** Blocked: OpenRouter allows it only inside approved apps, not direct API calls (403).
- **`openai/gpt-4.1-mini`.** Works and scores higher, but it isn't free. It stays available through `--model`.

## Open decisions

- **What GEPA rewrites:** the instruction text only, with the label lists, ticket and JSON format fixed, or the whole prompt. Leaning towards instruction text only: it can't break the format, and it's closer to how DSPy's GEPA rewrites instructions.
- **Scoring a minibatch:** `evaluate` takes a whole split file. GEPA needs to score a list of tickets, without writing a run file for every call.
- **Cache and call budget:** model calls aren't cached by input hash yet, and nothing counts calls against the stage budget. We still need to decide whether cache hits count against the budget.
- **Final reporting:** a confusion matrix for test, and the mean and spread over three runs per stage.
