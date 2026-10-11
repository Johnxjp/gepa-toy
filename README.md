# GEPA toy: ticket routing

Learn GEPA's mechanics by implementing it from scratch on a small, cheap task with a clean metric, then compare with DSPy's implementation.

## Task

Given a support ticket's subject and body, an LLM predicts four fields: queue, type, priority and tags. GEPA rewrites the LLM's instruction text to improve the score. No model weights change.

## Data

[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) on Hugging Face. Synthetic, about 62k tickets, English and German, labelled with queue, priority, type and tags. Licence CC BY-NC 4.0, fine for personal learning.

- English only.
- All 10 queues, including the overlapping ones (Technical Support, IT Support, Product Support). Roughly balanced.
- 500 tickets, balanced across queues, split once with a fixed seed and saved to disk:
  - **train (100):** minibatches that produce feedback for reflection
  - **pareto (50):** used to accept candidates and maintain the frontier
  - **test (300):** touched only for final numbers
  - **label (50):** hand-label these blind to estimate the achievable ceiling
- Expect some labels to be arbitrary: 6% of reworded ticket pairs disagree on queue.

See [docs/dataset_preparation.md](docs/dataset_preparation.md) for how the data is cleaned and split.

### Ticket format

Each ticket has a free-text `subject` (sometimes empty) and `body`. These are the only inputs the model sees. The labels are:

| Field | Values | How many to pick |
| --- | --- | --- |
| `queue` | 10, for example Technical Support, Billing and Payments, Human Resources | exactly one |
| `type` | 4: Incident, Request, Problem, Change | exactly one |
| `priority` | 3: low, medium, high | exactly one |
| `tags` | 20 tag groups, such as Performance or Data & Analytics | one or more |

The model replies with a JSON object, for example `{"queue": "Customer Service", "type": "Request", "priority": "low", "tags": ["Performance", "General Support"]}`. The reply is parsed strictly: any value outside the lists makes the whole prediction fail and score 0. The raw dataset has about 845 distinct tags, which we merged into the 20 groups. See [docs/dataset_preparation.md](docs/dataset_preparation.md).

### Why four labels and not just the queue

GEPA needs one score per ticket. A queue-only score is right or wrong, and with 10 overlapping queues the model is right only 20 to 30% of the time. Most tickets then tie at 0, so a child prompt has little to beat its parent on and the Pareto frontier has little to separate candidates with.

Scoring all four fields fixes both problems. Each ticket's `gepa_score` is the mean of exact match on queue, type and priority, plus the F1 between predicted and true tags. The tag F1 is a fraction, so scores spread out and ties are rarer. The feedback to the reflector also names more things that can be wrong. Hand-labelling 50 tickets pointed the same way: many tickets fit several queues or types and there was no way to say so. [docs/ticket_scoring.md](docs/ticket_scoring.md) has the details.

The cost is that the headline number moves less than the target. GEPA optimises `gepa_score`, so queue accuracy can stay flat while the score rises. Report both.

## Metric

Accuracy on test, plus a confusion matrix. Mean and spread over 3 runs per stage.

## GEPA in five steps

1. **Run** the current prompt on a minibatch of training tickets; keep the inputs and outputs.
2. **Feedback in words:** for each example, a sentence explaining the outcome. "Predicted Billing, correct is Technical Support; the ticket is about a failed update, not a charge."
3. **Reflect:** a stronger model reads the current prompt plus the feedback and proposes a new prompt.
4. **Accept** only if it beats its parent on the same minibatch, then score it on the full pareto set.
5. **Keep a diverse pool (Pareto frontier):** a candidate survives if it's best on at least one pareto example. Pick the next parent from the pool, weighted by how many examples it wins.

Stop when the budget of task-model calls runs out. Return the candidate with the best average on the pareto set.

## Lessons: where GEPA hit its limits on this dataset

We stopped after running DSPy's GEPA and our own loop once each. The labels carry too little signal for stage comparisons to be measurable. Details and run paths are in [docs/observations.md](docs/observations.md).

**The labels are mostly arbitrary.** The dataset is synthetic. On the 50 tickets I labelled blind, I agreed with the dataset's labels far less than the untuned model did:

| | Queue | Type | Priority | `gepa_score` |
| --- | --- | --- | --- | --- |
| My hand labels | 0.22 | 0.46 | 0.24 | 0.31 |
| Starting instruction, no optimisation | 0.30 | 0.68 | 0.30 | 0.42 |
| DSPy GEPA best candidate | 0.24 | 0.48 | 0.46 | 0.43 |
| Random queue (10 classes) | 0.10 | | | |

A human reading the ticket carefully scores 0.31, below the model with no optimisation. The model has likely picked up the generator's habits, and a better instruction has little real signal left to learn.

**DSPy's GEPA barely beat the starting instruction.** Its best candidate scored 0.43 against 0.42, a gain of 0.013 (0.3 times the spread). It traded fields: priority and tags rose, queue and type fell.

**Both GEPA runs memorised.** The feedback shows the true labels, so the reflector copied the minibatch tickets into the instruction. Children passed their minibatch check at 0.92 against their parents' 0.44, then scored no better on unseen tickets. The instruction grew from 207 to 1,876 words in DSPy's run and tenfold in ours, and picked up rules like sending "integration or data-exchange inquiries" to Returns and Exchanges.

**The one real gain was on tags.** Our best candidate rose from 0.397 to 0.465 `gepa_score` on the 300 test tickets, almost all from tags, while queue accuracy fell from 0.233 to 0.217. This is the clearest case for scoring several fields: the gain was real but sat in the field with the loosest labels.

**What to do differently:**

1. Measure the human ceiling first. Labelling 50 tickets was the cheapest test of whether the task had signal.
2. Size the sets for the gain you want to see. 50 pareto tickets can only show a gain of about 0.08, and 300 test tickets about 0.035. The pareto winner scored 0.499 on pareto and 0.432 on unseen tickets.
3. Keep true labels out of the feedback, test children on fresh tickets, or tell the reflector not to quote tickets. Otherwise the minibatch check only tests memorising.
4. Stop the run on any API error. DSPy scored every refused call 0 for 77 cycles after a rate limit.

## References

- Paper: [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457) (Agrawal et al., 2025; ICLR 2026 oral). Read sections on the algorithm and Pareto selection first.
- Reference implementation: [gepa-ai/gepa](https://github.com/gepa-ai/gepa)
- DSPy docs: [GEPA optimizer API](https://dspy.ai/api/optimizers/GEPA/overview/), [GEPA in depth](https://dspy.ai/diving-deeper/gepa-in-depth/index.md), [Prompt optimizing with GEPA tutorial](https://dspy.ai/getting-started/gepa-optimization/)
- Dataset: [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)