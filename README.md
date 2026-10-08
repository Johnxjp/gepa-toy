# GEPA toy: ticket routing

Learn GEPA's mechanics by implementing it from scratch on a small, cheap task with a clean metric, then compare with DSPy's implementation.

This is a learning exercise, not a showcase. A classifier is the easy case for GEPA: one prompt, thin feedback. The real test comes later, on playbook extraction in mini-Edra.

## Task

Given a support ticket's subject and body, an LLM predicts which queue it belongs to. GEPA rewrites the LLM's instruction text to improve accuracy. No model weights change.

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

## Metric

Accuracy on test, plus a confusion matrix. Mean and spread over 3 runs per stage.

## GEPA in five steps

1. **Run** the current prompt on a minibatch of training tickets; keep the inputs and outputs.
2. **Feedback in words:** for each example, a sentence explaining the outcome. "Predicted Billing, correct is Technical Support; the ticket is about a failed update, not a charge."
3. **Reflect:** a stronger model reads the current prompt plus the feedback and proposes a new prompt.
4. **Accept** only if it beats its parent on the same minibatch, then score it on the full pareto set.
5. **Keep a diverse pool (Pareto frontier):** a candidate survives if it's best on at least one pareto example. Pick the next parent from the pool, weighted by how many examples it wins.

Stop when the budget of task-model calls runs out. Return the candidate with the best average on the pareto set.

## Stages

| Stage | What | Question it answers |
| --- | --- | --- |
| 0 | Hand-written baseline prompt | Number to beat |
| 1 | Reflection loop, keep single best, score-only feedback ("wrong") | How far plain iteration gets |
| 2 | Stage 1 with written feedback | Is written feedback the big win? |
| 3 | Stage 2 with the Pareto pool | Does diversity beat greedy? |
| 4 | DSPy `dspy.GEPA` on the same split and budget | How close is mine, and what does theirs add? |

Stage 2 vs stage 1 is the key comparison.

## Rules

- Same budget for every stage (for example, 1,000 task-model calls).
- Cheap model for the task, stronger model for reflection.
- Log every candidate: its prompt, parent, which examples it wins, its score. Read the lineage at the end.
- Cache model calls by input hash.
- Watch for overfitting to label noise: a prompt full of oddly specific rules ("invoices mentioning printers go to IT Support") is memorising, not learning.

## Done means

1. A results table: stages 0–4, mean and spread over 3 runs.
2. Three sentences: does written feedback beat a bare score; does the pool beat greedy; what DSPy's GEPA does that mine doesn't.
3. The best prompt, and one paragraph on what it learned.
4. Can explain GEPA from memory in 10 minutes, including how it differs from TextGrad and from RL (GRPO).

## Rough size

From-scratch GEPA: 150–250 lines of Python. One day for stages 0–3, half a day for stage 4.

## References

- Paper: [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457) (Agrawal et al., 2025; ICLR 2026 oral). Read sections on the algorithm and Pareto selection first.
- Reference implementation: [gepa-ai/gepa](https://github.com/gepa-ai/gepa)
- DSPy docs: [GEPA optimizer API](https://dspy.ai/api/optimizers/GEPA/overview/), [GEPA in depth](https://dspy.ai/diving-deeper/gepa-in-depth/index.md), [Prompt optimizing with GEPA tutorial](https://dspy.ai/getting-started/gepa-optimization/)
- Worked example: [Hugging Face cookbook, DSPy GEPA](https://huggingface.co/learn/cookbook/dspy_gepa)
- For the comparison: [TextGrad](https://github.com/zou-group/textgrad)
- Dataset: [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)