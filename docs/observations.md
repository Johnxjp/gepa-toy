# Ticket Observations
From labelling 50 tickets I noticed that many follow the same format suggesting a template. I also noticed that there is a lot of overlap between labels and the tags were often not very clear. It seems like for 'type' and queue there's a missing label which is 'notification'. For 'type', I also hardly used change and didn't really know what it was for. I also couldn't judge clearly the difference between problem and incident.

Finally, there are some tickets that tick multiple boxes but there's no option for that.

## Outcome: experiment stopped after stage 4 (2026-10-11)

We stopped after running DSPy's GEPA (stage 4) once. On this dataset no stage comparison would be measurable, so stages 1 and 2 were not run. Stage 3, our own loop, was later run once to check it end to end (below).

### My labels against the dataset's

Scored on the 50 label tickets with `gepa_score` (docs/ticket_scoring.md):

| | Queue | Type | Priority | `gepa_score` |
| --- | --- | --- | --- | --- |
| My hand labels | 0.22 | 0.46 | 0.24 | 0.31 |
| Starting instruction, no optimisation | 0.30 | 0.68 | 0.30 | 0.42 |
| DSPy GEPA best candidate, 17 cycles | 0.24 | 0.48 | 0.46 | 0.43 |
| Random queue (10 classes) | 0.10 | | | |

The model agrees with the dataset's labels more than I do. The labels are synthetic and largely arbitrary, which matches the notes above, so a better instruction has little real signal to learn.

### What GEPA did

Run: `results/gepa/20261009-214907-448395_stage4_seed0` (gitignored). Paid Nemotron as task model, DeepSeek flash for reflection, budget 1,000 calls, minibatch 5, merge off. It cost $0.32 and took 33 minutes.

- **It memorised.** The feedback shows the true labels, so the reflector copied the 5 minibatch tickets and their labels into the instruction. All 17 children passed their minibatch, scoring 0.92 on average against their parents' 0.44. The instruction grew from 207 to 1,876 words.
- **No gain on unseen tickets.** On each parent's next, unseen minibatch, scores stayed between 0.28 and 0.61 with no upward trend. On the 50 label tickets the best candidate beat the starting instruction by +0.013 in `gepa_score`, 0.3 times its spread. It traded fields: priority and tags improved, queue and type got worse.
- **The pareto set picked on noise.** Of the 16 candidates, only 2 came near 2 spreads above the starting instruction on the 50 pareto tickets, about what luck gives across 16 comparisons. The winner scored 0.499 on pareto but 0.432 on unseen tickets.
- **Every acceptance cost 50 extra calls** to score on pareto, so the budget allowed 17 cycles, not the ~95 expected.

### Stage 3: our own loop, one run (2026-10-11)

Run: `results/gepa/20261011-102602-489455_stage3_seed0.json`, test run `results/20261011-111233-018340_test.json` (both gitignored). `scripts/run_gepa.py --stage 3 --seed 0 --budget 1000`: written feedback, Pareto parent selection, our reflection prompt. Task model `nvidia/nemotron-3.5-lightning` (reasoning off, up to 1,000 output tokens). Reflection model `~deepseek/deepseek-flash-latest`, which resolved to `deepseek/deepseek-v4.1-flash` (reasoning on, temperature 1, up to 16,000 tokens). It took 46 minutes. The winning prompt is in [stage3_best_prompt.txt](stage3_best_prompt.txt).

Scored on the 300 test tickets, against the starting instruction:

| | Queue | Type | Priority | Tag F1 | `gepa_score` |
| --- | --- | --- | --- | --- | --- |
| Starting instruction, no optimisation | 0.233 | 0.593 | 0.403 | 0.358 | 0.397 |
| Stage 3 best candidate | 0.217 | 0.657 | 0.417 | 0.568 | 0.465 |

- **The score rose, the headline didn't.** `gepa_score` rose by 0.068, about twice the smallest gain 300 tickets can show. Almost all of it came from tags, with some from type. Queue accuracy fell by 5 tickets out of 300, which is within chance.
- **The pareto set flattered the winner again.** It scored 0.513 on the 50 pareto tickets against 0.421 for the start, and 0.465 on test.
- **25 iterations:** 14 children accepted, 1 rejected, 9 proposals failed, and 1 accepted child was dropped because the budget couldn't score it. The winner is five generations from the start.
- **9 of 25 reflection replies were unusable.** They came back empty or stopped inside the code block. The log doesn't record why a reply ended, so running out of tokens while thinking is likely but unconfirmed.
- **The prompt grew tenfold,** from about 1,500 to 16,000 characters, and picked up oddly specific rules. For example, it now sends "integration or data-exchange inquiries" to Returns and Exchanges. That is the same memorising seen in stage 4, despite our reflection prompt asking for general rules.

### Lessons

1. **Measure the human ceiling first.** My own labelling was the cheapest test of whether the task had signal, and it explained the result.
2. **Size the sets for the gain you want to see.** 50 tickets can only show a `gepa_score` gain of about 0.08 or more, and 300 about 0.035. See the [statistics primer](https://claude.ai/code/artifact/868379e7-471f-4ce3-a1c6-67de730bd84b).
3. **Feedback that contains the answer makes the minibatch check a rubber stamp.** If the reflector sees the true labels for the tickets the child is then tested on, acceptance only tests memorising. Next time, leave the labels out, test children on fresh tickets, or tell the reflector not to quote tickets.
4. **API failures must stop a run.** The first full run hit OpenRouter's free-model daily limit, and DSPy silently scored every refused call 0 for 77 cycles. Any API error now stops the run (docs/ticket_scoring.md section 5), and `--resume` continues from the last checkpoint.

Expanding to the full 62,000 tickets might give a cleaner generalisation signal, but the run time isn't worth it for what it would teach. The README already names playbook extraction in mini-Edra as the real test.
