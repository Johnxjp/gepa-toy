# Ticket Observations
From labelling 50 tickets I noticed that many follow the same format suggesting a template. I also noticed that there is a lot of overlap between labels and the tags were often not very clear. It seems like for 'type' and queue there's a missing label which is 'notification'. For 'type', I also hardly used change and didn't really know what it was for. I also couldn't judge clearly the difference between problem and incident.

Finally, there are some tickets that tick multiple boxes but there's no option for that.

## Outcome: experiment stopped after stage 4 (2026-10-11)

We stopped after running DSPy's GEPA (stage 4) once. On this dataset no stage comparison would be measurable, so stages 1 to 3 were not run.

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

### Lessons

1. **Measure the human ceiling first.** My own labelling was the cheapest test of whether the task had signal, and it explained the result.
2. **Size the sets for the gain you want to see.** 50 tickets can only show a `gepa_score` gain of about 0.08 or more, and 300 about 0.035. See the [statistics primer](https://claude.ai/code/artifact/868379e7-471f-4ce3-a1c6-67de730bd84b).
3. **Feedback that contains the answer makes the minibatch check a rubber stamp.** If the reflector sees the true labels for the tickets the child is then tested on, acceptance only tests memorising. Next time, leave the labels out, test children on fresh tickets, or tell the reflector not to quote tickets.
4. **API failures must stop a run.** The first full run hit OpenRouter's free-model daily limit, and DSPy silently scored every refused call 0 for 77 cycles. Any API error now stops the run (docs/ticket_scoring.md section 5), and `--resume` continues from the last checkpoint.

Expanding to the full 62,000 tickets might give a cleaner generalisation signal, but the run time isn't worth it for what it would teach. The README already names playbook extraction in mini-Edra as the real test.