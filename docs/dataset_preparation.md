# Dataset preparation

How we turn the raw ticket CSV into the data GEPA runs on, and why each step works the way it does.

## Files

| File | What it holds |
| --- | --- |
| `data/customer_support_tickets/data.csv` | Raw download, English and German |
| `data/customer_support_tickets/english.tsv` | English tickets with cleaned `tags` and `tag_groups` columns (generated) |
| `data/customer_support_tickets/tag_spelling_map.tsv` | Every tag spelling that was merged, for tracing (generated) |
| `data/customer_support_tickets/tag_groups.tsv` | Hand-reviewed mapping from the top 200 tags to 20 groups (edit by hand) |
| `data/customer_support_tickets/labels.json` | Final label values for the annotation UI: queues, types, priorities, and the 20 tag groups as `tags` (generated) |
| `data/customer_support_tickets/splits/{train,pareto,test}.tsv` | The GEPA splits, with an `id` column pointing back to the row in `english.tsv` (generated) |
| `data/customer_support_tickets/splits/label_sheet.tsv` | 50 tickets to hand-label, queue hidden, fill in `my_queue` (generated) |
| `data/customer_support_tickets/splits/label_key.tsv` | The same 50 tickets with their real labels (generated) |

Rerun with `uv run` (paths come from `gepa_toy/config.py`, so any working directory works):

```sh
uv run scripts/prepare_data.py       # writes english.tsv, tag_spelling_map.tsv and labels.json, prints label breakdowns
uv run scripts/near_duplicates.py    # prints near-duplicate counts and sample pairs per similarity band
uv run scripts/split_data.py         # writes the splits and checks no test ticket has a near duplicate elsewhere
```

## Raw data at a glance

28,587 tickets, 16 columns: `subject`, `body`, `answer`, `type`, `queue`, `priority`, `language`, `version`, `tag_1` to `tag_8`. No blank rows. 3,838 tickets have no subject and 7 have no answer. `body` is always present.

## 1. Keep English only

**Decision:** drop German tickets.

**Why:** the task is English queue routing (see README). Mixing languages would make prompt changes harder to read and judge.

**Result:** 16,338 tickets (57% of the raw data). The `language` column is dropped from the output because every row is English.

## 2. Merge the eight tag columns into one list

**Decision:** combine `tag_1` to `tag_8` into a single comma-separated `tags` column.

**Why:** the eight columns have no meaningful order and most are empty past `tag_4`. Some cells also hold several tags already joined by commas (for example `Security,IT,Tech Support`), so we split those before merging and remove repeats.

## 3. Fix tag spelling variants

**Decision:** merge tags that differ only by case, hyphens or a trailing plural "s", keeping the most common spelling.

**Why:** pairs like `Data Breach` / `Data breach` and `Protocol` / `Protocols` split the same idea across two labels.

**Approach:** a plural is merged only when the singular also exists as a tag, so words like `Analytics` and `Sales` stay intact. All 36 merges are written to `tag_spelling_map.tsv` so any change can be traced back. This reduced 881 distinct tags to 845.

## 4. Group tags into 20 groups

**Decision:** map the 200 most common tags to 20 groups, and drop the rest from the groups.

**Why:** 845 tags is far too many to list in a prompt, and most are noise: 394 tags appear on only one ticket, and several mean nearly the same thing ("Support", "Tech Support", "Customer Support"). The top 200 tags cover about 98% of all tag uses.

**Approach:** Claude proposed the groups and the mapping, then we reviewed and edited `tag_groups.tsv` by hand. The raw `tags` column is kept next to `tag_groups` so any grouping can be checked. Every ticket still has at least one group.

**Known weakness:** `General Support` is on 72% of tickets because it collects vague tags such as "Tech Support", "IT" and "Issue". We are keeping it for now and may drop it later.

**Considered:**

- **Keep the top N tags as they are.** Simple, and few tickets lose all their tags (56 at top 20, 3 at top 50). Rejected because the kept tags still overlap heavily in meaning.
- **Drop generic tags** (on more than about 45% of tickets, or repeating `queue` or `type`). Not applied yet. This is the option to revisit for `General Support`.
- **Cluster tags by embedding similarity.** No hand work, but the clusters come out messy and still need naming. Too much effort for 845 short labels.

## 5. Find near-duplicate tickets

**Decision:** treat two tickets as near duplicates when their TF-IDF cosine similarity is at least **0.45**. When we split the data, each cluster of near duplicates must go entirely into one split.

**Why:** the dataset is synthetic, and most tickets are rewordings of another ticket. For example: "A significant data inconsistency has been observed within the investment analytics system" against "A significant data discrepancy has been observed in the investment analytics system". If one copy lands in train and the other in test, the test score overstates how well the prompt works.

**Approach:** `scripts/near_duplicates.py` turns subject plus body into TF-IDF vectors over words and word pairs, then compares every pair of tickets. TF-IDF gives low weight to words found in almost every ticket, so shared boilerplate like "Dear Customer Support Team, I hope this message reaches you well" barely counts. Pairs above the cut-off are linked, and linked tickets form clusters.

| Cut-off | Tickets with a near duplicate | Biggest cluster |
| --- | --- | --- |
| 0.3 | 87% | 5,129 |
| 0.4 | 79% | 711 |
| **0.45** | **76%** | **110** |
| 0.5 | 72% | 57 |
| 0.6 | 60% | 14 |
| 0.7 | 41% | 5 |
| 0.8 | 18% | 5 |
| 0.9 | 3% | 5 |

**Why 0.45:** we read sample pairs in each band. From 0.4 upwards nearly every pair is the same ticket reworded. Between 0.3 and 0.4 many pairs are only on the same topic, and clusters chain together (A is like B, B is like C) until a third of the data forms one cluster, which makes a balanced split impossible. At 0.45 the clusters still catch the rewordings and the biggest one is 110 tickets, small enough to keep splits balanced. 0.4 is stricter, but its 711-ticket cluster would make split sizes lumpy.

**Considered:**

- **Exact or normalised text match.** Finds only 19 duplicates (0.1%) and misses all rewordings.
- **Matching the first 100 characters.** Flags 827 tickets, but many only share a stock greeting and then differ completely.
- **MinHash word overlap.** Similar results to TF-IDF, but needs another library and does not discount boilerplate unless greetings are stripped first.
- **Embedding similarity.** Catches paraphrases with different wording, but needs a model and more setup. Worth adding if TF-IDF turns out to miss leaks.

## 6. Split the data

**Decision:** sample 500 tickets, 50 from each of the 10 queues, and split them per queue into train (10), pareto (5), test (30) and a hand-labelling set (5). Take at most one ticket from each near-duplicate cluster. Seed 0.

**Why these sizes:** there are 16,338 tickets, and 8,390 distinct ones after collapsing rewordings, so data is not the constraint. Cost and measurement noise are.

- **Test (300)** is the number that matters. At 100 tickets a 70% accuracy has a 95% margin of about ±9 points, too wide to tell most prompts apart. At 300 it is about ±5. Test is run only a few times, so the extra calls are cheap.
- **Pareto (50)** is scored for every accepted candidate, so it drives the cost of a GEPA run. A bigger set picks winners more reliably but leaves budget for fewer candidates.
- **Train (100)** only feeds minibatches of 3 to 5, so 100 is plenty.
- **Label (50)** is for reading and hand-labelling blind, to estimate how much of the task is achievable. Near-duplicate pairs disagree on queue 6% of the time, mostly Customer Service, Product Support and IT Support against Technical Support, so expect a ceiling below 100%.

**Why balanced:** stratifying in proportion to the natural mix would give General Inquiry (1.4% of tickets) one train ticket, one pareto ticket and four test tickets. GEPA would rarely see it in feedback, and one test mistake would move its accuracy by 25 points. Equal counts per queue let every queue shape the prompt and be measured. Accuracy on the natural mix can be recovered later by reweighting per-queue test accuracy by the real queue shares, at no extra cost.

**Why one ticket per cluster:** it rules out leakage between splits by construction, and it stops a split from spending slots on two rewordings of the same ticket. Every queue has far more distinct tickets than the 50 it needs (General Inquiry, the smallest, has 116).

**Check:** after splitting, the script compares each test ticket with every other sampled ticket. The closest match is 0.43, below the 0.45 cut-off. Reruns produce identical files.

**Considered:**

- **Proportional stratification.** Rejected for the reason above.
- **Assigning whole clusters to splits and keeping all copies.** Also leak-free, but it wastes samples on rewordings and lets one paraphrased ticket count several times in test.
- **Keeping the README's original 100-ticket test set.** Rejected as too noisy.

## Next

- Hand-label `label_sheet.tsv` and compare with `label_key.tsv` to set an expected ceiling.
- Reweight per-queue test accuracy by natural queue shares when reporting.
