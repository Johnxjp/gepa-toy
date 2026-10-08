"""Sample balanced, leak-free train/pareto/test splits plus a hand-labelling set."""

import random
from collections import defaultdict

import numpy as np

from gepa_toy.config import DUPLICATE_CUTOFF, ENGLISH_TSV, SPLITS_DIR
from gepa_toy.data_processing.near_duplicates import (
    closest_match,
    cluster_labels,
    similar_pairs,
    tfidf_matrix,
)
from gepa_toy.data_processing.tickets import ticket_text
from gepa_toy.data_processing.tsv import read_tsv, write_tsv

SEED = 0
PER_QUEUE = {"train": 10, "pareto": 5, "test": 30, "label": 5}


def main() -> None:
    rows = [{"id": i, **r} for i, r in enumerate(read_tsv(ENGLISH_TSV))]
    matrix = tfidf_matrix([ticket_text(r) for r in rows])
    left, right, _ = similar_pairs(matrix, DUPLICATE_CUTOFF)
    clusters = cluster_labels(left, right, len(rows))

    # One ticket per near-duplicate cluster, so no two sampled tickets are rewordings of each other.
    rng = random.Random(SEED)
    by_cluster = defaultdict(list)
    for row in rows:
        by_cluster[clusters[row["id"]]].append(row)
    by_queue = defaultdict(list)
    for members in by_cluster.values():
        pick = rng.choice(members)
        by_queue[pick["queue"]].append(pick)

    splits = defaultdict(list)
    for queue in sorted(by_queue):
        candidates = by_queue[queue]
        rng.shuffle(candidates)
        start = 0
        for split, count in PER_QUEUE.items():
            splits[split] += candidates[start : start + count]
            start += count

    SPLITS_DIR.mkdir(exist_ok=True)
    fields = list(rows[0])
    for split in ["train", "pareto", "test"]:
        write_tsv(SPLITS_DIR / f"{split}.tsv", splits[split], fields)
    # The labelling sheet hides the real queue, and is shuffled because rows arrive grouped by queue.
    rng.shuffle(splits["label"])
    write_tsv(SPLITS_DIR / "label_key.tsv", splits["label"], fields)
    write_tsv(
        SPLITS_DIR / "label_sheet.tsv",
        splits["label"],
        ["id", "subject", "body", "my_queue"],
    )

    sampled = [r["id"] for split in splits.values() for r in split]
    assert len(sampled) == len(set(sampled)), "a ticket landed in two splits"
    test_ids = [r["id"] for r in splits["test"]]
    other_ids = [i for i in sampled if i not in set(test_ids)]
    closest = closest_match(matrix, test_ids, other_ids)
    for split, members in splits.items():
        print(f"{split}: {len(members)} tickets")
    print(
        f"Closest non-test match for any test ticket: {closest.max():.2f} (cut-off {DUPLICATE_CUTOFF})"
    )
    print(
        f"Test tickets with a match at or above the cut-off: {int(np.sum(closest >= DUPLICATE_CUTOFF))}"
    )


if __name__ == "__main__":
    main()
