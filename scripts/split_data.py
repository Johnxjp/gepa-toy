"""Sample balanced, leak-free train/pareto/test splits plus a hand-labelling set."""

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from near_duplicates import (
    DATA,
    cluster_labels,
    load_texts,
    similar_pairs,
    tfidf_matrix,
)

OUT_DIR = Path("data/customer_support_tickets/splits")
# Chosen in docs/dataset_preparation.md; lower values chain clusters together.
DUPLICATE_CUTOFF = 0.45
SEED = 0
PER_QUEUE = {"train": 10, "pareto": 5, "test": 30, "label": 5}


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with DATA.open(newline="", encoding="utf-8") as f:
        rows = [{"id": i, **r} for i, r in enumerate(csv.DictReader(f, delimiter="\t"))]
    matrix = tfidf_matrix(load_texts())
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

    OUT_DIR.mkdir(exist_ok=True)
    fields = ["id"] + [k for k in rows[0] if k != "id"]
    for split in ["train", "pareto", "test"]:
        write_tsv(OUT_DIR / f"{split}.tsv", splits[split], fields)
    # The labelling sheet hides the real queue, and is shuffled because rows arrive grouped by queue.
    rng.shuffle(splits["label"])
    write_tsv(OUT_DIR / "label_key.tsv", splits["label"], fields)
    write_tsv(
        OUT_DIR / "label_sheet.tsv",
        splits["label"],
        ["id", "subject", "body", "my_queue"],
    )

    sampled = [r["id"] for split in splits.values() for r in split]
    assert len(sampled) == len(set(sampled)), "a ticket landed in two splits"
    test_ids = [r["id"] for r in splits["test"]]
    other_ids = [i for i in sampled if i not in set(test_ids)]
    closest = (matrix[test_ids] @ matrix[other_ids].T).toarray().max(axis=1)
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
