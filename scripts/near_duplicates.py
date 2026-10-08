"""Report near-duplicate tickets by TF-IDF cosine similarity at several thresholds."""

import random

import numpy as np

from gepa_toy.config import ENGLISH_TSV
from gepa_toy.data_processing.near_duplicates import (
    cluster_labels,
    similar_pairs,
    tfidf_matrix,
)
from gepa_toy.data_processing.tickets import ticket_text
from gepa_toy.data_processing.tsv import read_tsv

THRESHOLDS = [0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
SAMPLES_PER_BAND = 3


def main() -> None:
    texts = [ticket_text(r) for r in read_tsv(ENGLISH_TSV)]
    n = len(texts)
    left, right, sims = similar_pairs(tfidf_matrix(texts), min(THRESHOLDS))

    print(f"{n} tickets\n")
    print("threshold\tpairs\ttickets in a cluster\tclusters\tbiggest cluster")
    for threshold in THRESHOLDS:
        mask = sims >= threshold
        sizes = np.bincount(cluster_labels(left[mask], right[mask], n))
        in_cluster = sizes[sizes > 1]
        print(
            f"{threshold}\t{mask.sum()}\t{in_cluster.sum()} ({in_cluster.sum() / n:.1%})"
            f"\t{len(in_cluster)}\t{sizes.max()}"
        )

    random.seed(0)
    bands = list(zip(THRESHOLDS, THRESHOLDS[1:] + [1.01]))
    for low, high in bands:
        idx = np.nonzero((sims >= low) & (sims < high))[0]
        print(f"\n## similarity {low}-{min(high, 1.0)} ({len(idx)} pairs)")
        for i in random.sample(list(idx), min(SAMPLES_PER_BAND, len(idx))):
            print(f"\n[{sims[i]:.2f}]")
            print(f"  A: {texts[left[i]][:300]}")
            print(f"  B: {texts[right[i]][:300]}")


if __name__ == "__main__":
    main()
