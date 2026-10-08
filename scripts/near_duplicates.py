"""Report near-duplicate tickets by TF-IDF cosine similarity at several thresholds."""

import csv
import random
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path("data/customer_support_tickets/english.tsv")
THRESHOLDS = [0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
CHUNK = 1000
SAMPLES_PER_BAND = 3


def load_texts() -> list[str]:
    with DATA.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [f"{r['subject']} {r['body']}".replace("\\n", " ") for r in rows]


def tfidf_matrix(texts: list[str]) -> csr_matrix:
    # Low IDF weights stop shared greetings like "Dear Customer Support" from looking like duplicates.
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return tfidf.fit_transform(texts).astype(np.float32)


def similar_pairs(matrix: csr_matrix, min_sim: float) -> tuple[np.ndarray, ...]:
    left, right, sims = [], [], []
    # Chunked so the full 16k x 16k similarity matrix never sits in memory at once.
    for start in range(0, matrix.shape[0], CHUNK):
        block = (matrix[start : start + CHUNK] @ matrix.T).toarray()
        rows, cols = np.nonzero(block >= min_sim)
        rows += start
        keep = cols > rows
        left.append(rows[keep])
        right.append(cols[keep])
        sims.append(block[rows[keep] - start, cols[keep]])
    return np.concatenate(left), np.concatenate(right), np.concatenate(sims)


def cluster_labels(left: np.ndarray, right: np.ndarray, n: int) -> np.ndarray:
    graph = coo_matrix((np.ones(len(left)), (left, right)), (n, n))
    return connected_components(graph, directed=False)[1]


def main() -> None:
    texts = load_texts()
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
