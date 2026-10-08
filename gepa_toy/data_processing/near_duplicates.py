import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

CHUNK = 1000


def tfidf_matrix(texts: list[str]) -> csr_matrix:
    # Low IDF weights stop shared greetings like "Dear Customer Support" from looking like duplicates.
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return tfidf.fit_transform(texts).astype(np.float32)


def similar_pairs(matrix: csr_matrix, min_sim: float) -> tuple[np.ndarray, ...]:
    """Return (left, right, similarity) for every pair with left < right at or above min_sim."""
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


def closest_match(matrix: csr_matrix, ids: list[int], others: list[int]) -> np.ndarray:
    """Highest similarity between each row in ids and any row in others."""
    return (matrix[ids] @ matrix[others].T).toarray().max(axis=1)
