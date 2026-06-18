"""
Non-learned VPR baselines: Sum-of-Absolute-Differences (SAD) and PCA matching.

Both operate directly on the flattened 2-channel event frames and produce a
query-vs-reference similarity matrix comparable with the SpikeVPR descriptor
similarity, for the recall/precision metrics in ``metrics.py``.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def sad_similarity(queries, references):
    """Sum-of-absolute-differences turned into a [0, 1] similarity matrix [Q, R]."""
    q = np.asarray(queries).reshape(len(queries), -1)
    r = np.asarray(references).reshape(len(references), -1)
    sad = np.stack([np.sum(np.abs(r - q[i]), axis=1) for i in range(len(q))])
    return 1 - (sad / sad.max())


def pca_similarity(queries, references, n_components=60):
    """Cosine similarity in a PCA subspace fitted on the references."""
    q = np.asarray(queries).reshape(len(queries), -1)
    r = np.asarray(references).reshape(len(references), -1)
    pca = PCA(n_components=n_components).fit(r)
    return cosine_similarity(pca.transform(q), pca.transform(r))
