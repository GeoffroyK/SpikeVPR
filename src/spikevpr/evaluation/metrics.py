"""
Retrieval metrics for VPR.

The core metric is recall@N with a geographic tolerance: a query is correct if
any of its top-N retrieved references lies within ``threshold`` metres of the
query's true position. ``coordinate_distance`` makes this work for Brisbane
(lat/lon, geodesic) and NSAVP/NYC (planar metres) without branching at the call site.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve

from ..data.gps import coordinate_distance


# ── embedding extraction (pair-style loaders: Brisbane, NSAVP) ───────────────────

@torch.no_grad()
def extract_pair_embeddings(model, loader, device):
    """
    Run the model over a pair loader, treating anchors as queries and positives
    as references. Returns (q_emb, q_gps, r_emb, r_gps).
    """
    model.eval()
    q_emb, q_gps, r_emb, r_gps = [], [], [], []
    for batch in loader:
        q_emb.append(model(batch["anchor"].to(device).float()).cpu())
        r_emb.append(model(batch["positive"].to(device).float()).cpu())
        q_gps.append(batch["gps_anchor"].cpu())
        r_gps.append(batch["gps_positive"].cpu())
    return (torch.cat(q_emb), torch.cat(q_gps).numpy(),
            torch.cat(r_emb), torch.cat(r_gps).numpy())


# ── similarity / recall ──────────────────────────────────────────────────────────

def similarity_matrix(query_embeddings, reference_embeddings):
    """Cosine-similarity matrix [Q, R] between L2-normalised embeddings."""
    q = F.normalize(query_embeddings, p=2, dim=1)
    r = F.normalize(reference_embeddings, p=2, dim=1)
    return (q @ r.T).cpu().numpy()


def recall_at_n(sim_matrix, query_gps, reference_gps, threshold=30,
                n_values=(1, 5, 10, 15, 20, 25)):
    """
    Recall@N: fraction of queries whose top-N references contain at least one
    within ``threshold`` metres of the query position.
    """
    recalls = {}
    for n in n_values:
        correct = 0
        for i in range(sim_matrix.shape[0]):
            for j in np.argsort(-sim_matrix[i])[:n]:
                if coordinate_distance(query_gps[i], reference_gps[j]) < threshold:
                    correct += 1
                    break
        recalls[f"recall_{n}"] = correct / sim_matrix.shape[0]
    return recalls


def precision_recall(sim_matrix, query_gps, reference_gps, threshold=30):
    """Precision-recall curve points and average precision over all query/ref pairs."""
    scores, labels = [], []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            scores.append(sim_matrix[i, j])
            labels.append(int(coordinate_distance(query_gps[i], reference_gps[j]) < threshold))
    precision, recall, _ = precision_recall_curve(labels, scores)
    return precision, recall, average_precision_score(labels, scores)


# ── NYC cross-session recall ─────────────────────────────────────────────────────

def recall_at_n_nyc(query_embeddings, query_gps, ref_embeddings, ref_gps,
                    n_values=(1, 5, 10), threshold_m=25.0,
                    query_sessions=None, ref_sessions=None):
    """
    NYC recall@N. When session labels are given, also reports ``strict_recall_N``
    with same-session database frames masked out — the honest cross-traverse
    metric, since random frame-level splits otherwise leak same-session matches.
    """
    q = query_embeddings / (query_embeddings.norm(dim=1, keepdim=True) + 1e-8)
    r = ref_embeddings / (ref_embeddings.norm(dim=1, keepdim=True) + 1e-8)
    sim = q @ r.T
    cross_session = query_sessions is not None and ref_sessions is not None

    results = {}
    for n in n_values:
        correct_std = correct_strict = n_eligible = 0
        for qi in range(len(query_gps)):
            q_pos = query_gps[qi]
            top_n = sim[qi].topk(n).indices
            if ((ref_gps[top_n] - q_pos).norm(dim=1) <= threshold_m).any():
                correct_std += 1
            if cross_session:
                mask = torch.tensor([s != query_sessions[qi] for s in ref_sessions])
                if mask.any():
                    n_eligible += 1
                    sim_masked = sim[qi].clone()
                    sim_masked[~mask] = -2.0
                    top_n_s = sim_masked.topk(n).indices
                    if ((ref_gps[top_n_s] - q_pos).norm(dim=1) <= threshold_m).any():
                        correct_strict += 1
        results[f"recall_{n}"] = correct_std / len(query_gps)
        if cross_session and n_eligible > 0:
            results[f"strict_recall_{n}"] = correct_strict / n_eligible
    return results
