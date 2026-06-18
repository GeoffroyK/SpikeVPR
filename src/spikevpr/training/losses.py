"""
Contrastive training objective.

SpikeVPR is trained with a single objective across all three datasets: InfoNCE
(NT-Xent). A batch of (anchor, positive) pairs is embedded together and labelled
by place id; NT-Xent pulls same-place embeddings together and pushes every other
sample in the batch apart. No miner or explicit negative sampling is needed —
all other in-batch samples act as negatives.
"""
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity


def build_infonce_loss(temperature=0.07):
    """InfoNCE / NT-Xent loss on cosine similarity."""
    return losses.NTXentLoss(temperature=temperature, distance=CosineSimilarity())
