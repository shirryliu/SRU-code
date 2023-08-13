from collections import defaultdict

import numpy as np
import torch


def recall(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / labels.sum(1).float()).mean().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(n, k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = defaultdict(float)
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)

    for i in range(labels.size(0)):
        for k in sorted(ks, reverse=True):
            topk = rank[i]
            label_rank = torch.nonzero(topk == labels[i].item()).cpu().item()
            if label_rank < k:
                metrics['Recall@%d' % k] += 1.0
                metrics['NDCG@%d' % k] += 1.0 / np.log2(label_rank + 2)
    for k in sorted(ks, reverse=True):
        metrics['Recall@%d' % k] /= labels.size(0)
        metrics['NDCG@%d' % k] /= labels.size(0)

    return metrics
