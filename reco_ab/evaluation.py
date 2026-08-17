"""
evaluation.py
--------------
Standard offline top-N ranking metrics, computed on a held-out test set
of ratings: Precision@K, Recall@K, NDCG@K. A "relevant" item is one the
held-out test set shows the user rated >= 4.

This is the classic ML-evaluation half of the project (how good is the
ranking), complementary to the A/B test (how much did it move a business
metric under simulated user behavior).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def train_test_split_ratings(ratings_df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    ratings_df = ratings_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_idx = set()
    for u, grp in ratings_df.groupby("user_id"):
        n_test = max(1, int(len(grp) * test_frac)) if len(grp) >= 5 else 0
        if n_test:
            test_idx.update(rng.choice(grp.index.values, size=n_test, replace=False))
    test_df = ratings_df.loc[list(test_idx)]
    train_df = ratings_df.drop(index=test_idx)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _ndcg_at_k(ranked_items, relevant_set, k):
    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in relevant_set:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(model, test_df: pd.DataFrame, k: int = 10, rel_threshold: int = 4):
    """Returns dict(precision, recall, ndcg) averaged over test users
    that have at least one relevant held-out item."""
    relevant_by_user = (
        test_df[test_df["rating"] >= rel_threshold]
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    precisions, recalls, ndcgs = [], [], []
    for u, relevant in relevant_by_user.items():
        if not relevant:
            continue
        ranked = model.recommend(u, n=k, exclude_seen=True)
        hits = len(set(ranked) & relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))
        ndcgs.append(_ndcg_at_k(ranked, relevant, k))

    return {
        "precision@k": float(np.mean(precisions)) if precisions else 0.0,
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_eval_users": len(precisions),
    }
