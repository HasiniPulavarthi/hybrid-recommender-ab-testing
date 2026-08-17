"""
recommender.py
----------------
Four recommenders, increasing in sophistication:

  PopularityRecommender   - non-personalized baseline ("most watched")
  CollaborativeRecommender- matrix factorization (truncated SVD) on the
                             user-item rating matrix
  ContentBasedRecommender - cosine similarity between a user's genre
                             profile and item genre vectors
  HybridRecommender        - weighted blend of CF + content-based scores

All models share a common interface:
    .fit(ratings_df)
    .recommend(user_id, n=10, exclude_seen=True) -> list[item_id]
    .score_all(user_id) -> np.ndarray of scores for every item (for eval)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class BaseRecommender:
    name = "base"

    def __init__(self, n_users: int, n_items: int):
        self.n_users = n_users
        self.n_items = n_items
        self._seen = {}  # user_id -> set(item_ids) rated in training data

    def _build_seen(self, ratings_df: pd.DataFrame):
        self._seen = ratings_df.groupby("user_id")["item_id"].apply(set).to_dict()

    def score_all(self, user_id: int) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True):
        scores = self.score_all(user_id).copy()
        if exclude_seen:
            for it in self._seen.get(user_id, ()):
                scores[it] = -np.inf
        top = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
        top = top[np.argsort(-scores[top])]
        return list(top)


class PopularityRecommender(BaseRecommender):
    """Non-personalized baseline: same ranked list (by popularity score)
    for every user. This is the classic 'control' arm in an A/B test —
    simple, cheap, and what most systems ship before investing in ML."""

    name = "Popularity Baseline"

    def fit(self, ratings_df: pd.DataFrame):
        self._build_seen(ratings_df)
        counts = ratings_df.groupby("item_id")["rating"].agg(["count", "mean"]).reindex(
            range(self.n_items), fill_value=0
        )
        # score = volume * quality, log-damped so a handful of mega-hits
        # don't totally dominate
        self._scores = np.log1p(counts["count"].values) * (counts["mean"].values + 1)
        return self

    def score_all(self, user_id: int) -> np.ndarray:
        return self._scores


class CollaborativeRecommender(BaseRecommender):
    """Matrix-factorization CF via truncated SVD on the mean-centered
    user-item rating matrix (a lightweight stand-in for Surprise's SVD /
    LightFM's WARP — same idea: learn latent taste/item factors from the
    interaction matrix alone, no content features)."""

    name = "Collaborative Filtering (SVD)"

    def __init__(self, n_users, n_items, n_factors: int = 24, seed: int = 0):
        super().__init__(n_users, n_items)
        self.n_factors = n_factors
        self.seed = seed

    def fit(self, ratings_df: pd.DataFrame):
        self._build_seen(ratings_df)
        mat = np.zeros((self.n_users, self.n_items))
        mask = np.zeros_like(mat, dtype=bool)
        for u, i, r in ratings_df[["user_id", "item_id", "rating"]].itertuples(index=False):
            mat[u, i] = r
            mask[u, i] = True

        self.user_mean = np.divide(
            mat.sum(axis=1), mask.sum(axis=1), out=np.zeros(self.n_users), where=mask.sum(axis=1) > 0
        )
        centered = mat - self.user_mean[:, None]
        centered[~mask] = 0.0  # unseen entries treated as 0 after centering

        k = min(self.n_factors, min(mat.shape) - 1)
        svd = TruncatedSVD(n_components=k, random_state=self.seed)
        self.U = svd.fit_transform(centered)          # n_users x k
        self.Vt = svd.components_                      # k x n_items
        return self

    def score_all(self, user_id: int) -> np.ndarray:
        return self.user_mean[user_id] + self.U[user_id] @ self.Vt


class ContentBasedRecommender(BaseRecommender):
    """Builds a per-user genre-preference profile from the items they
    rated highly, then scores every item by cosine similarity to that
    profile. Works even for users with very few ratings (less prone to
    cold-start than pure CF) and can recommend niche/unpopular items
    that CF would never surface."""

    name = "Content-Based (genre similarity)"

    def __init__(self, n_users, n_items, item_genre_matrix: np.ndarray):
        super().__init__(n_users, n_items)
        self.item_vectors = normalize(item_genre_matrix.astype(float))

    def fit(self, ratings_df: pd.DataFrame):
        self._build_seen(ratings_df)
        self.user_profiles = np.zeros((self.n_users, self.item_vectors.shape[1]))
        for u, grp in ratings_df.groupby("user_id"):
            w = (grp["rating"].values - 2.5)  # like/dislike weighting, centered
            vecs = self.item_vectors[grp["item_id"].values]
            profile = (w[:, None] * vecs).sum(axis=0)
            self.user_profiles[u] = profile
        self.user_profiles = normalize(self.user_profiles)
        return self

    def score_all(self, user_id: int) -> np.ndarray:
        return self.item_vectors @ self.user_profiles[user_id]


class HybridRecommender(BaseRecommender):
    """Weighted blend of collaborative + content-based scores (each
    min-max normalized per user before blending so neither channel
    dominates purely due to scale). alpha controls the CF/content mix.

    This is the 'treatment' arm of the A/B test."""

    name = "Hybrid (CF + Content)"

    def __init__(self, cf: CollaborativeRecommender, cb: ContentBasedRecommender, alpha: float = 0.6):
        super().__init__(cf.n_users, cf.n_items)
        self.cf = cf
        self.cb = cb
        self.alpha = alpha

    def fit(self, ratings_df: pd.DataFrame):
        self._build_seen(ratings_df)  # cf/cb already fit externally
        return self

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    def score_all(self, user_id: int) -> np.ndarray:
        cf_s = self._norm(self.cf.score_all(user_id))
        cb_s = self._norm(self.cb.score_all(user_id))
        return self.alpha * cf_s + (1 - self.alpha) * cb_s


def build_all_models(ratings_df, item_genre_matrix, n_users, n_items, alpha: float = 0.6):
    """Convenience factory: fits all four models and returns them in a dict."""
    pop = PopularityRecommender(n_users, n_items).fit(ratings_df)
    cf = CollaborativeRecommender(n_users, n_items).fit(ratings_df)
    cb = ContentBasedRecommender(n_users, n_items, item_genre_matrix).fit(ratings_df)
    hybrid = HybridRecommender(cf, cb, alpha=alpha).fit(ratings_df)
    return {
        "popularity": pop,
        "collaborative": cf,
        "content": cb,
        "hybrid": hybrid,
    }
