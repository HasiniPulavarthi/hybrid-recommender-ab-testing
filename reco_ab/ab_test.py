"""
ab_test.py
-----------
Simulates a real A/B test of "Popularity Baseline" (control) vs
"Hybrid Recommender" (treatment).

How the simulation works
-------------------------
1. Users are randomly split 50/50 into control / treatment (a real
   randomized assignment, as in a genuine experiment).
2. Each user in their assigned arm receives a top-N recommendation list
   from that arm's model.
3. For each (user, recommended item) impression, we simulate whether the
   user *clicks*. Click probability is driven by the item's true
   relevance to the user (cosine similarity between the user's HIDDEN
   ground-truth taste vector and the item's genre vector -- the model
   itself never sees this vector, only the simulator does, exactly like
   real users' true preferences are unobserved by the recommender).
   Relevance is passed through a logistic function calibrated to a
   target baseline CTR, plus independent Bernoulli noise, so results
   look like noisy real-world click data rather than a deterministic
   score.
4. We aggregate impressions/clicks per arm into a 2x2 table and run a
   two-proportion z-test (equivalently a chi-square test of
   independence) for statistical significance, plus a lift % and a 95%
   confidence interval on the difference in CTR.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def _click_probability(relevance: np.ndarray, base_rate: float, sensitivity: float = 4.0):
    """Maps a 0..1 relevance score to a click probability via a logistic
    curve centered so that *average* relevance yields ~base_rate CTR."""
    # relevance is roughly centered around ~0.3-0.5 depending on data;
    # we standardize per-call so base_rate is respected on average.
    z = sensitivity * (relevance - relevance.mean())
    p = base_rate + (1 - base_rate) * (1 / (1 + np.exp(-z)) - 0.5) * 1.3
    return np.clip(p, 0.01, 0.95)


def simulate_ab_test(
    user_ids: np.ndarray,
    control_model,
    treatment_model,
    user_taste_matrix: np.ndarray,
    item_genre_matrix: np.ndarray,
    n_recommendations: int = 10,
    base_ctr: float = 0.08,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)
    assignment = rng.choice(["control", "treatment"], size=len(user_ids), p=[0.5, 0.5])

    item_vecs = item_genre_matrix / np.clip(item_genre_matrix.sum(axis=1, keepdims=True), 1, None)

    rows = []
    for u, arm in zip(user_ids, assignment):
        model = control_model if arm == "control" else treatment_model
        recs = model.recommend(u, n=n_recommendations, exclude_seen=True)
        if not recs:
            continue
        true_relevance = item_vecs[recs] @ user_taste_matrix[u]  # cosine-ish match to real taste
        probs = _click_probability(true_relevance, base_ctr)
        clicks = rng.binomial(1, probs)
        for item, rel, p, c in zip(recs, true_relevance, probs, clicks):
            rows.append((u, arm, item, rel, p, c))

    impressions_df = pd.DataFrame(
        rows, columns=["user_id", "arm", "item_id", "true_relevance", "click_prob", "clicked"]
    )

    summary = (
        impressions_df.groupby("arm")
        .agg(impressions=("clicked", "size"), clicks=("clicked", "sum"))
        .reindex(["control", "treatment"])
    )
    summary["ctr"] = summary["clicks"] / summary["impressions"]

    stats_result = two_proportion_z_test(
        clicks_a=int(summary.loc["control", "clicks"]),
        n_a=int(summary.loc["control", "impressions"]),
        clicks_b=int(summary.loc["treatment", "clicks"]),
        n_b=int(summary.loc["treatment", "impressions"]),
    )

    # secondary metric: "engagement" (e.g. a like/save), modeled as a
    # click-conditional secondary action so it's naturally noisier and
    # correlated but not identical to CTR
    impressions_df["engaged"] = np.where(
        impressions_df["clicked"] == 1,
        rng.binomial(1, 0.35, size=len(impressions_df)),
        0,
    )
    eng_summary = (
        impressions_df.groupby("arm")
        .agg(impressions=("engaged", "size"), engagements=("engaged", "sum"))
        .reindex(["control", "treatment"])
    )
    eng_summary["engagement_rate"] = eng_summary["engagements"] / eng_summary["impressions"]
    eng_stats = two_proportion_z_test(
        clicks_a=int(eng_summary.loc["control", "engagements"]),
        n_a=int(eng_summary.loc["control", "impressions"]),
        clicks_b=int(eng_summary.loc["treatment", "engagements"]),
        n_b=int(eng_summary.loc["treatment", "impressions"]),
    )

    return {
        "impressions_df": impressions_df,
        "ctr_summary": summary,
        "ctr_test": stats_result,
        "engagement_summary": eng_summary,
        "engagement_test": eng_stats,
    }


def two_proportion_z_test(clicks_a: int, n_a: int, clicks_b: int, n_b: int, alpha: float = 0.05):
    """Two-proportion z-test: control (a, 'baseline') vs treatment
    (b, 'hybrid'). Returns rate, lift, z, p-value, CI, and a
    significance verdict at the given alpha. Cross-checked against a
    chi-square test of independence (they're mathematically equivalent
    for a 2x2 table with a two-sided test)."""
    p_a = clicks_a / n_a
    p_b = clicks_b / n_b
    p_pool = (clicks_a + clicks_b) / (n_a + n_b)

    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    diff = p_b - p_a
    ci_low = diff - 1.96 * se_unpooled
    ci_high = diff + 1.96 * se_unpooled

    lift_pct = (diff / p_a * 100) if p_a > 0 else float("nan")

    # chi-square cross-check
    table = np.array([[clicks_a, n_a - clicks_a], [clicks_b, n_b - clicks_b]])
    chi2, chi2_p, _, _ = stats.chi2_contingency(table, correction=False)

    return {
        "rate_control": p_a,
        "rate_treatment": p_b,
        "abs_diff": diff,
        "lift_pct": lift_pct,
        "z_stat": z,
        "p_value": p_value,
        "chi2_stat": chi2,
        "chi2_p_value": chi2_p,
        "ci_95": (ci_low, ci_high),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
    }
