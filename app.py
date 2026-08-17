"""
app.py
-------
Interactive Streamlit dashboard for the Hybrid Recommendation Engine +
A/B Testing project. Lets you tweak dataset size, hybrid blend weight,
and A/B test parameters, and see offline model-quality metrics plus
simulated A/B test results (with significance testing) update live.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reco_ab.data_generator import generate_dataset
from reco_ab.recommender import build_all_models
from reco_ab.evaluation import train_test_split_ratings, evaluate_model
from reco_ab.ab_test import simulate_ab_test

st.set_page_config(page_title="Hybrid Recommender + A/B Test", layout="wide")

st.title("🎬 Hybrid Recommendation Engine + A/B Testing Simulator")
st.caption(
    "Collaborative filtering + content-based hybrid recommender, evaluated offline "
    "(Precision/Recall/NDCG) and validated via a simulated randomized A/B test against "
    "a popularity baseline, with two-proportion z-test significance."
)

with st.sidebar:
    st.header("Dataset")
    n_users = st.slider("Number of users", 200, 5000, 3000, step=200)
    n_items = st.slider("Number of items", 100, 1000, 500, step=50)
    seed = st.number_input("Random seed", value=42, step=1)

    st.header("Hybrid model")
    alpha = st.slider(
        "Blend weight α (CF weight; content weight = 1-α)", 0.0, 1.0, 0.6, step=0.05
    )
    k_eval = st.slider("Top-K recommendations", 5, 20, 10)

    st.header("A/B test")
    base_ctr = st.slider("Target baseline CTR", 0.02, 0.25, 0.08, step=0.01)
    run_btn = st.button("🚀 Run pipeline", type="primary", use_container_width=True)

if "results" not in st.session_state:
    run_btn = True  # auto-run on first load

if run_btn:
    with st.spinner("Generating data, fitting models, running simulation..."):
        users_df, items_df, ratings_df, item_genre_matrix, user_taste_matrix = generate_dataset(
            n_users=n_users, n_items=n_items, seed=seed
        )
        train_df, test_df = train_test_split_ratings(ratings_df, test_frac=0.2, seed=seed)
        models = build_all_models(train_df, item_genre_matrix, n_users, n_items, alpha=alpha)

        offline_results = {
            key: evaluate_model(model, test_df, k=k_eval) for key, model in models.items()
        }

        ab_results = simulate_ab_test(
            user_ids=users_df["user_id"].values,
            control_model=models["popularity"],
            treatment_model=models["hybrid"],
            user_taste_matrix=user_taste_matrix,
            item_genre_matrix=item_genre_matrix,
            n_recommendations=k_eval,
            base_ctr=base_ctr,
            seed=seed,
        )
        st.session_state["results"] = dict(
            users_df=users_df, items_df=items_df, ratings_df=ratings_df,
            models=models, offline_results=offline_results, ab_results=ab_results,
            k_eval=k_eval,
        )

res = st.session_state["results"]
models = res["models"]
offline_results = res["offline_results"]
ab_results = res["ab_results"]
k_eval = res["k_eval"]

tab1, tab2, tab3 = st.tabs(["📊 Offline Model Quality", "🧪 A/B Test Results", "🔎 Try a Recommendation"])

with tab1:
    st.subheader(f"Precision / Recall / NDCG @ {k_eval}")
    df = pd.DataFrame(offline_results).T
    df.index = [models[k].name for k in df.index]
    st.dataframe(
        df.style.format("{:.4f}").highlight_max(subset=["precision@k", "recall@k", "ndcg@k"], color="#c7e8ca"),
        use_container_width=True,
    )
    st.bar_chart(df[["precision@k", "recall@k", "ndcg@k"]])
    st.caption(
        "Popularity is non-personalized (same list for everyone), so it scores near zero on "
        "held-out relevance — that's expected and is exactly why it's a weak baseline in the "
        "A/B test. The Hybrid model combines CF's strength on active users with Content-Based's "
        "robustness for sparser profiles."
    )

with tab2:
    ctr_test = ab_results["ctr_test"]
    ctr_summary = ab_results["ctr_summary"]
    eng_test = ab_results["engagement_test"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Control CTR", f"{ctr_test['rate_control']:.2%}")
    c2.metric("Treatment CTR", f"{ctr_test['rate_treatment']:.2%}",
              delta=f"{ctr_test['lift_pct']:+.1f}%")
    c3.metric("p-value", f"{ctr_test['p_value']:.5f}")
    c4.metric("Significant @ α=0.05?", "✅ Yes" if ctr_test["significant"] else "❌ No")

    st.markdown(
        f"**Two-proportion z-test:** z = {ctr_test['z_stat']:.3f}, "
        f"95% CI on the CTR difference = [{ctr_test['ci_95'][0]:+.4f}, {ctr_test['ci_95'][1]:+.4f}]  \n"
        f"**Chi-square cross-check:** χ² = {ctr_test['chi2_stat']:.3f}, p = {ctr_test['chi2_p_value']:.5f}"
    )

    fig, ax = plt.subplots(figsize=(4, 3.2))
    arms = ["control", "treatment"]
    rates = [ctr_test["rate_control"], ctr_test["rate_treatment"]]
    ns = [ctr_summary.loc[a, "impressions"] for a in arms]
    errs = [1.96 * np.sqrt(r * (1 - r) / n) for r, n in zip(rates, ns)]
    ax.bar(["Control", "Treatment"], rates, yerr=errs, capsize=6, color=["#94a3b8", "#2563eb"])
    ax.set_ylabel("CTR")
    ax.set_title("CTR by arm (95% CI)")
    st.pyplot(fig, use_container_width=False)

    with st.expander("Secondary metric: engagement rate (click-conditional 'like/save')"):
        st.write(
            f"Control: {eng_test['rate_control']:.4f} · Treatment: {eng_test['rate_treatment']:.4f} · "
            f"Lift: {eng_test['lift_pct']:+.1f}% · p = {eng_test['p_value']:.5f}"
        )

    st.caption(
        "Methodology: users are randomly split 50/50. Clicks are simulated from a logistic "
        "function of each recommended item's true relevance to the user's hidden taste vector "
        "(unknown to any model) plus random noise — identical click-generation process for both "
        "arms, so any lift reflects genuinely better recommendations, not simulator bias."
    )

with tab3:
    uid = st.number_input("User ID", min_value=0, max_value=len(res["users_df"]) - 1, value=0)
    n_show = st.slider("How many recommendations", 3, 15, 10)
    cols = st.columns(4)
    for col, key in zip(cols, ["popularity", "collaborative", "content", "hybrid"]):
        model = models[key]
        recs = model.recommend(uid, n=n_show)
        titles = res["items_df"].set_index("item_id").loc[recs, ["title", "genres"]]
        col.markdown(f"**{model.name}**")
        col.dataframe(titles, use_container_width=True, hide_index=True)
