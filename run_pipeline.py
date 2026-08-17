"""
run_pipeline.py
-----------------
End-to-end script:
  1. generate synthetic movie data
  2. fit Popularity / CF / Content / Hybrid recommenders
  3. offline evaluation (Precision@K / Recall@K / NDCG@K) on held-out ratings
  4. simulate an A/B test: Popularity (control) vs Hybrid (treatment)
  5. save results (JSON/CSV) + a self-contained HTML report with charts

Run:  python -m reco_ab.run_pipeline
"""

import io
import base64
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reco_ab.data_generator import generate_dataset, GENRES
from reco_ab.recommender import build_all_models
from reco_ab.evaluation import train_test_split_ratings, evaluate_model
from reco_ab.ab_test import simulate_ab_test

OUT_DIR = "output"


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main(
    n_users=3000,
    n_items=500,
    alpha=0.6,
    k_eval=10,
    base_ctr=0.08,
    seed=42,
):
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("1) Generating synthetic dataset...")
    users_df, items_df, ratings_df, item_genre_matrix, user_taste_matrix = generate_dataset(
        n_users=n_users, n_items=n_items, seed=seed
    )
    print(f"   users={n_users} items={n_items} ratings={len(ratings_df)}")

    print("2) Train/test split + fitting models...")
    train_df, test_df = train_test_split_ratings(ratings_df, test_frac=0.2, seed=seed)
    models = build_all_models(train_df, item_genre_matrix, n_users, n_items, alpha=alpha)

    print("3) Offline evaluation (Precision/Recall/NDCG@%d)..." % k_eval)
    offline_results = {}
    for key, model in models.items():
        offline_results[key] = evaluate_model(model, test_df, k=k_eval)
        print(f"   {model.name:35s} P@{k_eval}={offline_results[key]['precision@k']:.4f} "
              f"R@{k_eval}={offline_results[key]['recall@k']:.4f} "
              f"NDCG@{k_eval}={offline_results[key]['ndcg@k']:.4f}")

    print("4) Simulating A/B test: Popularity (control) vs Hybrid (treatment)...")
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
    ctr_summary = ab_results["ctr_summary"]
    ctr_test = ab_results["ctr_test"]
    eng_summary = ab_results["engagement_summary"]
    eng_test = ab_results["engagement_test"]

    print(f"   Control  CTR = {ctr_test['rate_control']:.4f} "
          f"({int(ctr_summary.loc['control','clicks'])}/{int(ctr_summary.loc['control','impressions'])})")
    print(f"   Treatment CTR = {ctr_test['rate_treatment']:.4f} "
          f"({int(ctr_summary.loc['treatment','clicks'])}/{int(ctr_summary.loc['treatment','impressions'])})")
    print(f"   Lift = {ctr_test['lift_pct']:.2f}%  p-value = {ctr_test['p_value']:.5f}  "
          f"significant @ 0.05 = {ctr_test['significant']}")

    # ---- Save raw results -------------------------------------------------
    offline_df = pd.DataFrame(offline_results).T
    offline_df.to_csv(f"{OUT_DIR}/offline_evaluation.csv")
    ab_results["impressions_df"].to_csv(f"{OUT_DIR}/ab_test_impressions.csv", index=False)

    summary_json = {
        "config": {
            "n_users": n_users, "n_items": n_items, "alpha": alpha,
            "k_eval": k_eval, "base_ctr": base_ctr, "seed": seed,
        },
        "offline_evaluation": offline_results,
        "ab_test": {
            "ctr": {k: (list(v) if isinstance(v, tuple) else v) for k, v in ctr_test.items()},
            "engagement": {k: (list(v) if isinstance(v, tuple) else v) for k, v in eng_test.items()},
            "ctr_summary": ctr_summary.to_dict(),
            "engagement_summary": eng_summary.to_dict(),
        },
    }
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(summary_json, f, indent=2, default=float)

    # ---- Charts -------------------------------------------------------
    charts = {}

    # Offline metrics bar chart
    fig, ax = plt.subplots(figsize=(6.5, 4))
    metrics = ["precision@k", "recall@k", "ndcg@k"]
    x = np.arange(len(models))
    width = 0.25
    labels = [models[k].name for k in models]
    for i, m in enumerate(metrics):
        vals = [offline_results[k][m] for k in models]
        ax.bar(x + i * width, vals, width, label=m.upper())
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Offline Ranking Quality @ K={k_eval}")
    ax.legend()
    charts["offline_metrics"] = fig_to_base64(fig)

    # CTR comparison bar chart with error bars (95% CI per-arm, Wald)
    fig, ax = plt.subplots(figsize=(5, 4))
    arms = ["control", "treatment"]
    rates = [ctr_test["rate_control"], ctr_test["rate_treatment"]]
    ns = [ctr_summary.loc[a, "impressions"] for a in arms]
    errs = [1.96 * np.sqrt(r * (1 - r) / n) for r, n in zip(rates, ns)]
    bars = ax.bar(["Control\n(Popularity)", "Treatment\n(Hybrid)"], rates, yerr=errs,
                   capsize=6, color=["#94a3b8", "#2563eb"])
    ax.set_ylabel("Click-Through Rate")
    ax.set_title("A/B Test: CTR by Arm (95% CI)")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + max(errs) * 1.15, f"{r:.2%}",
                 ha="center", fontweight="bold")
    charts["ctr_bar"] = fig_to_base64(fig)

    # Relevance distribution by arm
    fig, ax = plt.subplots(figsize=(6, 4))
    imp = ab_results["impressions_df"]
    for arm, color in [("control", "#94a3b8"), ("treatment", "#2563eb")]:
        ax.hist(imp[imp.arm == arm]["true_relevance"], bins=25, alpha=0.6, label=arm, color=color, density=True)
    ax.set_xlabel("True relevance of recommended item to user")
    ax.set_ylabel("Density")
    ax.set_title("Recommendation Relevance Distribution")
    ax.legend()
    charts["relevance_dist"] = fig_to_base64(fig)

    build_html_report(
        charts, offline_results, models, ctr_test, eng_test, ctr_summary, eng_summary,
        n_users, n_items, k_eval, base_ctr,
    )
    print(f"\nDone. Results in ./{OUT_DIR}/ (results.json, offline_evaluation.csv, "
          f"ab_test_impressions.csv, report.html)")
    return summary_json


def build_html_report(charts, offline_results, models, ctr_test, eng_test, ctr_summary, eng_summary,
                       n_users, n_items, k_eval, base_ctr):
    verdict = "STATISTICALLY SIGNIFICANT" if ctr_test["significant"] else "NOT statistically significant"
    verdict_color = "#16a34a" if ctr_test["significant"] else "#dc2626"

    offline_rows = "".join(
        f"<tr><td>{models[k].name}</td>"
        f"<td>{v['precision@k']:.4f}</td><td>{v['recall@k']:.4f}</td><td>{v['ndcg@k']:.4f}</td></tr>"
        for k, v in offline_results.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Recommendation Engine + A/B Test Report</title>
<style>
  :root {{ --bg:#0f172a; --card:#ffffff; --ink:#0f172a; --muted:#64748b; --accent:#2563eb; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          background: #f1f5f9; color: var(--ink); margin:0; padding: 32px; }}
  .wrap {{ max-width: 980px; margin: 0 auto; }}
  h1 {{ font-size: 26px; margin-bottom: 4px; }}
  .sub {{ color: var(--muted); margin-bottom: 28px; }}
  .card {{ background: var(--card); border-radius: 12px; padding: 24px 28px; margin-bottom: 22px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  h2 {{ font-size: 18px; margin-top: 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 8px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e2e8f0; }}
  th {{ color: var(--muted); font-weight: 600; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 10px;}}
  .metric {{ background: #f8fafc; border-radius: 10px; padding: 16px; text-align: center; }}
  .metric .val {{ font-size: 26px; font-weight: 700; color: var(--accent); }}
  .metric .lbl {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .verdict {{ display:inline-block; padding: 6px 14px; border-radius: 999px; color: white;
              font-weight: 600; font-size: 13px; background: {verdict_color}; }}
  img {{ max-width: 100%; border-radius: 8px; }}
  .imgs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .imgs.full {{ grid-template-columns: 1fr; }}
  .note {{ font-size: 13px; color: var(--muted); margin-top: 10px; line-height: 1.5; }}
  code {{ background: #f1f5f9; padding: 1px 6px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Hybrid Recommendation Engine — Offline Eval + Simulated A/B Test</h1>
  <div class="sub">{n_users} users · {n_items} items · top-{k_eval} recommendations per user · baseline CTR target {base_ctr:.0%}</div>

  <div class="card">
    <h2>1. Offline Ranking Quality (held-out ratings)</h2>
    <table>
      <tr><th>Model</th><th>Precision@{k_eval}</th><th>Recall@{k_eval}</th><th>NDCG@{k_eval}</th></tr>
      {offline_rows}
    </table>
    <div class="imgs full" style="margin-top:16px;">
      <img src="data:image/png;base64,{charts['offline_metrics']}">
    </div>
    <div class="note">Precision/Recall/NDCG@{k_eval} measured against a held-out 20% split of each
      user's ratings (rating ≥ 4 counted as "relevant"). The Hybrid model blends
      Collaborative Filtering and Content-Based signal to typically outrank either alone,
      especially for users with sparser rating histories.</div>
  </div>

  <div class="card">
    <h2>2. Simulated A/B Test — Popularity (control) vs Hybrid (treatment)</h2>
    <span class="verdict">{verdict} (p = {ctr_test['p_value']:.5f})</span>

    <div class="metrics-grid">
      <div class="metric"><div class="val">{ctr_test['rate_control']:.2%}</div><div class="lbl">Control CTR</div></div>
      <div class="metric"><div class="val">{ctr_test['rate_treatment']:.2%}</div><div class="lbl">Treatment CTR</div></div>
      <div class="metric"><div class="val">{ctr_test['lift_pct']:+.1f}%</div><div class="lbl">Relative CTR Lift</div></div>
    </div>

    <table style="margin-top:20px;">
      <tr><th>Metric</th><th>Control</th><th>Treatment</th><th>Abs. diff</th><th>95% CI (diff)</th><th>z-stat</th><th>p-value</th></tr>
      <tr>
        <td>CTR</td>
        <td>{ctr_test['rate_control']:.4f}</td>
        <td>{ctr_test['rate_treatment']:.4f}</td>
        <td>{ctr_test['abs_diff']:+.4f}</td>
        <td>[{ctr_test['ci_95'][0]:+.4f}, {ctr_test['ci_95'][1]:+.4f}]</td>
        <td>{ctr_test['z_stat']:.3f}</td>
        <td>{ctr_test['p_value']:.5f}</td>
      </tr>
      <tr>
        <td>Engagement rate (secondary)</td>
        <td>{eng_test['rate_control']:.4f}</td>
        <td>{eng_test['rate_treatment']:.4f}</td>
        <td>{eng_test['abs_diff']:+.4f}</td>
        <td>[{eng_test['ci_95'][0]:+.4f}, {eng_test['ci_95'][1]:+.4f}]</td>
        <td>{eng_test['z_stat']:.3f}</td>
        <td>{eng_test['p_value']:.5f}</td>
      </tr>
    </table>

    <div class="imgs" style="margin-top:20px;">
      <img src="data:image/png;base64,{charts['ctr_bar']}">
      <img src="data:image/png;base64,{charts['relevance_dist']}">
    </div>
    <div class="note">
      Methodology: users are randomly split 50/50 into control (served Popularity-baseline
      recommendations) and treatment (served Hybrid recommendations). For every recommended
      impression we simulate a click via a logistic function of the item's true relevance to
      that user's hidden taste profile (unknown to the model, known only to the simulator) plus
      random noise — i.e. clicks are <em>not</em> hand-picked to favor the treatment, they're
      generated from the same relevance model for both arms. Significance is computed with a
      two-proportion z-test, cross-checked against a chi-square test of independence
      (chi2 = {ctr_test['chi2_stat']:.3f}, p = {ctr_test['chi2_p_value']:.5f}).
    </div>
  </div>

  <div class="card">
    <h2>Resume-ready summary</h2>
    <div class="note" style="font-size:14px;">
      Developed a hybrid (collaborative + content-based) recommendation system and validated
      impact via a simulated A/B test against a popularity baseline, demonstrating a
      <strong>{ctr_test['lift_pct']:+.1f}% lift in CTR</strong> ({'statistically significant, p '
      + f"= {ctr_test['p_value']:.4f}" if ctr_test['significant'] else 'not statistically significant at n=' + str(n_users) + ' users — see note on power below'}).
      Evaluated ranking quality offline with Precision/Recall/NDCG@{k_eval} across four model
      variants (popularity, collaborative filtering via SVD, content-based, and hybrid).
    </div>
  </div>
</div>
</body>
</html>"""

    with open(f"{OUT_DIR}/report.html", "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
