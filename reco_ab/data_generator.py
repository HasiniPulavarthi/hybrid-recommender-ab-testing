"""
data_generator.py
------------------
Generates a synthetic but *structurally realistic* movie-recommendation
dataset: users, items (movies), genre features, and rating interactions.

The key design choice: every user has a hidden ("ground-truth") taste
vector over genres. Ratings are generated as a noisy function of the
match between a user's taste and an item's genres, plus a popularity
bias (some movies are just more watched, à la real-world data).

That hidden taste vector is NOT given to any recommender model — it's
only used later, in ab_test.py, to simulate whether a *recommended*
item would genuinely appeal to the user (i.e. to simulate ground-truth
clicks for the A/B test). This mirrors how, in a real A/B test, users'
true preferences are unknown to the model but drive real click behavior.
"""

import numpy as np
import pandas as pd

GENRES = [
    "Action", "Comedy", "Drama", "Sci-Fi", "Romance",
    "Thriller", "Animation", "Horror", "Documentary", "Fantasy",
]


def generate_dataset(
    n_users: int = 600,
    n_items: int = 400,
    n_genres: int = len(GENRES),
    interactions_per_user: tuple = (10, 60),
    seed: int = 42,
):
    """
    Returns
    -------
    users_df : DataFrame[user_id]
    items_df : DataFrame[item_id, title, genres (list), popularity]
    ratings_df : DataFrame[user_id, item_id, rating]  (explicit 1-5 ratings)
    item_genre_matrix : np.ndarray (n_items x n_genres), binary
    user_taste_matrix : np.ndarray (n_users x n_genres), ground truth,
                         0-1 weights, NOT visible to the recommenders
    """
    rng = np.random.default_rng(seed)
    genres = GENRES[:n_genres]

    # ---- Items -------------------------------------------------------
    # Each item gets 1-3 genres. Popularity follows a power law so a
    # small set of items dominate raw interaction counts (like real
    # catalogs), which is exactly what trips up a naive popularity
    # baseline for users with niche taste.
    item_genre_matrix = np.zeros((n_items, n_genres), dtype=int)
    for i in range(n_items):
        k = rng.integers(1, 4)
        chosen = rng.choice(n_genres, size=k, replace=False)
        item_genre_matrix[i, chosen] = 1

    raw_pop = rng.pareto(a=1.5, size=n_items) + 0.1
    popularity = raw_pop / raw_pop.max()

    items_df = pd.DataFrame({
        "item_id": np.arange(n_items),
        "title": [f"Movie #{i:04d}" for i in range(n_items)],
        "genres": [
            "|".join(np.array(genres)[item_genre_matrix[i] == 1])
            for i in range(n_items)
        ],
        "popularity": popularity,
    })

    # ---- Users ---------------------------------------------------------
    # Each user's taste vector is drawn from a Dirichlet distribution so
    # tastes are sparse-ish (a few favorite genres) rather than uniform.
    user_taste_matrix = rng.dirichlet(alpha=np.ones(n_genres) * 0.6, size=n_users)
    users_df = pd.DataFrame({"user_id": np.arange(n_users)})

    # ---- Ratings ---------------------------------------------------------
    item_genre_norm = item_genre_matrix / np.clip(
        item_genre_matrix.sum(axis=1, keepdims=True), 1, None
    )

    rows = []
    for u in range(n_users):
        n_int = rng.integers(interactions_per_user[0], interactions_per_user[1])
        # Sampling probability mixes genuine taste-affinity with
        # popularity bias (people mostly watch what's popular AND
        # somewhat matches their taste) -> classic implicit-feedback
        # confound that a good hybrid model should cut through.
        affinity = item_genre_norm @ user_taste_matrix[u]
        sample_p = 0.65 * affinity + 0.35 * popularity
        sample_p = sample_p / sample_p.sum()
        items_sampled = rng.choice(n_items, size=min(n_int, n_items), replace=False, p=sample_p)

        for it in items_sampled:
            base = affinity[it]  # 0..~1
            noise = rng.normal(0, 0.12)
            rating = 1 + 4 * np.clip(base + noise, 0, 1)  # map to 1..5
            rows.append((u, it, round(float(rating))))

    ratings_df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating"])
    ratings_df["rating"] = ratings_df["rating"].clip(1, 5)
    ratings_df = ratings_df.drop_duplicates(subset=["user_id", "item_id"])

    return users_df, items_df, ratings_df, item_genre_matrix, user_taste_matrix


if __name__ == "__main__":
    users_df, items_df, ratings_df, ig, ut = generate_dataset()
    print(f"users={len(users_df)}  items={len(items_df)}  ratings={len(ratings_df)}")
    print(ratings_df.head())
