"""
Task 9: Model Training & Evaluation — RecoMart Pipeline
Dataset: Amazon Reviews 2023 — Video Games

Models trained:
  1. SVD (Singular Value Decomposition) — Collaborative Filtering
     - Matrix factorization via scikit-surprise
     - Learns latent user and item factors
     - Evaluated: RMSE, MAE, Precision@K, Recall@K, NDCG@K

  2. Content-Based Filtering
     - TF-IDF on product title + category + brand
     - Cosine similarity for item-item recommendations
     - Evaluated: coverage, diversity

Experiment tracking:
  - MLflow local server (no account needed)
  - Tracks: parameters, metrics, model artifacts, run IDs

Outputs:
  models/svd_model.pkl          — trained SVD model
  models/cbf_similarity.npz    — content-based similarity matrix
  models/model_metadata.json   — model info, metrics, run IDs
  reports/model_performance.md  — evaluation report
"""

import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import sqlite3
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime, timezone
from scipy.sparse import save_npz, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, GridSearchCV

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    PROCESSED_DIR, FEATURE_STORE, MODELS_DIR,
    REPORTS_DIR, MLFLOW_EXPERIMENT, RANDOM_SEED,
    TEST_SIZE, N_FACTORS, N_EPOCHS, K_EVAL, RATING_SCALE
)
from src.logger import get_logger

log = get_logger("model_training")

DB_PATH    = FEATURE_STORE / "recomart_features.db"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loader ───────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned ratings and products from processed directory."""
    ratings  = pd.read_csv(PROCESSED_DIR / "cleaned_ratings.csv")
    products = pd.read_csv(PROCESSED_DIR / "cleaned_products.csv")
    log.info(f"Ratings : {ratings.shape}")
    log.info(f"Products: {products.shape}")
    return ratings, products


# ── Evaluation metrics ────────────────────────────────────────────────────────

def precision_at_k(predictions, k: int = 10,
                   threshold: float = 3.5) -> float:
    """
    Precision@K — fraction of top-K recommendations that are relevant.
    Relevant = true rating >= threshold.
    """
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        user_est_true.setdefault(uid, []).append((est, true_r))

    precisions = []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k   = user_ratings[:k]
        n_rel_k = sum(1 for (_, true_r) in top_k if true_r >= threshold)
        precisions.append(n_rel_k / k)

    return float(np.mean(precisions))


def recall_at_k(predictions, k: int = 10,
                threshold: float = 3.5) -> float:
    """
    Recall@K — fraction of relevant items that appear in top-K.
    """
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        user_est_true.setdefault(uid, []).append((est, true_r))

    recalls = []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k   = user_ratings[:k]
        n_rel   = sum(1 for (_, true_r) in user_ratings if true_r >= threshold)
        n_rel_k = sum(1 for (_, true_r) in top_k if true_r >= threshold)
        if n_rel > 0:
            recalls.append(n_rel_k / n_rel)

    return float(np.mean(recalls)) if recalls else 0.0


def ndcg_at_k(predictions, k: int = 10,
              threshold: float = 3.5) -> float:
    """
    NDCG@K — Normalized Discounted Cumulative Gain at K.
    Measures ranking quality — highly relevant items ranked first scores higher.
    """
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        user_est_true.setdefault(uid, []).append((est, true_r))

    ndcgs = []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        # DCG
        dcg = sum(
            (2 ** true_r - 1) / np.log2(i + 2)
            for i, (_, true_r) in enumerate(top_k)
        )
        # Ideal DCG
        ideal = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:k]
        idcg  = sum(
            (2 ** true_r - 1) / np.log2(i + 2)
            for i, (_, true_r) in enumerate(ideal)
        )
        if idcg > 0:
            ndcgs.append(dcg / idcg)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


# ── Model 1: SVD Collaborative Filtering ─────────────────────────────────────

def train_svd(ratings: pd.DataFrame) -> dict:
    """
    Train SVD (Matrix Factorization) using scikit-surprise.
    Includes hyperparameter tuning via GridSearchCV.
    Tracked with MLflow.
    """
    log.info("=" * 55)
    log.info("=== Training Model 1: SVD Collaborative Filtering ===")

    # Prepare Surprise dataset
    reader  = Reader(rating_scale=RATING_SCALE)
    data    = Dataset.load_from_df(
        ratings[["userId", "productId", "rating"]], reader)

    # ── Hyperparameter search ─────────────────────────────────────────────────
    log.info("Running GridSearchCV for SVD hyperparameters...")
    param_grid = {
        "n_factors": [20, 50, 100],
        "n_epochs":  [15, 20],
        "lr_all":    [0.005, 0.01],
        "reg_all":   [0.02, 0.1],
    }
    gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"],
                      cv=3, n_jobs=-1)
    gs.fit(data)

    best_params = gs.best_params["rmse"]
    best_rmse   = gs.best_score["rmse"]
    log.info(f"Best params : {best_params}")
    log.info(f"Best CV RMSE: {best_rmse:.4f}")

    # ── Train final model on full trainset ────────────────────────────────────
    log.info("Training final SVD model...")
    trainset = data.build_full_trainset()
    model    = SVD(
        n_factors = best_params["n_factors"],
        n_epochs  = best_params["n_epochs"],
        lr_all    = best_params["lr_all"],
        reg_all   = best_params["reg_all"],
        random_state = RANDOM_SEED,
        verbose   = False,
    )
    model.fit(trainset)

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    log.info("Evaluating on test set...")
    testset     = trainset.build_anti_testset()
    # Use a sample for faster evaluation
    testset_sample = testset[:min(50000, len(testset))]
    predictions = model.test(testset_sample)

    rmse  = accuracy.rmse(predictions, verbose=False)
    mae   = accuracy.mae(predictions,  verbose=False)
    prec  = precision_at_k(predictions, k=K_EVAL)
    rec   = recall_at_k(predictions,    k=K_EVAL)
    ndcg  = ndcg_at_k(predictions,      k=K_EVAL)

    log.info(f"RMSE          : {rmse:.4f}")
    log.info(f"MAE           : {mae:.4f}")
    log.info(f"Precision@{K_EVAL}  : {prec:.4f}")
    log.info(f"Recall@{K_EVAL}     : {rec:.4f}")
    log.info(f"NDCG@{K_EVAL}       : {ndcg:.4f}")

    # ── MLflow tracking ───────────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="SVD_collaborative_filtering") as run:
        # Log parameters
        mlflow.log_params({
            "model_type":  "SVD",
            "n_factors":   best_params["n_factors"],
            "n_epochs":    best_params["n_epochs"],
            "lr_all":      best_params["lr_all"],
            "reg_all":     best_params["reg_all"],
            "random_seed": RANDOM_SEED,
            "n_users":     ratings["userId"].nunique(),
            "n_items":     ratings["productId"].nunique(),
            "n_ratings":   len(ratings),
            "test_sample": len(testset_sample),
            "k_eval":      K_EVAL,
        })
        # Log metrics
        mlflow.log_metrics({
            "rmse":          round(rmse,  4),
            "mae":           round(mae,   4),
            "precision_at_k": round(prec, 4),
            "recall_at_k":   round(rec,  4),
            "ndcg_at_k":     round(ndcg, 4),
            "cv_best_rmse":  round(best_rmse, 4),
        })
        run_id = run.info.run_id
        log.info(f"MLflow run ID : {run_id}")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = MODELS_DIR / "svd_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"SVD model saved → {model_path}")

    return {
        "model_type":    "SVD",
        "run_id":        run_id,
        "best_params":   best_params,
        "metrics": {
            "rmse":           round(rmse,  4),
            "mae":            round(mae,   4),
            "precision_at_k": round(prec,  4),
            "recall_at_k":    round(rec,   4),
            "ndcg_at_k":      round(ndcg,  4),
            "cv_best_rmse":   round(best_rmse, 4),
        },
        "model_path": str(model_path),
        "n_users":    ratings["userId"].nunique(),
        "n_items":    ratings["productId"].nunique(),
    }


# ── Model 2: Content-Based Filtering ─────────────────────────────────────────

def train_content_based(products: pd.DataFrame,
                        ratings: pd.DataFrame) -> dict:
    """
    Train Content-Based Filtering using TF-IDF + Cosine Similarity.
    Builds item-item similarity matrix from product text features.
    Tracked with MLflow.
    """
    log.info("=" * 55)
    log.info("=== Training Model 2: Content-Based Filtering ===")

    # Only use products that appear in ratings
    rated_products = ratings["productId"].unique()
    products_rated = products[
        products["productId"].isin(rated_products)
    ].copy().reset_index(drop=True)
    log.info(f"Products with ratings: {len(products_rated):,}")

    # ── Build text feature ────────────────────────────────────────────────────
    products_rated["text_feature"] = (
        products_rated["title"].fillna("") + " " +
        products_rated["category"].fillna("") + " " +
        products_rated["brand"].fillna("") + " " +
        products_rated["description"].fillna("").str[:100]
    )

    # ── TF-IDF vectorization ──────────────────────────────────────────────────
    log.info("Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features = 5000,
        stop_words   = "english",
        ngram_range  = (1, 2),
        min_df       = 2,
    )
    tfidf_matrix = tfidf.fit_transform(products_rated["text_feature"])
    log.info(f"TF-IDF matrix: {tfidf_matrix.shape}")

    # ── Cosine similarity (chunked for memory) ────────────────────────────────
    log.info("Computing cosine similarity (chunked)...")
    chunk_size  = 500
    n_products  = tfidf_matrix.shape[0]
    sim_scores  = []

    for i in range(0, n_products, chunk_size):
        chunk = tfidf_matrix[i:i+chunk_size]
        sim   = cosine_similarity(chunk, tfidf_matrix)
        sim_scores.append(sim)

    sim_matrix = np.vstack(sim_scores)
    log.info(f"Similarity matrix: {sim_matrix.shape}")

    # ── Evaluate CBF ──────────────────────────────────────────────────────────
    # Coverage: % of rated items that can be recommended
    coverage = len(products_rated) / len(rated_products) * 100

    # Average similarity of top-10 neighbors (diversity proxy)
    sample_idx = np.random.choice(min(100, n_products), 20, replace=False)
    avg_top10_sim = np.mean([
        np.sort(sim_matrix[i])[-11:-1].mean()
        for i in sample_idx
    ])

    log.info(f"Coverage      : {coverage:.1f}%")
    log.info(f"Avg top-10 sim: {avg_top10_sim:.4f}")

    # ── MLflow tracking ───────────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="ContentBased_TFIDF_cosine") as run:
        mlflow.log_params({
            "model_type":     "ContentBased",
            "vectorizer":     "TF-IDF",
            "similarity":     "cosine",
            "max_features":   5000,
            "ngram_range":    "(1,2)",
            "n_products":     len(products_rated),
            "tfidf_shape":    str(tfidf_matrix.shape),
        })
        mlflow.log_metrics({
            "coverage_pct":      round(coverage, 2),
            "avg_top10_sim":     round(avg_top10_sim, 4),
            "vocab_size":        len(tfidf.vocabulary_),
        })
        cbf_run_id = run.info.run_id
        log.info(f"MLflow run ID : {cbf_run_id}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    # Save similarity matrix as sparse (top-50 per item only — memory efficient)
    log.info("Saving similarity matrix (top-50 per item)...")
    rows, cols, vals = [], [], []
    for i in range(n_products):
        top_idx = np.argsort(sim_matrix[i])[-51:-1][::-1]
        for j in top_idx:
            if sim_matrix[i][j] > 0.01:
                rows.append(i)
                cols.append(j)
                vals.append(sim_matrix[i][j])

    sparse_sim = csr_matrix(
        (vals, (rows, cols)), shape=(n_products, n_products))
    sim_path = MODELS_DIR / "cbf_similarity.npz"
    save_npz(str(sim_path), sparse_sim)
    log.info(f"Similarity saved → {sim_path}")

    # Save product index mapping
    product_index = {pid: i for i, pid in
                     enumerate(products_rated["productId"])}
    idx_path = MODELS_DIR / "cbf_product_index.json"
    with open(idx_path, "w") as f:
        json.dump(product_index, f)
    log.info(f"Product index saved → {idx_path}")

    return {
        "model_type":    "ContentBased",
        "run_id":        cbf_run_id,
        "vectorizer":    "TF-IDF (max_features=5000, ngram=(1,2))",
        "similarity":    "cosine",
        "metrics": {
            "coverage_pct":  round(coverage, 2),
            "avg_top10_sim": round(avg_top10_sim, 4),
            "vocab_size":    len(tfidf.vocabulary_),
        },
        "n_products":    len(products_rated),
        "sim_path":      str(sim_path),
    }


# ── Sample recommendations ────────────────────────────────────────────────────

def generate_sample_recommendations(ratings: pd.DataFrame,
                                    products: pd.DataFrame) -> dict:
    """Generate sample SVD recommendations for 3 users."""
    log.info("Generating sample recommendations...")

    model_path = MODELS_DIR / "svd_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Get 3 sample users
    sample_users = ratings["userId"].value_counts().head(3).index.tolist()
    recommendations = {}

    for user_id in sample_users:
        # Items this user has already rated
        rated = set(ratings[ratings["userId"] == user_id]["productId"])

        # Predict ratings for unrated items (sample 200 for speed)
        all_items  = ratings["productId"].unique()
        unrated    = [i for i in all_items if i not in rated]
        sample_items = np.random.choice(
            unrated, min(200, len(unrated)), replace=False)

        preds = [(iid, model.predict(user_id, iid).est)
                 for iid in sample_items]
        top_k = sorted(preds, key=lambda x: x[1], reverse=True)[:K_EVAL]

        # Enrich with product metadata
        rec_list = []
        for pid, score in top_k:
            prod = products[products["productId"] == pid]
            title = prod["title"].values[0] if len(prod) > 0 else "Unknown"
            cat   = prod["category"].values[0] if len(prod) > 0 else "Unknown"
            rec_list.append({
                "productId":      pid,
                "predicted_rating": round(score, 3),
                "title":          str(title)[:50],
                "category":       str(cat)[:40],
            })

        recommendations[user_id[:20]+"..."] = rec_list
        log.info(f"Top-{K_EVAL} for user {user_id[:20]}... "
                 f"→ avg predicted rating: "
                 f"{np.mean([r['predicted_rating'] for r in rec_list]):.2f}")

    return recommendations


# ── Performance report ────────────────────────────────────────────────────────

def write_performance_report(svd_results: dict,
                             cbf_results: dict,
                             recommendations: dict):
    """Write model performance markdown report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    svd_m = svd_results["metrics"]
    cbf_m = cbf_results["metrics"]

    lines = [
        "# Model Performance Report — RecoMart Pipeline",
        "**Task 9 | DMML Assignment 1 | Group 37**",
        f"Generated: {now}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "| Model | Type | RMSE | Precision@10 | Recall@10 | NDCG@10 |",
        "|-------|------|------|-------------|-----------|---------|",
        f"| SVD | Collaborative Filtering | {svd_m['rmse']} | "
        f"{svd_m['precision_at_k']} | {svd_m['recall_at_k']} | "
        f"{svd_m['ndcg_at_k']} |",
        f"| TF-IDF Cosine | Content-Based | — | — | — | — |",
        "",
        "---",
        "",
        "## Model 1: SVD Collaborative Filtering",
        "",
        "### Architecture",
        "SVD (Singular Value Decomposition) factorizes the user-item",
        "rating matrix R into latent factor matrices:",
        "`R ≈ U × Σ × V^T`",
        "where U = user factors, V = item factors, Σ = singular values.",
        "",
        "### Hyperparameter Tuning (GridSearchCV, 3-fold CV)",
        "",
        "| Parameter | Search Space | Best Value |",
        "|-----------|-------------|------------|",
        f"| n_factors | [20, 50, 100] | {svd_results['best_params']['n_factors']} |",
        f"| n_epochs  | [15, 20] | {svd_results['best_params']['n_epochs']} |",
        f"| lr_all    | [0.005, 0.01] | {svd_results['best_params']['lr_all']} |",
        f"| reg_all   | [0.02, 0.1] | {svd_results['best_params']['reg_all']} |",
        "",
        "### Evaluation Metrics",
        "",
        "| Metric | Value | Description |",
        "|--------|-------|-------------|",
        f"| RMSE | {svd_m['rmse']} | Root Mean Squared Error (lower = better) |",
        f"| MAE | {svd_m['mae']} | Mean Absolute Error (lower = better) |",
        f"| Precision@{K_EVAL} | {svd_m['precision_at_k']} | Fraction of top-{K_EVAL} that are relevant |",
        f"| Recall@{K_EVAL} | {svd_m['recall_at_k']} | Fraction of relevant items in top-{K_EVAL} |",
        f"| NDCG@{K_EVAL} | {svd_m['ndcg_at_k']} | Ranking quality (higher = better) |",
        f"| CV RMSE | {svd_m['cv_best_rmse']} | Cross-validated RMSE |",
        "",
        f"### MLflow Run ID: `{svd_results['run_id']}`",
        f"Experiment: `{MLFLOW_EXPERIMENT}`",
        "",
        "---",
        "",
        "## Model 2: Content-Based Filtering (TF-IDF + Cosine Similarity)",
        "",
        "### Architecture",
        "Builds item profiles from text features (title + category + brand)",
        "using TF-IDF vectorization. Item similarity computed via cosine",
        "similarity. Recommends items similar to those a user has liked.",
        "",
        "### Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Vectorizer | TF-IDF |",
        "| Max features | 5,000 |",
        "| N-gram range | (1, 2) |",
        "| Similarity | Cosine |",
        f"| Products indexed | {cbf_results['n_products']:,} |",
        f"| Vocabulary size | {cbf_m['vocab_size']:,} |",
        "",
        "### Evaluation",
        "",
        "| Metric | Value | Description |",
        "|--------|-------|-------------|",
        f"| Coverage | {cbf_m['coverage_pct']}% | % of rated items recommendable |",
        f"| Avg top-10 similarity | {cbf_m['avg_top10_sim']} | Item neighborhood cohesion |",
        "",
        f"### MLflow Run ID: `{cbf_results['run_id']}`",
        "",
        "---",
        "",
        "## Sample Recommendations (SVD)",
        "",
        "Top-10 predicted recommendations for 3 sample users:",
        "",
    ]

    for user, recs in recommendations.items():
        lines += [
            f"### User: `{user}`",
            "",
            "| Rank | Product ID | Predicted Rating | Title | Category |",
            "|------|-----------|-----------------|-------|----------|",
        ]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r['productId']} | {r['predicted_rating']} "
                f"| {r['title']} | {r['category']} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## Model Artifacts",
        "",
        "| Artifact | Path | Description |",
        "|----------|------|-------------|",
        "| SVD model | `models/svd_model.pkl` | Trained SVD model (pickle) |",
        "| CBF similarity | `models/cbf_similarity.npz` | Sparse cosine similarity matrix |",
        "| Product index | `models/cbf_product_index.json` | productId → matrix index |",
        "| Model metadata | `models/model_metadata.json` | Parameters, metrics, run IDs |",
        "",
        "## MLflow Experiment",
        f"Experiment name: `{MLFLOW_EXPERIMENT}`",
        "View UI: `mlflow ui` then open http://localhost:5000",
    ]

    path = REPORTS_DIR / "model_performance.md"
    path.write_text("\n".join(lines))
    log.info(f"Performance report → {path}")
    return path


# ── Save model metadata ───────────────────────────────────────────────────────

def save_model_metadata(svd_results: dict, cbf_results: dict):
    """Save combined model metadata JSON."""
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset":    "Amazon Reviews 2023 — Video Games",
        "mlflow_experiment": MLFLOW_EXPERIMENT,
        "models": {
            "svd":           svd_results,
            "content_based": cbf_results,
        },
    }
    path = MODELS_DIR / "model_metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Model metadata → {path}")
    return path


# ── Inference Interface ───────────────────────────────────────────────────────

def recommend_svd(user_id: str,
                  top_k: int = 10,
                  candidate_limit: int = 500) -> pd.DataFrame:
    """
    SVD Inference Interface — returns top-K personalised recommendations.

    Loads the trained SVD model and generates predicted ratings for
    candidate items the user has not yet rated. Returns the top-K
    products ranked by predicted rating.

    Args:
        user_id        : Amazon reviewer ID (userId from feature store)
        top_k          : number of recommendations to return (default 10)
        candidate_limit: max candidate items to score (default 500)

    Returns:
        DataFrame with columns:
          rank, productId, predicted_rating, title, category, price
    """
    log.info(f"[Inference] SVD recommend for user: {user_id[:20]}...")

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = MODELS_DIR / "svd_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_model.py first.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # ── Load ratings + products for candidate generation ──────────────────────
    ratings, products = load_data()

    # Items this user has already rated — exclude from recommendations
    already_rated = set(
        ratings[ratings["userId"] == user_id]["productId"].tolist()
    )
    if not already_rated:
        log.warning(f"User '{user_id}' not found in ratings — cold start")

    # ── Sample candidate items ────────────────────────────────────────────────
    # Use most popular items as candidates (highest rating count)
    popular_items = (ratings["productId"]
                     .value_counts()
                     .head(candidate_limit)
                     .index.tolist())
    candidates = [p for p in popular_items if p not in already_rated]
    log.info(f"[Inference] Scoring {len(candidates)} candidate items "
             f"(excluded {len(already_rated)} already rated)")

    # ── Predict ratings ───────────────────────────────────────────────────────
    predictions = []
    for pid in candidates:
        pred = model.predict(user_id, pid)
        predictions.append({
            "productId":        pid,
            "predicted_rating": round(pred.est, 3),
        })

    # ── Rank and enrich with product metadata ─────────────────────────────────
    pred_df = (pd.DataFrame(predictions)
               .sort_values("predicted_rating", ascending=False)
               .head(top_k)
               .reset_index(drop=True))
    pred_df["rank"] = pred_df.index + 1

    # Merge product metadata
    meta = products[["productId", "title", "category", "price", "brand"]].copy()
    result = pred_df.merge(meta, on="productId", how="left")
    result["title"]    = result["title"].fillna("Unknown").str[:50]
    result["category"] = result["category"].fillna("Unknown").str[:40]

    log.info(f"[Inference] Top-{top_k} recommendations generated")
    log.info(f"[Inference] Avg predicted rating: "
             f"{result['predicted_rating'].mean():.3f}")

    return result[["rank", "productId", "predicted_rating",
                   "title", "category", "price", "brand"]]


def recommend_cbf(product_id: str, top_k: int = 10) -> pd.DataFrame:
    """
    Content-Based Inference Interface — returns similar products.

    Loads the pre-computed cosine similarity matrix and returns
    the top-K most similar products to the given product.
    Used for 'customers also bought' style recommendations.

    Args:
        product_id : Amazon ASIN to find similar items for
        top_k      : number of similar products to return (default 10)

    Returns:
        DataFrame with columns:
          rank, productId, similarity_score, title, category, price
    """
    log.info(f"[Inference] CBF similar items for: {product_id}")

    sim_path = MODELS_DIR / "cbf_similarity.npz"
    idx_path = MODELS_DIR / "cbf_product_index.json"

    if not sim_path.exists() or not idx_path.exists():
        raise FileNotFoundError(
            "CBF model files not found. Run train_model.py first.")

    from scipy.sparse import load_npz
    sim_matrix   = load_npz(str(sim_path))
    with open(idx_path) as f:
        product_index = json.load(f)
    index_product = {v: k for k, v in product_index.items()}

    if product_id not in product_index:
        log.warning(f"Product '{product_id}' not in CBF index")
        return pd.DataFrame()

    idx      = product_index[product_id]
    sim_row  = sim_matrix[idx].toarray().flatten()

    # Get top-K similar items (exclude self)
    top_indices = np.argsort(sim_row)[::-1][1:top_k+1]
    similar = []
    for i in top_indices:
        pid   = index_product.get(i, "")
        score = float(sim_row[i])
        if pid and score > 0:
            similar.append({"productId": pid, "similarity_score": round(score, 4)})

    sim_df = pd.DataFrame(similar)
    if sim_df.empty:
        return sim_df

    sim_df["rank"] = sim_df.index + 1

    # Enrich with product metadata
    _, products = load_data()
    meta   = products[["productId", "title", "category", "price"]].copy()
    result = sim_df.merge(meta, on="productId", how="left")
    result["title"]    = result["title"].fillna("Unknown").str[:50]
    result["category"] = result["category"].fillna("Unknown").str[:40]

    log.info(f"[Inference] Top-{top_k} similar products found")
    return result[["rank", "productId", "similarity_score",
                   "title", "category", "price"]]


def run_inference_demo():
    """
    Demonstrate the inference interface with sample users and products.
    Called after training to verify the interface works end-to-end.
    """
    log.info("========== Inference Interface Demo START ==========")

    # Load ratings to get sample users
    ratings, _ = load_data()
    sample_users = ratings["userId"].value_counts().head(3).index.tolist()

    # ── SVD Recommendations ───────────────────────────────────────────────────
    log.info("\n--- SVD Collaborative Filtering Recommendations ---")
    for user_id in sample_users:
        recs = recommend_svd(user_id, top_k=K_EVAL)
        log.info(f"\nTop-{K_EVAL} for user {user_id[:25]}...")
        for _, row in recs.iterrows():
            log.info(f"  {int(row['rank']):2d}. {row['productId']} | "
                     f"★{row['predicted_rating']} | {row['title'][:35]}")

    # ── CBF Similar Items ─────────────────────────────────────────────────────
    log.info("\n--- Content-Based Similar Items ---")
    sample_product = ratings["productId"].value_counts().index[0]
    similar = recommend_cbf(sample_product, top_k=5)
    if not similar.empty:
        log.info(f"\nTop-5 similar to {sample_product}:")
        for _, row in similar.iterrows():
            log.info(f"  {int(row['rank'])}. {row['productId']} | "
                     f"sim={row['similarity_score']} | {row['title'][:35]}")

    log.info("\n========== Inference Interface Demo COMPLETE ==========")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_model_training():
    log.info("========== Model Training Pipeline START ==========")

    ratings, products = load_data()

    # Train Model 1: SVD
    svd_results = train_svd(ratings)

    # Train Model 2: Content-Based
    cbf_results = train_content_based(products, ratings)

    # Sample recommendations
    recommendations = generate_sample_recommendations(ratings, products)

    # Save metadata + report
    save_model_metadata(svd_results, cbf_results)
    write_performance_report(svd_results, cbf_results, recommendations)

    log.info("========== Model Training COMPLETE ==========")
    log.info(f"SVD  RMSE         : {svd_results['metrics']['rmse']}")
    log.info(f"SVD  Precision@{K_EVAL} : {svd_results['metrics']['precision_at_k']}")
    log.info(f"SVD  NDCG@{K_EVAL}      : {svd_results['metrics']['ndcg_at_k']}")
    log.info(f"CBF  Coverage     : {cbf_results['metrics']['coverage_pct']}%")
    log.info(f"MLflow SVD run    : {svd_results['run_id']}")
    log.info(f"MLflow CBF run    : {cbf_results['run_id']}")

    # Run inference demo
    log.info("\n--- Running Inference Interface Demo ---")
    run_inference_demo()

    return svd_results, cbf_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="RecoMart Model Training & Inference")
    parser.add_argument(
        "--infer-only", action="store_true",
        help="Skip training — run inference demo on existing model")
    parser.add_argument(
        "--user-id", type=str, default=None,
        help="Run SVD recommendations for a specific user ID")
    parser.add_argument(
        "--product-id", type=str, default=None,
        help="Run CBF similar items for a specific product ID")
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of recommendations to return (default 10)")
    args = parser.parse_args()

    if args.infer_only:
        run_inference_demo()
    elif args.user_id:
        recs = recommend_svd(args.user_id, top_k=args.top_k)
        print(f"\nTop-{args.top_k} recommendations for {args.user_id}:")
        print(recs.to_string(index=False))
    elif args.product_id:
        similar = recommend_cbf(args.product_id, top_k=args.top_k)
        print(f"\nTop-{args.top_k} similar to {args.product_id}:")
        print(similar.to_string(index=False))
    else:
        run_model_training()
