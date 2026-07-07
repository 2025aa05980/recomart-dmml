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

    return svd_results, cbf_results


if __name__ == "__main__":
    run_model_training()
