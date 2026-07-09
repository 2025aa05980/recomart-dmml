"""
Task 10: Pipeline Orchestration — RecoMart Pipeline
Tool: Prefect 3.x

Orchestrates the complete end-to-end data pipeline as a Prefect flow:

DAG structure:
  ingest_data
      ↓
  validate_data
      ↓
  prepare_data (EDA cleaning)
      ↓
  engineer_features
      ↓
  update_feature_store
      ↓
  train_models
      ↓
  pipeline_complete (summary)

Features:
  - @flow and @task decorators
  - Task-level error handling and retries
  - Pipeline state logging
  - Failure notifications via log
  - Run summary report

Usage:
  # Run full pipeline once
  python src/orchestration/pipeline_dag.py

  # Run with Prefect UI monitoring
  prefect server start          (in separate terminal)
  python src/orchestration/pipeline_dag.py

  # View UI
  http://localhost:4200
"""

import sys
import time
import traceback
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta

from config import (
    BASE_DIR, INTERACTIONS_DIR, PRODUCTS_DIR,
    PROCESSED_DIR, FEATURE_STORE, MODELS_DIR,
    REPORTS_DIR, DATASET_NAME, SAMPLE_USERS,
    RATING_SCALE, RANDOM_SEED
)


# ── Helper ────────────────────────────────────────────────────────────────────

def load_latest_csv(directory: Path, prefix: str) -> pd.DataFrame:
    """Load most recent timestamped CSV."""
    expected_len = len(prefix) + 1 + 8 + 1 + 6
    files = sorted([
        f for f in directory.glob(f"{prefix}_*.csv")
        if len(f.stem) == expected_len
        and f.stem[len(prefix)+1:].replace("_","").isdigit()
    ], reverse=True)
    if not files:
        raise FileNotFoundError(
            f"No timestamped {prefix} CSV in {directory}. "
            f"Run ingest_data.py first."
        )
    return pd.read_csv(str(files[0])), files[0].name


# ── Task 0: API Server ────────────────────────────────────────────────────────

@task(
    name="start_api_server",
    description="Start the RecoMart Product Catalog REST API server",
    tags=["ingestion", "api"],
)
def task_start_api_server() -> dict:
    """
    Start the local Flask REST API server as a background process.
    The ingestion task calls this API to fetch product data (Source Type 2).
    """
    logger = get_run_logger()
    logger.info("=== Task: Start Product Catalog REST API ===")
    import subprocess, sys, time as _time

    try:
        # Check if already running
        try:
            import requests as _req
            r = _req.get("http://127.0.0.1:8080/api/health", timeout=2)
            if r.status_code == 200:
                logger.info("API server already running ✅")
                return {"status": "success", "message": "already running",
                        "url": "http://127.0.0.1:8080", "elapsed": 0}
        except Exception:
            pass

        # Start server as background process
        proc = subprocess.Popen(
            [sys.executable,
             str(BASE_DIR / "src" / "ingestion" / "api_server.py")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait for server to be ready
        for _ in range(10):
            _time.sleep(1)
            try:
                import requests as _req
                r = _req.get("http://127.0.0.1:8080/api/health", timeout=2)
                if r.status_code == 200:
                    data = r.json()
                    logger.info(
                        f"API server started (pid={proc.pid}) ✅ | "
                        f"Products: {data.get('products_loaded'):,}"
                    )
                    return {
                        "status":  "success",
                        "pid":     proc.pid,
                        "url":     "http://127.0.0.1:8080",
                        "products": data.get("products_loaded", 0),
                        "elapsed": 0,
                    }
            except Exception:
                continue

        logger.warning("API server did not start in time — ingestion will skip API source")
        return {"status": "warning", "message": "server not ready", "elapsed": 0}

    except Exception as e:
        logger.error(f"API server start failed: {e}")
        return {"status": "failed", "error": str(e), "elapsed": 0}


# ── Task 1: Data Ingestion ────────────────────────────────────────────────────

@task(
    name="ingest_data",
    description="Ingest ratings (JSONL file) and products (JSONL file + REST API)",
    retries=2,
    retry_delay_seconds=10,
    tags=["ingestion", "data"],
)
def task_ingest_data() -> dict:
    logger = get_run_logger()
    logger.info("=== Task: Data Ingestion (File + REST API) ===")
    start = time.time()

    try:
        from src.ingestion.ingest_data import run_ingestion
        r_path, p_path, api_path = run_ingestion(include_api=True)
        elapsed = time.time() - start

        result = {
            "status":   "success",
            "ratings":  str(r_path)   if r_path   else None,
            "products": str(p_path)   if p_path   else None,
            "api_products": str(api_path) if api_path else "skipped",
            "elapsed":  round(elapsed, 1),
        }
        logger.info(f"Ingestion complete in {elapsed:.1f}s ✅")
        logger.info(f"  Source 1a JSONL ratings  → {Path(r_path).name if r_path else 'failed'}")
        logger.info(f"  Source 1b JSONL products → {Path(p_path).name if p_path else 'failed'}")
        logger.info(f"  Source 2  REST API       → {Path(api_path).name if api_path else 'skipped'}")
        return result

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "failed", "error": str(e)}


# ── Task 2: Data Validation ───────────────────────────────────────────────────

@task(
    name="validate_data",
    description="Run pandas + Great Expectations validation suite",
    retries=1,
    retry_delay_seconds=5,
    tags=["validation", "quality"],
)
def task_validate_data() -> dict:
    logger = get_run_logger()
    logger.info("=== Task: Data Validation ===")
    start = time.time()

    try:
        from src.validation.validate_data import (
            validate_ratings, validate_products,
            generate_report, run_great_expectations
        )

        r_results, r_df, r_path = validate_ratings()
        p_results, p_df, p_path = validate_products()
        ge_r = run_great_expectations(r_df, "ratings")
        ge_p = run_great_expectations(p_df, "products")
        generate_report(r_results, r_df, r_path, p_results, p_df, p_path)

        pandas_pass = sum(1 for r in r_results + p_results if r["passed"])
        pandas_total = len(r_results) + len(p_results)
        ge_pass  = sum(1 for r in ge_r + ge_p if r["passed"])
        ge_total = len(ge_r) + len(ge_p)

        elapsed = time.time() - start
        result = {
            "status":        "success",
            "pandas_checks": f"{pandas_pass}/{pandas_total}",
            "ge_checks":     f"{ge_pass}/{ge_total}",
            "elapsed":       round(elapsed, 1),
        }
        logger.info(
            f"Validation complete: pandas {pandas_pass}/{pandas_total}, "
            f"GE {ge_pass}/{ge_total} ✅"
        )
        return result

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {"status": "failed", "error": str(e)}


# ── Task 3: Data Preparation ──────────────────────────────────────────────────

@task(
    name="prepare_data",
    description="Clean, deduplicate and prepare datasets for feature engineering",
    tags=["preparation", "cleaning"],
)
def task_prepare_data() -> dict:
    logger = get_run_logger()
    logger.info("=== Task: Data Preparation ===")
    start = time.time()

    try:
        ratings,  r_name = load_latest_csv(INTERACTIONS_DIR, "ratings")
        products, p_name = load_latest_csv(PRODUCTS_DIR,     "products")

        logger.info(f"Loaded: {r_name} ({len(ratings):,} rows)")
        logger.info(f"Loaded: {p_name} ({len(products):,} rows)")

        # Clean ratings
        before = len(ratings)
        ratings = ratings.drop_duplicates()
        ratings = (ratings
                   .sort_values("rating", ascending=False)
                   .drop_duplicates(subset=["userId","productId"])
                   .reset_index(drop=True))
        TS_MIN, TS_MAX = 946684800, 1893456000
        ratings = ratings[
            ratings["timestamp"].between(TS_MIN, TS_MAX)
        ].reset_index(drop=True)
        logger.info(
            f"Ratings cleaned: {before:,} → {len(ratings):,} rows "
            f"({before - len(ratings):,} removed)"
        )

        # Clean products
        products["price"] = pd.to_numeric(
            products["price"], errors="coerce")
        cat_median = products.groupby("category")["price"].transform("median")
        products["price"] = products["price"].fillna(cat_median)
        products["price"] = products["price"].fillna(
            products["price"].median())
        products["brand"]    = products["brand"].fillna("Unknown")
        products["category"] = products["category"].fillna("Unknown")
        products["title"]    = products["title"].fillna("Untitled")
        products["description"] = products["description"].fillna("")

        # Save processed
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        r_out = PROCESSED_DIR / "cleaned_ratings.csv"
        p_out = PROCESSED_DIR / "cleaned_products.csv"
        ratings.to_csv(r_out,  index=False)
        products.to_csv(p_out, index=False)

        elapsed = time.time() - start
        result = {
            "status":         "success",
            "n_ratings":      len(ratings),
            "n_products":     len(products),
            "n_users":        ratings["userId"].nunique(),
            "n_items":        ratings["productId"].nunique(),
            "elapsed":        round(elapsed, 1),
        }
        logger.info(
            f"Preparation complete: {len(ratings):,} interactions, "
            f"{ratings['userId'].nunique():,} users ✅"
        )
        return result

    except Exception as e:
        logger.error(f"Preparation failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "failed", "error": str(e)}


# ── Task 4: Feature Engineering ───────────────────────────────────────────────

@task(
    name="engineer_features",
    description="Build user, item, interaction and co-occurrence features",
    tags=["features", "transformation"],
)
def task_engineer_features() -> dict:
    logger = get_run_logger()
    logger.info("=== Task: Feature Engineering ===")
    start = time.time()

    try:
        from src.features.feature_engineering import run_feature_engineering
        db_path = run_feature_engineering()

        # Verify DB
        conn = sqlite3.connect(db_path)
        counts = {}
        for t in ["user_features","item_features",
                  "interaction_features","cooccurrence"]:
            counts[t] = conn.execute(
                f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        conn.close()

        elapsed = time.time() - start
        result = {
            "status":  "success",
            "db_path": str(db_path),
            "tables":  counts,
            "elapsed": round(elapsed, 1),
        }
        logger.info(
            f"Features built: {counts['user_features']:,} users, "
            f"{counts['item_features']:,} items, "
            f"{counts['interaction_features']:,} interactions ✅"
        )
        return result

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "failed", "error": str(e)}


# ── Task 5: Feature Store Update ──────────────────────────────────────────────

@task(
    name="update_feature_store",
    description="Register new feature version in the feature store",
    tags=["feature_store", "versioning"],
)
def task_update_feature_store(feature_results: dict) -> dict:
    logger = get_run_logger()
    logger.info("=== Task: Feature Store Update ===")
    start = time.time()

    try:
        from src.features.feature_store import register_version
        now_tag = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M")
        vid = register_version(
            version_tag=now_tag,
            notes=f"Automated pipeline run — "
                  f"{feature_results.get('tables',{}).get('interaction_features',0):,} "
                  f"interactions"
        )

        elapsed = time.time() - start
        result  = {
            "status":      "success",
            "version_tag": now_tag,
            "version_id":  vid,
            "elapsed":     round(elapsed, 1),
        }
        logger.info(
            f"Feature store updated: version {now_tag} (id={vid}) ✅"
        )
        return result

    except Exception as e:
        logger.error(f"Feature store update failed: {e}")
        return {"status": "failed", "error": str(e)}


# ── Task 6: Model Training ────────────────────────────────────────────────────

@task(
    name="train_models",
    description="Train SVD collaborative filter and content-based model",
    tags=["model", "training", "mlflow"],
)
def task_train_models() -> dict:
    logger = get_run_logger()
    logger.info("=== Task: Model Training ===")
    start = time.time()

    try:
        from src.model.train_model import run_model_training
        svd_r, cbf_r = run_model_training()

        elapsed = time.time() - start
        result  = {
            "status":       "success",
            "svd_rmse":     svd_r["metrics"]["rmse"],
            "svd_prec_k":   svd_r["metrics"]["precision_at_k"],
            "svd_ndcg_k":   svd_r["metrics"]["ndcg_at_k"],
            "cbf_coverage": cbf_r["metrics"]["coverage_pct"],
            "svd_run_id":   svd_r["run_id"],
            "cbf_run_id":   cbf_r["run_id"],
            "elapsed":      round(elapsed, 1),
        }
        logger.info(
            f"Training complete: SVD RMSE={svd_r['metrics']['rmse']}, "
            f"P@10={svd_r['metrics']['precision_at_k']}, "
            f"NDCG@10={svd_r['metrics']['ndcg_at_k']} ✅"
        )
        return result

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "failed", "error": str(e)}


# ── Task 7: Pipeline Summary ──────────────────────────────────────────────────

@task(
    name="pipeline_summary",
    description="Generate pipeline execution summary report",
    tags=["reporting"],
)
def task_pipeline_summary(
    ingest_r, validate_r, prepare_r,
    feature_r, store_r, model_r,
    pipeline_start: float
) -> dict:
    logger   = get_run_logger()
    elapsed  = time.time() - pipeline_start
    now      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    logger.info("=== Pipeline Summary ===")

    tasks = [
        ("ingest_data",          ingest_r),
        ("validate_data",        validate_r),
        ("prepare_data",         prepare_r),
        ("engineer_features",    feature_r),
        ("update_feature_store", store_r),
        ("train_models",         model_r),
    ]

    all_success = all(r.get("status") == "success" for _, r in tasks)

    # Log summary table
    logger.info(f"\n{'Task':<25} {'Status':<10} {'Time(s)':<10}")
    logger.info("-" * 45)
    for name, r in tasks:
        status  = r.get("status", "unknown")
        elapsed_t = r.get("elapsed", "—")
        logger.info(f"{name:<25} {status:<10} {str(elapsed_t):<10}")

    logger.info(f"\nTotal pipeline time: {elapsed:.1f}s")

    if model_r.get("status") == "success":
        logger.info(f"\nModel Results:")
        logger.info(f"  SVD RMSE         : {model_r.get('svd_rmse')}")
        logger.info(f"  SVD Precision@10 : {model_r.get('svd_prec_k')}")
        logger.info(f"  SVD NDCG@10      : {model_r.get('svd_ndcg_k')}")
        logger.info(f"  CBF Coverage     : {model_r.get('cbf_coverage')}%")
        logger.info(f"  MLflow SVD run   : {model_r.get('svd_run_id')}")

    # Write execution log
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pipeline Execution Log — RecoMart",
        "**Task 10 | DMML Assignment 1 | Group 37**",
        f"Run at: {now}",
        f"Total time: {elapsed:.1f}s",
        f"Status: {'✅ SUCCESS' if all_success else '⚠️ PARTIAL'}",
        "",
        "## Task Results",
        "",
        "| Task | Status | Time (s) | Key Output |",
        "|------|--------|----------|------------|",
        f"| ingest_data | {ingest_r.get('status')} | "
        f"{ingest_r.get('elapsed','—')} | "
        f"ratings + products CSV |",
        f"| validate_data | {validate_r.get('status')} | "
        f"{validate_r.get('elapsed','—')} | "
        f"pandas {validate_r.get('pandas_checks','—')}, "
        f"GE {validate_r.get('ge_checks','—')} |",
        f"| prepare_data | {prepare_r.get('status')} | "
        f"{prepare_r.get('elapsed','—')} | "
        f"{prepare_r.get('n_ratings',0):,} interactions |",
        f"| engineer_features | {feature_r.get('status')} | "
        f"{feature_r.get('elapsed','—')} | "
        f"4 feature tables in SQLite |",
        f"| update_feature_store | {store_r.get('status')} | "
        f"{store_r.get('elapsed','—')} | "
        f"version {store_r.get('version_tag','—')} |",
        f"| train_models | {model_r.get('status')} | "
        f"{model_r.get('elapsed','—')} | "
        f"SVD RMSE={model_r.get('svd_rmse','—')} |",
        "",
        "## Model Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| SVD RMSE | {model_r.get('svd_rmse','—')} |",
        f"| SVD Precision@10 | {model_r.get('svd_prec_k','—')} |",
        f"| SVD NDCG@10 | {model_r.get('svd_ndcg_k','—')} |",
        f"| CBF Coverage | {model_r.get('cbf_coverage','—')}% |",
        f"| MLflow SVD run ID | `{model_r.get('svd_run_id','—')}` |",
        f"| MLflow CBF run ID | `{model_r.get('cbf_run_id','—')}` |",
    ]

    log_path = REPORTS_DIR / "pipeline_execution_log.md"
    log_path.write_text("\n".join(lines))
    logger.info(f"Execution log → {log_path}")

    return {
        "status":       "success" if all_success else "partial",
        "total_elapsed": round(elapsed, 1),
        "all_tasks":    all_success,
        "log_path":     str(log_path),
    }


# ── Main Flow ─────────────────────────────────────────────────────────────────

@flow(
    name="recomart_pipeline",
    description=(
        "End-to-end RecoMart recommendation pipeline: "
        "ingestion → validation → preparation → "
        "feature engineering → feature store → model training"
    ),
    log_prints=True,
)
def recomart_pipeline(
    skip_ingestion:  bool = False,
    skip_training:   bool = False,
) -> dict:
    """
    Full RecoMart pipeline flow.

    Args:
        skip_ingestion : True = skip ingestion (use existing raw files)
        skip_training  : True = skip model training (use existing models)
    """
    logger        = get_run_logger()
    pipeline_start = time.time()
    now           = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    logger.info("=" * 60)
    logger.info("  RecoMart End-to-End Pipeline")
    logger.info(f"  Dataset : {DATASET_NAME}")
    logger.info(f"  Started : {now}")
    logger.info(f"  skip_ingestion={skip_ingestion}, "
                f"skip_training={skip_training}")
    logger.info("=" * 60)

    # ── Stage 0: Start API server ─────────────────────────────────────────────
    if not skip_ingestion:
        api_server_r = task_start_api_server()
        logger.info(f"API server: {api_server_r.get('status')}")
    
    # ── Stage 1: Ingestion ────────────────────────────────────────────────────
    if skip_ingestion:
        logger.info("Skipping ingestion (skip_ingestion=True)")
        ingest_r = {"status": "skipped", "elapsed": 0}
    else:
        ingest_r = task_ingest_data()

    # ── Stage 2: Validation ───────────────────────────────────────────────────
    validate_r = task_validate_data()

    # ── Stage 3: Preparation ──────────────────────────────────────────────────
    prepare_r = task_prepare_data()

    # ── Stage 4: Feature Engineering ─────────────────────────────────────────
    feature_r = task_engineer_features()

    # ── Stage 5: Feature Store ────────────────────────────────────────────────
    store_r = task_update_feature_store(feature_r)

    # ── Stage 6: Model Training ───────────────────────────────────────────────
    if skip_training:
        logger.info("Skipping training (skip_training=True)")
        model_r = {"status": "skipped", "elapsed": 0}
    else:
        model_r = task_train_models()

    # ── Stage 7: Summary ──────────────────────────────────────────────────────
    summary = task_pipeline_summary(
        ingest_r, validate_r, prepare_r,
        feature_r, store_r, model_r,
        pipeline_start,
    )

    logger.info("=" * 60)
    logger.info(f"  Pipeline {summary['status'].upper()}")
    logger.info(f"  Total time: {summary['total_elapsed']}s")
    logger.info("=" * 60)

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="RecoMart End-to-End Pipeline (Prefect)")
    parser.add_argument(
        "--skip-ingestion", action="store_true",
        help="Skip ingestion, use existing raw files")
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip model training, use existing models")
    args = parser.parse_args()

    result = recomart_pipeline(
        skip_ingestion = args.skip_ingestion,
        skip_training  = args.skip_training,
    )
    print(f"\nPipeline status : {result['status']}")
    print(f"Total time      : {result['total_elapsed']}s")
    print(f"Log             : {result['log_path']}")
