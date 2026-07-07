"""
Task 7: Feature Store — RecoMart Pipeline
Dataset: Amazon Reviews 2023 — Video Games

Implements a lightweight feature store using:
  - SQLite backend (recomart_features.db) for feature data
  - JSON metadata registry (feature_metadata.json) for documentation
  - Version table in SQLite for tracking feature versions
  - Simple retrieval API for both training and inference

Feature Store capabilities:
  1. register_version()  — log a new feature version with metadata
  2. get_user_features() — retrieve features for one or many users
  3. get_item_features() — retrieve features for one or many items
  4. get_training_set()  — join user + item + interaction features for model training
  5. get_inference_features() — retrieve features for a user at inference time
  6. list_versions()     — show all registered versions
  7. get_feature_stats() — summary statistics for any feature table
"""

import sys
import json
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FEATURE_STORE, REPORTS_DIR, DATASET_NAME
from src.logger import get_logger

log = get_logger("feature_store")

DB_PATH       = FEATURE_STORE / "recomart_features.db"
METADATA_PATH = FEATURE_STORE / "feature_metadata.json"


# ── Version management ────────────────────────────────────────────────────────

def _ensure_version_table(conn: sqlite3.Connection):
    """Create version tracking table if not exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_versions (
            version_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            version_tag  TEXT NOT NULL,
            feature_set  TEXT NOT NULL,
            n_users      INTEGER,
            n_items      INTEGER,
            n_interactions INTEGER,
            dataset      TEXT,
            created_at   TEXT,
            notes        TEXT
        )
    """)
    conn.commit()


def register_version(version_tag: str,
                     notes: str = "",
                     conn: sqlite3.Connection = None) -> int:
    """
    Register a new feature version in the version table.
    Called automatically after feature_engineering.py runs.

    Args:
        version_tag : human-readable tag e.g. "v1.0", "v1.1-resampled"
        notes       : optional description of what changed
    Returns:
        version_id  : auto-assigned integer ID
    """
    close = False
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        close = True

    _ensure_version_table(conn)

    n_users = conn.execute(
        "SELECT COUNT(*) FROM user_features").fetchone()[0]
    n_items = conn.execute(
        "SELECT COUNT(*) FROM item_features").fetchone()[0]
    n_inter = conn.execute(
        "SELECT COUNT(*) FROM interaction_features").fetchone()[0]

    cur = conn.execute("""
        INSERT INTO feature_versions
            (version_tag, feature_set, n_users, n_items,
             n_interactions, dataset, created_at, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        version_tag,
        "user_features,item_features,interaction_features,cooccurrence",
        n_users, n_items, n_inter,
        DATASET_NAME,
        datetime.now(timezone.utc).isoformat(),
        notes,
    ))
    conn.commit()
    vid = cur.lastrowid
    log.info(f"[FeatureStore] Version registered: {version_tag} "
             f"(id={vid}) — {n_users:,} users, {n_items:,} items, "
             f"{n_inter:,} interactions")
    if close:
        conn.close()
    return vid


def list_versions() -> pd.DataFrame:
    """List all registered feature versions."""
    conn = sqlite3.connect(DB_PATH)
    _ensure_version_table(conn)
    df = pd.read_sql(
        "SELECT * FROM feature_versions ORDER BY version_id DESC", conn)
    conn.close()
    return df


# ── Feature retrieval API ─────────────────────────────────────────────────────

def get_user_features(user_ids: list | None = None,
                      limit: int | None = None) -> pd.DataFrame:
    """
    Retrieve user features for training or inference.

    Args:
        user_ids : list of userId strings. None = all users.
        limit    : max rows to return (for sampling)
    Returns:
        DataFrame with user feature columns
    """
    conn  = sqlite3.connect(DB_PATH)
    if user_ids is not None:
        placeholders = ",".join("?" * len(user_ids))
        query = f"""
            SELECT * FROM user_features
            WHERE userId IN ({placeholders})
        """
        df = pd.read_sql(query, conn, params=user_ids)
    else:
        limit_clause = f"LIMIT {limit}" if limit else ""
        df = pd.read_sql(
            f"SELECT * FROM user_features {limit_clause}", conn)
    conn.close()
    log.info(f"[FeatureStore] get_user_features → {len(df):,} rows")
    return df


def get_item_features(product_ids: list | None = None,
                      limit: int | None = None) -> pd.DataFrame:
    """
    Retrieve item features for training or inference.

    Args:
        product_ids : list of productId strings. None = all items.
        limit       : max rows to return
    Returns:
        DataFrame with item feature columns
    """
    conn = sqlite3.connect(DB_PATH)
    if product_ids is not None:
        placeholders = ",".join("?" * len(product_ids))
        query = f"""
            SELECT * FROM item_features
            WHERE productId IN ({placeholders})
        """
        df = pd.read_sql(query, conn, params=product_ids)
    else:
        limit_clause = f"LIMIT {limit}" if limit else ""
        df = pd.read_sql(
            f"SELECT * FROM item_features {limit_clause}", conn)
    conn.close()
    log.info(f"[FeatureStore] get_item_features → {len(df):,} rows")
    return df


def get_training_set(sample_frac: float = 1.0,
                     random_state: int = 42) -> pd.DataFrame:
    """
    Retrieve joined training set:
      interaction_features
        LEFT JOIN user_features  ON userId
        LEFT JOIN item_features  ON productId

    Args:
        sample_frac  : fraction of interactions to sample (1.0 = all)
        random_state : reproducibility seed
    Returns:
        DataFrame with all features joined — ready for model training
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT
            i.userId,
            i.productId,
            i.rating,
            i.rating_normalized,
            i.days_since_epoch,
            i.user_item_rating_diff,
            u.user_rating_count,
            u.user_avg_rating,
            u.user_rating_std,
            u.user_unique_items,
            u.user_recency_days,
            u.user_category_diversity,
            p.item_rating_count,
            p.item_avg_rating,
            p.item_rating_std,
            p.item_unique_users,
            p.price_normalized,
            p.category_encoded,
            p.brand_encoded
        FROM interaction_features i
        LEFT JOIN user_features u ON i.userId    = u.userId
        LEFT JOIN item_features p ON i.productId = p.productId
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)
        df = df.reset_index(drop=True)

    log.info(f"[FeatureStore] get_training_set → {len(df):,} rows × "
             f"{len(df.columns)} features")
    return df


def get_inference_features(user_id: str,
                           candidate_items: list | None = None) -> pd.DataFrame:
    """
    Retrieve features for a single user at inference time.
    Joins user features with candidate item features.

    Args:
        user_id         : userId string for the target user
        candidate_items : list of productId strings to score.
                          None = top 100 popular items.
    Returns:
        DataFrame ready for model.predict() — one row per candidate item
    """
    conn = sqlite3.connect(DB_PATH)

    # Get user features
    user_f = pd.read_sql(
        "SELECT * FROM user_features WHERE userId = ?",
        conn, params=[user_id]
    )
    if user_f.empty:
        log.warning(f"[FeatureStore] User '{user_id}' not found")
        conn.close()
        return pd.DataFrame()

    # Get candidate items
    if candidate_items is None:
        item_f = pd.read_sql("""
            SELECT * FROM item_features
            ORDER BY item_rating_count DESC
            LIMIT 100
        """, conn)
    else:
        placeholders = ",".join("?" * len(candidate_items))
        item_f = pd.read_sql(
            f"SELECT * FROM item_features WHERE productId IN ({placeholders})",
            conn, params=candidate_items
        )

    conn.close()

    # Cross join user × items
    user_f["_key"] = 1
    item_f["_key"] = 1
    merged = user_f.merge(item_f, on="_key").drop(columns=["_key"])

    log.info(f"[FeatureStore] get_inference_features → "
             f"user='{user_id}' × {len(item_f)} items = {len(merged)} rows")
    return merged


def get_cooccurrence(product_id: str, top_n: int = 5) -> pd.DataFrame:
    """
    Retrieve top-N co-occurring items for a given product.
    Used for item-based recommendations ('customers also bought').

    Args:
        product_id : source productId
        top_n      : number of co-occurring items to return
    Returns:
        DataFrame with co_productId and cooc_score
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT co_productId, cooc_score
        FROM cooccurrence
        WHERE productId = ?
        ORDER BY cooc_score DESC
        LIMIT ?
    """, conn, params=[product_id, top_n])
    conn.close()
    log.info(f"[FeatureStore] get_cooccurrence('{product_id}') → "
             f"{len(df)} items")
    return df


def get_feature_stats(table: str = "user_features") -> pd.DataFrame:
    """
    Return descriptive statistics for any feature table.

    Args:
        table : one of user_features, item_features,
                interaction_features, cooccurrence
    Returns:
        DataFrame with count, mean, std, min, max per column
    """
    valid = {"user_features", "item_features",
             "interaction_features", "cooccurrence"}
    if table not in valid:
        raise ValueError(f"table must be one of {valid}")
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    stats = df.describe().round(4)
    log.info(f"[FeatureStore] get_feature_stats('{table}') — "
             f"{len(df):,} rows")
    return stats


# ── Metadata reader ───────────────────────────────────────────────────────────

def get_feature_metadata(feature_set: str | None = None) -> dict:
    """
    Load feature metadata from JSON registry.

    Args:
        feature_set : one of 'user_features', 'item_features',
                      'interaction_features', 'cooccurrence'.
                      None = full metadata dict.
    Returns:
        dict with feature definitions, encoders, source info
    """
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    if feature_set:
        return meta["features"].get(feature_set, {})
    return meta


# ── Demo: versioned retrieval ─────────────────────────────────────────────────

def run_feature_store_demo():
    """
    Demonstrate feature store capabilities:
    - Register version
    - Retrieve user/item features
    - Build training set
    - Inference retrieval
    - Co-occurrence lookup
    """
    log.info("========== Feature Store Demo START ==========")

    # ── 1. Register version ───────────────────────────────────────────────────
    log.info("\n[1] Registering feature version...")
    vid = register_version(
        version_tag="v1.0",
        notes="Initial feature set — Amazon Video Games 2023, 10K users sampled"
    )

    # ── 2. List versions ──────────────────────────────────────────────────────
    log.info("\n[2] Listing all versions...")
    versions = list_versions()
    print("\nRegistered versions:")
    print(versions.to_string(index=False))

    # ── 3. User feature retrieval ─────────────────────────────────────────────
    log.info("\n[3] Retrieving sample user features...")
    users = get_user_features(limit=5)
    print("\nSample user features:")
    print(users.to_string(index=False))

    # ── 4. Item feature retrieval ─────────────────────────────────────────────
    log.info("\n[4] Retrieving sample item features...")
    items = get_item_features(limit=3)
    print("\nSample item features:")
    print(items[["productId","item_rating_count","item_avg_rating",
                 "price_normalized","category_encoded"]].to_string(index=False))

    # ── 5. Training set ───────────────────────────────────────────────────────
    log.info("\n[5] Building training set (10% sample)...")
    train = get_training_set(sample_frac=0.1)
    print(f"\nTraining set: {train.shape}")
    print(f"Columns: {list(train.columns)}")
    print(f"Missing values: {train.isnull().sum().sum()}")

    # ── 6. Inference features ─────────────────────────────────────────────────
    log.info("\n[6] Inference feature retrieval...")
    conn   = sqlite3.connect(DB_PATH)
    sample_user = conn.execute(
        "SELECT userId FROM user_features LIMIT 1").fetchone()[0]
    conn.close()
    inf_df = get_inference_features(sample_user, candidate_items=None)
    print(f"\nInference features for user '{sample_user[:20]}...':")
    print(f"Shape: {inf_df.shape} (user × 100 candidate items)")

    # ── 7. Co-occurrence ──────────────────────────────────────────────────────
    log.info("\n[7] Co-occurrence lookup...")
    conn       = sqlite3.connect(DB_PATH)
    sample_item = conn.execute(
        "SELECT productId FROM cooccurrence LIMIT 1").fetchone()[0]
    conn.close()
    cooc = get_cooccurrence(sample_item, top_n=5)
    print(f"\nTop-5 co-occurring items for '{sample_item}':")
    print(cooc.to_string(index=False))

    # ── 8. Feature stats ──────────────────────────────────────────────────────
    log.info("\n[8] Feature statistics...")
    stats = get_feature_stats("user_features")
    print("\nUser feature statistics:")
    print(stats.to_string())

    # ── 9. Metadata ───────────────────────────────────────────────────────────
    log.info("\n[9] Feature metadata (user_features)...")
    meta = get_feature_metadata("user_features")
    print(f"\nFeature set: user_features")
    print(f"Source      : {meta.get('source')}")
    print(f"n_records   : {meta.get('n_records'):,}")
    print(f"n_features  : {meta.get('n_features')}")
    print("Columns:")
    for col, info in meta.get("columns", {}).items():
        print(f"  {col:30s} {info['type']:6s} — {info['description']}")

    # ── Write demo report ─────────────────────────────────────────────────────
    _write_feature_store_doc(versions, users, items, train, inf_df, cooc)

    log.info("\n========== Feature Store Demo COMPLETE ==========")
    return True


def _write_feature_store_doc(versions, users, items, train, inf_df, cooc):
    """Generate feature store documentation markdown."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Feature Store Documentation — RecoMart Pipeline",
        "**Task 7 | DMML Assignment 1 | Group 37**",
        f"Generated: {now}",
        "",
        "---",
        "",
        "## Overview",
        "",
        "The RecoMart feature store is implemented using:",
        "- **SQLite** backend (`recomart_features.db`) for feature data storage",
        "- **JSON metadata registry** (`feature_metadata.json`) for feature documentation",
        "- **Version table** in SQLite for tracking feature versions",
        "- **Python API** (`feature_store.py`) for retrieval at training and inference time",
        "",
        "## Feature Store API",
        "",
        "| Function | Purpose | Use case |",
        "|----------|---------|----------|",
        "| `register_version()` | Log feature version with metadata | After each pipeline run |",
        "| `list_versions()` | Show all registered versions | Audit, reproducibility |",
        "| `get_user_features()` | Retrieve user features by ID | Training + inference |",
        "| `get_item_features()` | Retrieve item features by ID | Training + inference |",
        "| `get_training_set()` | Joined features for model training | Task 9 model training |",
        "| `get_inference_features()` | User × candidate items for scoring | Real-time recommendation |",
        "| `get_cooccurrence()` | Top-N co-occurring items | Item-based recommendation |",
        "| `get_feature_stats()` | Descriptive stats for any table | Monitoring, debugging |",
        "| `get_feature_metadata()` | Feature definitions from registry | Documentation, lineage |",
        "",
        "## Registered Versions",
        "",
        versions.to_markdown(index=False),
        "",
        "## Training Set Schema",
        "",
        f"Shape: {train.shape[0]:,} rows × {train.shape[1]} features",
        "",
        "| Column | Source | Description |",
        "|--------|--------|-------------|",
        "| userId | interaction | Amazon reviewer ID |",
        "| productId | interaction | Amazon ASIN |",
        "| rating | interaction | Raw star rating (1–5) |",
        "| rating_normalized | interaction | Rating scaled to [0,1] |",
        "| days_since_epoch | interaction | Timestamp in days |",
        "| user_item_rating_diff | interaction | User rating minus item avg |",
        "| user_rating_count | user | Total ratings given |",
        "| user_avg_rating | user | Mean rating given |",
        "| user_rating_std | user | Rating consistency |",
        "| user_unique_items | user | Distinct products rated |",
        "| user_recency_days | user | Days since last rating |",
        "| user_category_diversity | user | Categories explored |",
        "| item_rating_count | item | Total ratings received |",
        "| item_avg_rating | item | Mean rating received |",
        "| item_rating_std | item | Rating controversy score |",
        "| item_unique_users | item | Reach |",
        "| price_normalized | item | MinMaxScaled price [0,1] |",
        "| category_encoded | item | LabelEncoded category |",
        "| brand_encoded | item | LabelEncoded brand |",
        "",
        "## Inference Flow",
        "",
        "```",
        "user_id → get_inference_features(user_id)",
        "        → joins user_features × top-100 candidate items",
        "        → returns DataFrame ready for model.predict()",
        "        → rank by predicted_rating → top-K recommendations",
        "```",
        "",
        f"Sample inference shape: {inf_df.shape} "
        f"(1 user × {len(inf_df)} candidate items × {inf_df.shape[1]} features)",
        "",
        "## Co-occurrence Sample",
        "",
        cooc.to_markdown(index=False),
        "",
        "## Versioning Strategy",
        "",
        "Each pipeline run registers a new version with:",
        "- Unique version tag (e.g. v1.0, v1.1)",
        "- Row counts for all feature tables",
        "- Dataset name and creation timestamp",
        "- Human-readable notes describing changes",
        "",
        "This enables reproducible training — any historical version",
        "can be reconstructed by re-running the pipeline with the",
        "same source data snapshot (tracked via DVC in Task 8).",
    ]

    path = REPORTS_DIR / "feature_store_documentation.md"
    path.write_text("\n".join(lines))
    log.info(f"Feature store docs → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found.")
        print("Run src/features/feature_engineering.py first.")
    else:
        run_feature_store_demo()
