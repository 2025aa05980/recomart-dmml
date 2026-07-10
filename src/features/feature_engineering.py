"""
Task 6: Feature Engineering & Transformation — RecoMart Pipeline
Dataset: Amazon Reviews 2023 — Video Games

Features created:
  User-level:
    user_rating_count      — total ratings given (activity level)
    user_avg_rating        — mean rating given (preference tendency)
    user_rating_std        — std dev of ratings (rating consistency)
    user_unique_items      — distinct products rated
    user_recency_days      — days since last rating (engagement recency)
    user_category_diversity— number of distinct categories rated

  Item-level:
    item_rating_count      — total ratings received (popularity)
    item_avg_rating        — mean rating received (quality signal)
    item_rating_std        — std dev of ratings (controversy score)
    item_unique_users      — distinct users who rated
    price_normalized       — MinMaxScaler on price [0,1]
    category_encoded       — LabelEncoder on category string
    brand_encoded          — LabelEncoder on brand string

  Interaction-level:
    rating_normalized      — rating scaled to [0,1]
    user_item_rating_diff  — rating minus item avg (user sentiment relative to crowd)
    days_since_epoch       — timestamp → days (normalized temporal feature)

Storage:
  feature_store/recomart_features.db — SQLite database (3 tables)
  feature_store/feature_metadata.json — feature registry

SQL Schema:
  user_features        (userId PK, + 6 features)
  item_features        (productId PK, + 7 features)
  interaction_features (userId+productId PK, + 4 features)
"""

import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PROCESSED_DIR, FEATURE_STORE, REPORTS_DIR, RANDOM_SEED
from src.logger import get_logger

log = get_logger("feature_engineering")

DB_PATH       = FEATURE_STORE / "recomart_features.db"
METADATA_PATH = FEATURE_STORE / "feature_metadata.json"


# ── Data loader ───────────────────────────────────────────────────────────────

def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    r_path = PROCESSED_DIR / "cleaned_ratings.csv"
    p_path = PROCESSED_DIR / "cleaned_products.csv"
    if not r_path.exists():
        raise FileNotFoundError(
            f"cleaned_ratings.csv not found in {PROCESSED_DIR}\n"
            f"Run the EDA notebook (Task 5) first."
        )
    ratings  = pd.read_csv(r_path)
    products = pd.read_csv(p_path)
    log.info(f"Loaded ratings  : {ratings.shape}")
    log.info(f"Loaded products : {products.shape}")
    return ratings, products


# ── User features ─────────────────────────────────────────────────────────────

def build_user_features(ratings: pd.DataFrame,
                        products: pd.DataFrame) -> pd.DataFrame:
    log.info("Building user features...")

    merged = ratings.merge(
        products[["productId", "category"]], on="productId", how="left"
    )

    uf = ratings.groupby("userId").agg(
        user_rating_count  = ("rating",    "count"),
        user_avg_rating    = ("rating",    "mean"),
        user_rating_std    = ("rating",    "std"),
        user_unique_items  = ("productId", "nunique"),
        last_timestamp     = ("timestamp", "max"),
        first_timestamp    = ("timestamp", "min"),
    ).reset_index()

    # Recency: days since last rating (relative to dataset max)
    max_ts = ratings["timestamp"].max()
    uf["user_recency_days"] = ((max_ts - uf["last_timestamp"]) / 86400).round(1)

    # Category diversity: number of distinct categories rated per user
    cat_div = (merged.groupby("userId")["category"]
               .nunique()
               .reset_index()
               .rename(columns={"category": "user_category_diversity"}))
    uf = uf.merge(cat_div, on="userId", how="left")

    # Fill NaN std (users with 1 rating)
    uf["user_rating_std"] = uf["user_rating_std"].fillna(0.0)

    # Drop helper columns
    uf.drop(columns=["last_timestamp", "first_timestamp"], inplace=True)

    uf = uf.round(4)
    log.info(f"User features   : {uf.shape} — {list(uf.columns)}")
    return uf


# ── Item features ─────────────────────────────────────────────────────────────

def build_item_features(ratings: pd.DataFrame,
                        products: pd.DataFrame) -> pd.DataFrame:
    log.info("Building item features...")

    # ── Computed from sampled ratings (10K users) ─────────────────────────────
    itf = ratings.groupby("productId").agg(
        item_rating_count = ("rating",  "count"),
        item_avg_rating   = ("rating",  "mean"),
        item_rating_std   = ("rating",  "std"),
        item_unique_users = ("userId",  "nunique"),
    ).reset_index()

    itf["item_rating_std"] = itf["item_rating_std"].fillna(0.0)

    # ── Merge product metadata ────────────────────────────────────────────────
    # Select base columns + API-enriched columns if available
    meta_cols = ["productId", "price", "category", "brand"]

    # Add API-enriched columns if present in cleaned_products.csv
    api_cols = []
    if "rating_count" in products.columns:
        meta_cols.append("rating_count")
        api_cols.append("rating_count")
    if "avg_rating" in products.columns:
        meta_cols.append("avg_rating")
        api_cols.append("avg_rating")

    itf = itf.merge(
        products[meta_cols],
        on="productId", how="left"
    )

    # Rename API columns to distinguish from sampled computed columns
    if "rating_count" in itf.columns:
        itf.rename(columns={
            "rating_count": "api_rating_count",
            "avg_rating":   "api_avg_rating",
        }, inplace=True)
        itf["api_rating_count"] = itf["api_rating_count"].fillna(0).astype(int)
        itf["api_avg_rating"]   = itf["api_avg_rating"].fillna(0.0).round(4)
        log.info(f"API enrichment fields added: api_rating_count, api_avg_rating")
        log.info(f"  Products with API ratings: "
                 f"{(itf['api_rating_count'] > 0).sum():,}")
    else:
        log.info("API enrichment fields not available — "
                 "run api_server.py + ingest_data.py to enable")

    # ── Encoders ──────────────────────────────────────────────────────────────
    # LabelEncoder — category
    le_cat = LabelEncoder()
    itf["category"] = itf["category"].fillna("Unknown")
    itf["category_encoded"] = le_cat.fit_transform(itf["category"])

    # LabelEncoder — brand
    le_brand = LabelEncoder()
    itf["brand"] = itf["brand"].fillna("Unknown")
    itf["brand_encoded"] = le_brand.fit_transform(itf["brand"])

    # MinMaxScaler — price
    itf["price"] = pd.to_numeric(itf["price"], errors="coerce").fillna(
        itf["price"].median() if "price" in itf else 29.99
    )
    scaler = MinMaxScaler()
    itf["price_normalized"] = scaler.fit_transform(
        itf[["price"]]
    ).round(4)

    # Save encoders metadata for inference reuse
    cat_classes   = list(le_cat.classes_)
    brand_classes = list(le_brand.classes_)

    itf = itf.round(4)
    log.info(f"Item features   : {itf.shape} — {list(itf.columns)}")
    return itf, cat_classes, brand_classes


# ── Interaction features ──────────────────────────────────────────────────────

def build_interaction_features(ratings: pd.DataFrame,
                               item_features: pd.DataFrame) -> pd.DataFrame:
    log.info("Building interaction features...")

    inf = ratings[["userId", "productId", "rating", "timestamp"]].copy()

    # rating_normalized: scale [1,5] → [0,1]
    inf["rating_normalized"] = ((inf["rating"] - 1) / 4).round(4)

    # days_since_epoch: timestamp → days
    inf["days_since_epoch"] = (inf["timestamp"] / 86400).astype(int)

    # user_item_rating_diff: user rating minus item average
    item_avg = item_features[["productId", "item_avg_rating"]]
    inf = inf.merge(item_avg, on="productId", how="left")
    inf["user_item_rating_diff"] = (
        inf["rating"] - inf["item_avg_rating"]
    ).round(4)
    inf.drop(columns=["item_avg_rating"], inplace=True)

    log.info(f"Interaction features: {inf.shape} — {list(inf.columns)}")
    return inf


# ── Co-occurrence features ────────────────────────────────────────────────────

def build_cooccurrence(ratings: pd.DataFrame,
                       top_n: int = 500) -> pd.DataFrame:
    """
    Compute item-item co-occurrence scores.
    Two items co-occur if the same user rated both.
    Limited to top_n items by popularity for tractability.
    """
    log.info(f"Building co-occurrence matrix (top {top_n} items)...")

    top_items = (ratings["productId"]
                 .value_counts()
                 .head(top_n)
                 .index.tolist())
    sub = ratings[ratings["productId"].isin(top_items)]

    # Pivot to user-item matrix
    pivot = (sub.pivot_table(index="userId", columns="productId",
                             values="rating", aggfunc="mean")
             .fillna(0))

    # Co-occurrence = items rated by same user (dot product of binary matrix)
    binary = (pivot > 0).astype(float)
    cooc   = binary.T.dot(binary)

    # Convert to long format (top 3 co-occurring items per item)
    records = []
    for item in cooc.index:
        row = cooc[item].drop(index=item).nlargest(3)
        for co_item, score in row.items():
            if score > 0:
                records.append({
                    "productId":    item,
                    "co_productId": co_item,
                    "cooc_score":   round(float(score), 2),
                })

    df_cooc = pd.DataFrame(records)
    log.info(f"Co-occurrence pairs: {len(df_cooc):,}")
    return df_cooc


# ── SQLite storage ────────────────────────────────────────────────────────────

def store_to_sqlite(user_f: pd.DataFrame,
                    item_f: pd.DataFrame,
                    inter_f: pd.DataFrame,
                    cooc_f: pd.DataFrame) -> Path:
    """
    Store all feature tables to SQLite database.
    Schema:
      user_features        — userId PK
      item_features        — productId PK
      interaction_features — userId + productId composite PK
      cooccurrence         — productId + co_productId
    """
    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # ── Create tables with explicit schema ────────────────────────────────────
    cur.executescript("""
    DROP TABLE IF EXISTS user_features;
    CREATE TABLE user_features (
        userId                  TEXT PRIMARY KEY,
        user_rating_count       INTEGER,
        user_avg_rating         REAL,
        user_rating_std         REAL,
        user_unique_items       INTEGER,
        user_recency_days       REAL,
        user_category_diversity INTEGER
    );

    DROP TABLE IF EXISTS item_features;
    CREATE TABLE item_features (
        productId           TEXT PRIMARY KEY,
        item_rating_count   INTEGER,
        item_avg_rating     REAL,
        item_rating_std     REAL,
        item_unique_users   INTEGER,
        price               REAL,
        category            TEXT,
        brand               TEXT,
        category_encoded    INTEGER,
        brand_encoded       INTEGER,
        price_normalized    REAL,
        api_rating_count    INTEGER,
        api_avg_rating      REAL
    );

    DROP TABLE IF EXISTS interaction_features;
    CREATE TABLE interaction_features (
        userId                  TEXT,
        productId               TEXT,
        rating                  REAL,
        timestamp               INTEGER,
        rating_normalized       REAL,
        days_since_epoch        INTEGER,
        user_item_rating_diff   REAL,
        PRIMARY KEY (userId, productId)
    );

    DROP TABLE IF EXISTS cooccurrence;
    CREATE TABLE cooccurrence (
        productId       TEXT,
        co_productId    TEXT,
        cooc_score      REAL,
        PRIMARY KEY (productId, co_productId)
    );
    """)

    # ── Insert data ───────────────────────────────────────────────────────────
    cols_u = ["userId","user_rating_count","user_avg_rating","user_rating_std",
              "user_unique_items","user_recency_days","user_category_diversity"]
    user_f[cols_u].to_sql("user_features", conn,
                          if_exists="replace", index=False)

    cols_i = ["productId","item_rating_count","item_avg_rating","item_rating_std",
              "item_unique_users","price","category","brand",
              "category_encoded","brand_encoded","price_normalized",
              "api_rating_count","api_avg_rating"]
    item_f[[c for c in cols_i if c in item_f.columns]].to_sql(
        "item_features", conn, if_exists="replace", index=False)

    cols_n = ["userId","productId","rating","timestamp",
              "rating_normalized","days_since_epoch","user_item_rating_diff"]
    inter_f[cols_n].to_sql("interaction_features", conn,
                           if_exists="replace", index=False)

    cooc_f.to_sql("cooccurrence", conn, if_exists="replace", index=False)

    conn.commit()

    # ── Verify row counts ─────────────────────────────────────────────────────
    for table in ["user_features","item_features",
                  "interaction_features","cooccurrence"]:
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        log.info(f"[DB] {table}: {count:,} rows")

    conn.close()
    log.info(f"SQLite DB saved → {DB_PATH}")
    return DB_PATH


# ── Feature metadata registry ─────────────────────────────────────────────────

def write_metadata(cat_classes: list, brand_classes: list,
                   user_f: pd.DataFrame, item_f: pd.DataFrame,
                   inter_f: pd.DataFrame):
    """Write feature metadata JSON for feature store (Task 7)."""
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "version":    "1.0",
        "created_at": now,
        "dataset":    "Amazon Reviews 2023 — Video Games",
        "features": {
            "user_features": {
                "source":    "cleaned_ratings.csv",
                "n_features": len(user_f.columns) - 1,
                "n_records":  len(user_f),
                "columns": {
                    "userId":                   {"type": "str",   "description": "Amazon reviewer ID (PK)"},
                    "user_rating_count":         {"type": "int",   "description": "Total ratings given by user"},
                    "user_avg_rating":           {"type": "float", "description": "Mean rating given by user (1–5)"},
                    "user_rating_std":           {"type": "float", "description": "Std dev of user ratings (consistency)"},
                    "user_unique_items":         {"type": "int",   "description": "Distinct products rated"},
                    "user_recency_days":         {"type": "float", "description": "Days since last rating (lower = more recent)"},
                    "user_category_diversity":   {"type": "int",   "description": "Number of distinct categories rated"},
                },
            },
            "item_features": {
                "source":     "cleaned_ratings.csv + cleaned_products.csv",
                "n_features":  len(item_f.columns) - 1,
                "n_records":   len(item_f),
                "columns": {
                    "productId":          {"type": "str",   "description": "Amazon ASIN (PK)"},
                    "item_rating_count":  {"type": "int",   "description": "Total ratings received"},
                    "item_avg_rating":    {"type": "float", "description": "Mean rating received (quality signal)"},
                    "item_rating_std":    {"type": "float", "description": "Std dev of ratings (controversy score)"},
                    "item_unique_users":  {"type": "int",   "description": "Distinct users who rated"},
                    "price_normalized":   {"type": "float", "description": "MinMaxScaler price in [0,1]", "scaler": "MinMaxScaler"},
                    "category_encoded":   {"type": "int",   "description": "LabelEncoder category index", "encoder": "LabelEncoder", "n_classes": len(cat_classes)},
                    "brand_encoded":      {"type": "int",   "description": "LabelEncoder brand index",    "encoder": "LabelEncoder", "n_classes": len(brand_classes)},
                },
                "encoders": {
                    "category": {"type": "LabelEncoder", "n_classes": len(cat_classes),   "sample_classes": cat_classes[:5]},
                    "brand":    {"type": "LabelEncoder", "n_classes": len(brand_classes), "sample_classes": brand_classes[:5]},
                    "price":    {"type": "MinMaxScaler",  "feature_range": [0, 1]},
                },
            },
            "interaction_features": {
                "source":    "cleaned_ratings.csv + item_features",
                "n_features": len(inter_f.columns) - 2,
                "n_records":  len(inter_f),
                "columns": {
                    "userId":                 {"type": "str",   "description": "Amazon reviewer ID (FK)"},
                    "productId":              {"type": "str",   "description": "Amazon ASIN (FK)"},
                    "rating":                 {"type": "float", "description": "Raw star rating (1–5)"},
                    "rating_normalized":      {"type": "float", "description": "Rating scaled to [0,1]"},
                    "days_since_epoch":       {"type": "int",   "description": "Timestamp in days (temporal feature)"},
                    "user_item_rating_diff":  {"type": "float", "description": "User rating minus item avg (relative sentiment)"},
                },
            },
            "cooccurrence": {
                "source":      "cleaned_ratings.csv",
                "description": "Item-item co-occurrence scores from user interaction overlap",
                "top_n_items": 500,
                "columns": {
                    "productId":    {"type": "str",   "description": "Source item ASIN"},
                    "co_productId": {"type": "str",   "description": "Co-occurring item ASIN"},
                    "cooc_score":   {"type": "float", "description": "Number of shared users who rated both"},
                },
            },
        },
    }
    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Feature metadata → {METADATA_PATH}")


# ── Summary report ────────────────────────────────────────────────────────────

def write_feature_summary(user_f, item_f, inter_f, cooc_f):
    """Write feature engineering summary markdown for PDF report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Feature Engineering Summary — RecoMart Pipeline",
        "**Task 6 | DMML Assignment 1 | Group 37**",
        f"Generated: {now}",
        "",
        "---",
        "",
        "## Feature Tables",
        "",
        f"| Table | Rows | Features | Storage |",
        f"|-------|------|----------|---------|",
        f"| user_features | {len(user_f):,} | {len(user_f.columns)-1} | SQLite |",
        f"| item_features | {len(item_f):,} | {len(item_f.columns)-1} | SQLite |",
        f"| interaction_features | {len(inter_f):,} | {len(inter_f.columns)-2} | SQLite |",
        f"| cooccurrence | {len(cooc_f):,} | 3 | SQLite |",
        "",
        "## User Features",
        "",
        "| Feature | Logic | Rationale |",
        "|---------|-------|-----------|",
        "| user_rating_count | COUNT(ratings) per user | Activity level — power vs casual users |",
        "| user_avg_rating | MEAN(rating) per user | Preference tendency — optimist vs critic |",
        "| user_rating_std | STD(rating) per user | Consistency — polarized vs moderate rater |",
        "| user_unique_items | NUNIQUE(productId) per user | Breadth of engagement |",
        "| user_recency_days | (max_ts - last_ts) / 86400 | Engagement recency — lower = more active |",
        "| user_category_diversity | NUNIQUE(category) per user | Cross-category interest |",
        "",
        "## Item Features",
        "",
        "| Feature | Source | Logic | Rationale |",
        "|---------|--------|-------|-----------|",
        "| item_rating_count | ratings (sampled) | COUNT(ratings) per item | Popularity from 10K user sample |",
        "| item_avg_rating | ratings (sampled) | MEAN(rating) per item | Quality signal from sampled users |",
        "| item_rating_std | ratings (sampled) | STD(rating) per item | Controversy score — polarising items |",
        "| item_unique_users | ratings (sampled) | NUNIQUE(userId) per item | Reach from sampled users |",
        "| price_normalized | products (JSONL) | MinMaxScaler(price) → [0,1] | Normalized for model input |",
        "| category_encoded | products (JSONL) | LabelEncoder(category) | Categorical → numeric for ML |",
        "| brand_encoded | products (JSONL) | LabelEncoder(brand) | Categorical → numeric for ML |",
        "| api_rating_count | REST API (Source 2) | rating_count from API | Full catalog popularity (all Amazon users) |",
        "| api_avg_rating | REST API (Source 2) | avg_rating from API | Full catalog quality signal (all Amazon users) |",
        "",
        "## Interaction Features",
        "",
        "| Feature | Logic | Rationale |",
        "|---------|-------|-----------|",
        "| rating_normalized | (rating - 1) / 4 → [0,1] | Normalized rating for model |",
        "| days_since_epoch | timestamp / 86400 | Temporal feature (days) |",
        "| user_item_rating_diff | rating - item_avg_rating | Relative sentiment above/below crowd |",
        "",
        "## Co-occurrence Features",
        "",
        "Item-item co-occurrence is computed as the number of shared users",
        "who rated both items. Built from the top 500 most-rated items",
        "using a binary user-item pivot matrix and dot product.",
        "Used for item-based collaborative filtering and 'customers also bought' recommendations.",
        "",
        "## Encoders Applied (Task 5 gap closure)",
        "",
        "| Column | Encoder | Output |",
        "|--------|---------|--------|",
        "| category | sklearn LabelEncoder | Integer index 0..N_categories |",
        "| brand | sklearn LabelEncoder | Integer index 0..N_brands |",
        "| price | sklearn MinMaxScaler | Float in [0.0, 1.0] |",
        "| rating | Manual scaling (r-1)/4 | Float in [0.0, 1.0] |",
        "",
        "## SQL Schema (SQLite)",
        "",
        "```sql",
        "-- user_features",
        "CREATE TABLE user_features (",
        "    userId                  TEXT PRIMARY KEY,",
        "    user_rating_count       INTEGER,",
        "    user_avg_rating         REAL,",
        "    user_rating_std         REAL,",
        "    user_unique_items       INTEGER,",
        "    user_recency_days       REAL,",
        "    user_category_diversity INTEGER",
        ");",
        "",
        "-- item_features",
        "CREATE TABLE item_features (",
        "    productId           TEXT PRIMARY KEY,",
        "    item_rating_count   INTEGER,",
        "    item_avg_rating     REAL,",
        "    item_rating_std     REAL,",
        "    item_unique_users   INTEGER,",
        "    price               REAL,",
        "    category            TEXT,",
        "    brand               TEXT,",
        "    category_encoded    INTEGER,",
        "    brand_encoded       INTEGER,",
        "    price_normalized    REAL",
        ");",
        "",
        "-- interaction_features",
        "CREATE TABLE interaction_features (",
        "    userId                  TEXT,",
        "    productId               TEXT,",
        "    rating                  REAL,",
        "    timestamp               INTEGER,",
        "    rating_normalized       REAL,",
        "    days_since_epoch        INTEGER,",
        "    user_item_rating_diff   REAL,",
        "    PRIMARY KEY (userId, productId)",
        ");",
        "",
        "-- cooccurrence",
        "CREATE TABLE cooccurrence (",
        "    productId       TEXT,",
        "    co_productId    TEXT,",
        "    cooc_score      REAL,",
        "    PRIMARY KEY (productId, co_productId)",
        ");",
        "```",
    ]

    path = REPORTS_DIR / "feature_engineering_summary.md"
    path.write_text("\n".join(lines))
    log.info(f"Feature summary → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_feature_engineering():
    log.info("========== Feature Engineering Pipeline START ==========")

    ratings, products = load_processed()

    # Build features
    user_f             = build_user_features(ratings, products)
    item_f, cats, brands = build_item_features(ratings, products)
    inter_f            = build_interaction_features(ratings, item_f)
    cooc_f             = build_cooccurrence(ratings, top_n=500)

    # Store to SQLite
    store_to_sqlite(user_f, item_f, inter_f, cooc_f)

    # Write metadata + summary
    write_metadata(cats, brands, user_f, item_f, inter_f)
    write_feature_summary(user_f, item_f, inter_f, cooc_f)

    log.info("========== Feature Engineering COMPLETE ==========")
    log.info(f"DB         → {DB_PATH}")
    log.info(f"Metadata   → {METADATA_PATH}")

    # Print sample feature retrieval
    log.info("\n--- Sample feature retrieval (first 3 users) ---")
    conn = sqlite3.connect(DB_PATH)
    sample = pd.read_sql(
        "SELECT * FROM user_features LIMIT 3", conn)
    log.info(f"\n{sample.to_string(index=False)}")
    conn.close()

    return DB_PATH


if __name__ == "__main__":
    run_feature_engineering()
