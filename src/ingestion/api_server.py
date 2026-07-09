"""
RecoMart Product Catalog API Server
Task 2 — REST API Data Source (Source 2)

Serves product metadata as a RESTful HTTP API, simulating a production
internal catalog microservice. The ingestion pipeline calls this API
to fetch product data — demonstrating REST API ingestion alongside
file-based interaction ingestion.

Endpoints:
  GET /api/health                    — health check
  GET /api/products                  — paginated product list
  GET /api/products/<productId>      — single product by ID
  GET /api/products/category/<cat>   — products by category
  GET /api/products/popular          — top products by rating count
  GET /api/stats                     — catalog statistics

Usage:
  # Terminal 1 — start server
  python src/ingestion/api_server.py

  # Terminal 2 — call API
  curl http://localhost:8080/api/products?page=1&limit=5
  curl http://localhost:8080/api/products/B00000JRSB
  curl http://localhost:8080/api/stats
"""

import sys
import json
import logging
from pathlib import Path
from glob import glob
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Flask, jsonify, request, abort
import pandas as pd

from config import PRODUCTS_DIR, INTERACTIONS_DIR, DATASET_NAME
from src.logger import get_logger

log = get_logger("api_server")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ── Load product data at startup ──────────────────────────────────────────────

def load_products() -> pd.DataFrame:
    """Load most recent products CSV into memory at server startup."""
    expected_len_prefix = len("products") + 1 + 8 + 1 + 6
    files = sorted([
        f for f in PRODUCTS_DIR.glob("products_*.csv")
        if len(f.stem) == expected_len_prefix
        and f.stem[len("products")+1:].replace("_","").isdigit()
    ], reverse=True)

    if not files:
        log.warning("No products CSV found — using empty catalog")
        return pd.DataFrame(columns=[
            "productId","title","price","brand","category","description"])

    df = pd.read_csv(str(files[0]))
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["title"] = df["title"].fillna("Unknown Title")
    df["brand"] = df["brand"].fillna("Unknown")
    df["category"] = df["category"].fillna("Unknown")
    df["description"] = df["description"].fillna("")
    log.info(f"API server loaded {len(df):,} products from {files[0].name}")
    return df


def load_item_stats() -> pd.DataFrame:
    """Load item rating stats from ratings CSV for enrichment."""
    expected_len_prefix = len("ratings") + 1 + 8 + 1 + 6
    files = sorted([
        f for f in INTERACTIONS_DIR.glob("ratings_*.csv")
        if len(f.stem) == expected_len_prefix
        and f.stem[len("ratings")+1:].replace("_","").isdigit()
    ], reverse=True)

    if not files:
        return pd.DataFrame(columns=["productId","rating_count","avg_rating"])

    df = pd.read_csv(str(files[0]))
    stats = df.groupby("productId").agg(
        rating_count=("rating","count"),
        avg_rating=("rating","mean")
    ).reset_index()
    stats["avg_rating"] = stats["avg_rating"].round(2)
    return stats


# Load at startup
PRODUCTS_DF = load_products()
ITEM_STATS  = load_item_stats()

# Merge rating stats into products
if not ITEM_STATS.empty:
    PRODUCTS_DF = PRODUCTS_DF.merge(ITEM_STATS, on="productId", how="left")
    PRODUCTS_DF["rating_count"] = PRODUCTS_DF["rating_count"].fillna(0).astype(int)
    PRODUCTS_DF["avg_rating"]   = PRODUCTS_DF["avg_rating"].fillna(0.0).round(2)
else:
    PRODUCTS_DF["rating_count"] = 0
    PRODUCTS_DF["avg_rating"]   = 0.0


# ── Response helpers ──────────────────────────────────────────────────────────

def product_to_dict(row) -> dict:
    """Convert a DataFrame row to API response dict."""
    return {
        "productId":    row["productId"],
        "title":        row["title"],
        "price":        round(float(row.get("price", 0)), 2),
        "brand":        row.get("brand", "Unknown"),
        "category":     row.get("category", "Unknown"),
        "description":  str(row.get("description", ""))[:200],
        "rating_count": int(row.get("rating_count", 0)),
        "avg_rating":   float(row.get("avg_rating", 0.0)),
    }


def api_response(data, total=None, page=None, limit=None) -> dict:
    """Standard API response envelope."""
    resp = {
        "status":     "success",
        "dataset":    DATASET_NAME,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "data":       data,
    }
    if total is not None:
        resp["pagination"] = {
            "total":   total,
            "page":    page,
            "limit":   limit,
            "pages":   (total + limit - 1) // limit,
        }
    return resp


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint — used by ingestion script to verify server is up."""
    return jsonify({
        "status":        "healthy",
        "products_loaded": len(PRODUCTS_DF),
        "dataset":       DATASET_NAME,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/products", methods=["GET"])
def get_products():
    """
    Paginated product list.
    Query params:
      page  (int, default 1)
      limit (int, default 20, max 100)
      category (str, optional filter)
      min_price (float, optional)
      max_price (float, optional)
    """
    page      = max(1, int(request.args.get("page",  1)))
    limit     = min(100, max(1, int(request.args.get("limit", 20))))
    category  = request.args.get("category", None)
    min_price = request.args.get("min_price", None)
    max_price = request.args.get("max_price", None)

    df = PRODUCTS_DF.copy()

    # Apply filters
    if category:
        df = df[df["category"].str.contains(category, case=False, na=False)]
    if min_price:
        df = df[df["price"] >= float(min_price)]
    if max_price:
        df = df[df["price"] <= float(max_price)]

    total  = len(df)
    start  = (page - 1) * limit
    end    = start + limit
    page_df = df.iloc[start:end]

    products = [product_to_dict(row) for _, row in page_df.iterrows()]

    log.info(f"GET /api/products page={page} limit={limit} → {len(products)} items")
    return jsonify(api_response(products, total=total, page=page, limit=limit))


@app.route("/api/products/<product_id>", methods=["GET"])
def get_product(product_id):
    """Single product lookup by productId (ASIN)."""
    row = PRODUCTS_DF[PRODUCTS_DF["productId"] == product_id]
    if row.empty:
        log.warning(f"GET /api/products/{product_id} → 404")
        abort(404, description=f"Product '{product_id}' not found")

    product = product_to_dict(row.iloc[0])
    log.info(f"GET /api/products/{product_id} → found")
    return jsonify(api_response(product))


@app.route("/api/products/category/<path:category>", methods=["GET"])
def get_by_category(category):
    """Products filtered by category path."""
    limit = min(100, max(1, int(request.args.get("limit", 20))))
    df    = PRODUCTS_DF[
        PRODUCTS_DF["category"].str.contains(category, case=False, na=False)
    ].head(limit)

    if df.empty:
        return jsonify(api_response([], total=0, page=1, limit=limit))

    products = [product_to_dict(row) for _, row in df.iterrows()]
    log.info(f"GET /api/products/category/{category} → {len(products)} items")
    return jsonify(api_response(products, total=len(products), page=1, limit=limit))


@app.route("/api/products/popular", methods=["GET"])
def get_popular():
    """Top products by rating count — most reviewed items."""
    limit = min(100, max(1, int(request.args.get("limit", 20))))
    df    = PRODUCTS_DF.nlargest(limit, "rating_count")
    products = [product_to_dict(row) for _, row in df.iterrows()]
    log.info(f"GET /api/products/popular limit={limit} → {len(products)} items")
    return jsonify(api_response(products, total=len(products), page=1, limit=limit))


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Catalog-level statistics."""
    stats = {
        "total_products":       len(PRODUCTS_DF),
        "total_categories":     int(PRODUCTS_DF["category"].nunique()),
        "total_brands":         int(PRODUCTS_DF["brand"].nunique()),
        "avg_price":            round(float(PRODUCTS_DF["price"].mean()), 2),
        "median_price":         round(float(PRODUCTS_DF["price"].median()), 2),
        "products_with_ratings":int((PRODUCTS_DF["rating_count"] > 0).sum()),
        "avg_rating_overall":   round(float(PRODUCTS_DF["avg_rating"].mean()), 2),
        "top_category":         PRODUCTS_DF["category"].value_counts().index[0]
                                if len(PRODUCTS_DF) > 0 else "N/A",
        "top_brand":            PRODUCTS_DF["brand"].value_counts().index[0]
                                if len(PRODUCTS_DF) > 0 else "N/A",
    }
    log.info("GET /api/stats")
    return jsonify(api_response(stats))


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "message": str(e)}), 404


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"status": "error", "message": str(e)}), 400


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="RecoMart Product Catalog REST API")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to run on (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    log.info(f"Starting RecoMart Product Catalog API")
    log.info(f"Products loaded: {len(PRODUCTS_DF):,}")
    log.info(f"Listening at: http://{args.host}:{args.port}")
    log.info(f"Endpoints:")
    log.info(f"  GET /api/health")
    log.info(f"  GET /api/products?page=1&limit=20")
    log.info(f"  GET /api/products/<productId>")
    log.info(f"  GET /api/products/category/<category>")
    log.info(f"  GET /api/products/popular?limit=20")
    log.info(f"  GET /api/stats")

    # Disable Flask's default logger to avoid duplicate logs
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    app.run(host=args.host, port=args.port, debug=False)
