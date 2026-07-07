"""
RecoMart Pipeline Configuration
DMML Assignment 1 - Group 37
Student: Thanigaivel S (2025aa05980@wilp.bits-pilani.ac.in)

Dataset: Amazon Reviews 2023 — Video Games
Source:  McAuley Lab, UCSD (https://amazon-reviews-2023.github.io)
Format:  JSONL (JSON Lines) — one record per line
"""

from pathlib import Path

BASE_DIR         = Path(__file__).parent

# ── Data lake folders ─────────────────────────────────────────────────────────
RAW_DIR          = BASE_DIR / "data" / "raw"
INTERACTIONS_DIR = RAW_DIR  / "interactions"
PRODUCTS_DIR     = RAW_DIR  / "products"
LOGS_DIR         = RAW_DIR  / "logs"
PROCESSED_DIR    = BASE_DIR / "data" / "processed"

# ── Output folders ────────────────────────────────────────────────────────────
REPORTS_DIR      = BASE_DIR / "reports"
MODELS_DIR       = BASE_DIR / "models"
FEATURE_STORE    = BASE_DIR / "feature_store"
LOG_FILE         = BASE_DIR / "logs" / "pipeline.log"

# ── Local dataset files (place downloaded files here) ─────────────────────────
# Source 1: User-item interactions  → Video_Games.jsonl
# Source 2: Product metadata        → meta_Video_Games.jsonl
LOCAL_FILES = {
    "ratings":  INTERACTIONS_DIR / "Video_Games.jsonl",
    "metadata": PRODUCTS_DIR     / "meta_Video_Games.jsonl",
}

# ── Dataset info ──────────────────────────────────────────────────────────────
DATASET_NAME     = "Amazon Reviews 2023 — Video Games"
DATASET_CATEGORY = "Video_Games"

# ── Ratings JSONL field mapping ───────────────────────────────────────────────
# Raw fields in Video_Games.jsonl:
#   user_id, parent_asin, rating, timestamp, ...
RATINGS_FIELDS = {
    "user_id":     "userId",       # reviewer ID
    "parent_asin": "productId",    # product ASIN
    "rating":      "rating",       # 1.0 – 5.0
    "timestamp":   "timestamp",    # Unix ms epoch
}

# ── Metadata JSONL field mapping ──────────────────────────────────────────────
# Raw fields in meta_Video_Games.jsonl:
#   parent_asin, title, price, store, categories, features, description, ...
METADATA_FIELDS = {
    "parent_asin": "productId",
    "title":       "title",
    "price":       "price",
    "store":       "brand",
    "categories":  "category",
    "description": "description",
}

# ── Data parameters ───────────────────────────────────────────────────────────
RATING_SCALE             = (1, 5)
SAMPLE_USERS             = 10000   # top-N users by activity for pipeline dev
RANDOM_SEED              = 42

# ── Ingestion schedule ────────────────────────────────────────────────────────
INGESTION_INTERVAL_HOURS = 24

# ── Model parameters ──────────────────────────────────────────────────────────
TEST_SIZE                = 0.2
N_FACTORS                = 50      # SVD latent factors
N_EPOCHS                 = 20
K_EVAL                   = 10      # Precision@K, Recall@K, NDCG@K

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT        = "recomart_videogames_svd"
