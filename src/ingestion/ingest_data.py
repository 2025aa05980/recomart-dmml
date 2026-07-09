"""
Task 2: Data Ingestion Script — RecoMart Pipeline
Dataset: Amazon Reviews 2023 — Video Games (JSONL format)

Three ingestion sources (two types):
  Source Type 1 — File-based (JSONL):
    1. Video_Games.jsonl      — user-item interactions (ratings)
    2. meta_Video_Games.jsonl — product metadata (catalog)

  Source Type 2 — REST API:
    3. RecoMart Product Catalog API (http://localhost:8080)
       Serves enriched product data with rating stats
       Start server: python src/ingestion/api_server.py

Features:
  - JSONL parsing (pandas read_json with lines=True)
  - REST API ingestion (requests + pagination)
  - Retry logic (3 attempts with backoff)
  - Structured logging + audit trail (MD5 checksum)
  - Timestamped raw file storage (data lake layout)
  - Scheduler for periodic re-ingestion
"""

import sys
import time
import shutil
import hashlib
import requests
import schedule
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    INTERACTIONS_DIR, PRODUCTS_DIR, LOGS_DIR,
    LOCAL_FILES, RATINGS_FIELDS, METADATA_FIELDS,
    INGESTION_INTERVAL_HOURS, SAMPLE_USERS, RANDOM_SEED,
    DATASET_NAME
)
from src.logger import get_logger

log = get_logger("ingestion")


# ── Helpers ───────────────────────────────────────────────────────────────────

def timestamp() -> str:
    return now_utc().strftime("%Y%m%d_%H%M%S")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, retries: int = 3, backoff: int = 5) -> bool:
    """Download file with retry logic. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            log.info(f"Attempt {attempt}/{retries} — downloading {url}")
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    f.write(chunk)
            log.info(f"Downloaded → {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            return True
        except Exception as e:
            log.warning(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                log.info(f"Retrying in {backoff}s ...")
                time.sleep(backoff)
    log.error(f"All {retries} attempts failed.")
    return False


# ── Source 1: Ratings JSONL ───────────────────────────────────────────────────

def ingest_ratings() -> Path | None:
    """
    Ingest Video_Games.jsonl — user-item interactions.

    Raw JSONL fields used:
        user_id      → userId
        parent_asin  → productId
        rating       → rating  (1.0–5.0)
        timestamp    → timestamp (Unix ms)

    Output: data/raw/interactions/ratings_YYYYMMDD_HHMMSS.csv
    """
    log.info("=== Ingesting ratings (user-item interactions) ===")
    ts   = timestamp()
    dest = INTERACTIONS_DIR / f"ratings_{ts}.csv"
    INTERACTIONS_DIR.mkdir(parents=True, exist_ok=True)

    local = LOCAL_FILES["ratings"]
    if not local.exists():
        log.error(
            f"Ratings file not found: {local}\n"
            f"  → Place 'Video_Games.jsonl' at:\n"
            f"    {local}"
        )
        return None

    log.info(f"Reading JSONL: {local.name} ({local.stat().st_size / 1e6:.1f} MB)")

    try:
        # Read JSONL — one JSON object per line
        df_raw = pd.read_json(local, lines=True)
        log.info(f"Raw records loaded: {len(df_raw):,} | Columns: {list(df_raw.columns)}")

        # Select and rename only the fields we need
        available = {k: v for k, v in RATINGS_FIELDS.items() if k in df_raw.columns}
        missing   = set(RATINGS_FIELDS.keys()) - set(available.keys())
        if missing:
            log.warning(f"Missing expected fields: {missing}")

        df = df_raw[list(available.keys())].rename(columns=available).copy()

        # Validate rating range
        before = len(df)
        df = df[df["rating"].between(1, 5)]
        log.info(f"Rating range filter: {before - len(df):,} rows dropped")

        # Drop rows missing userId or productId
        df.dropna(subset=["userId", "productId"], inplace=True)

        # Convert timestamp: pandas may auto-parse as datetime or keep as int
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = df["timestamp"].astype("int64") // 10**9
        else:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)
            # If milliseconds (>1e12), convert to seconds
            if df["timestamp"].max() > 1e12:
                df["timestamp"] = (df["timestamp"] / 1000).astype(int)

        # Sample to top SAMPLE_USERS by activity
        top_users = (
            df.groupby("userId")
            .size()
            .nlargest(SAMPLE_USERS)
            .index
        )
        df_sample = df[df["userId"].isin(top_users)].copy()

        log.info(
            f"Sampled: {df_sample['userId'].nunique():,} users | "
            f"{df_sample['productId'].nunique():,} products | "
            f"{len(df_sample):,} interactions"
        )

        df_sample.to_csv(dest, index=False)
        checksum = md5(dest)
        log.info(f"Ratings saved → {dest.name} | MD5: {checksum}")
        _write_audit_log("ratings_jsonl", dest, len(df_sample), checksum)
        return dest

    except Exception as e:
        log.error(f"Ratings ingestion error: {e}", exc_info=True)
        return None


# ── Source 2: Product Metadata JSONL ─────────────────────────────────────────

def ingest_products() -> Path | None:
    """
    Ingest meta_Video_Games.jsonl — product metadata.

    Raw JSONL fields used:
        parent_asin  → productId
        title        → title
        price        → price
        store        → brand
        categories   → category
        description  → description

    Output: data/raw/products/products_YYYYMMDD_HHMMSS.csv
    """
    log.info("=== Ingesting product metadata (catalog) ===")
    ts       = timestamp()
    csv_dest = PRODUCTS_DIR / f"products_{ts}.csv"
    PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)

    local = LOCAL_FILES["metadata"]
    if not local.exists():
        log.warning(
            f"Metadata file not found: {local}\n"
            f"  → Falling back to synthetic product catalog."
        )
        return _generate_synthetic_products(csv_dest)

    log.info(f"Reading JSONL: {local.name} ({local.stat().st_size / 1e6:.1f} MB)")

    try:
        df_raw = pd.read_json(local, lines=True)
        log.info(f"Raw records loaded: {len(df_raw):,} | Columns: {list(df_raw.columns)}")

        # Select and rename fields
        available = {k: v for k, v in METADATA_FIELDS.items() if k in df_raw.columns}
        df = df_raw[list(available.keys())].rename(columns=available).copy()

        # Clean: categories list → single string
        if "category" in df.columns:
            df["category"] = df["category"].apply(
                lambda x: " > ".join(x) if isinstance(x, list) and x else ""
            )

        # Clean: description list → single string, truncate
        if "description" in df.columns:
            df["description"] = df["description"].apply(
                lambda x: " ".join(x)[:300] if isinstance(x, list) else str(x)[:300]
            )

        # Clean: price — strip non-numeric characters
        if "price" in df.columns:
            df["price"] = pd.to_numeric(
                df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True),
                errors="coerce"
            )

        # Deduplicate on productId
        before = len(df)
        df.drop_duplicates(subset="productId", inplace=True)
        df.dropna(subset=["productId"], inplace=True)
        log.info(f"Deduplication: {before - len(df):,} duplicates removed → {len(df):,} products")

        df.to_csv(csv_dest, index=False)
        checksum = md5(csv_dest)
        log.info(f"Products saved → {csv_dest.name} | MD5: {checksum}")
        _write_audit_log("metadata_jsonl", csv_dest, len(df), checksum)
        return csv_dest

    except Exception as e:
        log.error(f"Metadata ingestion error: {e}", exc_info=True)
        log.warning("Falling back to synthetic product catalog.")
        return _generate_synthetic_products(csv_dest)


# ── Fallback: Synthetic Product Catalog ───────────────────────────────────────

def _generate_synthetic_products(dest: Path) -> Path:
    """
    Generate synthetic Video Games product catalog using Faker.
    Used only when meta_Video_Games.jsonl is unavailable.
    """
    from faker import Faker
    import random
    fake   = Faker()
    random.seed(RANDOM_SEED)

    categories = [
        "Video Games > PlayStation 5 > Games",
        "Video Games > Xbox Series X > Games",
        "Video Games > Nintendo Switch > Games",
        "Video Games > PC > Games",
        "Video Games > Accessories > Controllers",
        "Video Games > Accessories > Headsets",
        "Video Games > Retro Gaming > Classic Consoles",
        "Video Games > Virtual Reality > VR Games",
    ]
    brands = [
        "Sony", "Microsoft", "Nintendo", "Activision", "EA Sports",
        "Ubisoft", "Rockstar Games", "Bethesda", "Square Enix", "Capcom",
        "SEGA", "Bandai Namco", "2K Games", "Warner Bros Games"
    ]

    records = []
    for _ in range(5000):
        records.append({
            "productId":   fake.bothify(text="B0??????????").upper(),
            "title":       fake.catch_phrase() + " - " + random.choice(
                               ["PS5", "Xbox", "Switch", "PC", "VR Edition"]),
            "price":       round(random.uniform(9.99, 79.99), 2),
            "brand":       random.choice(brands),
            "category":    random.choice(categories),
            "description": fake.text(max_nb_chars=250),
        })

    df = pd.DataFrame(records).drop_duplicates(subset="productId")
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    log.info(f"Synthetic catalog → {dest.name} ({len(df):,} products)")
    _write_audit_log("products_synthetic", dest, len(df), md5(dest))
    return dest


# ── Audit Log ─────────────────────────────────────────────────────────────────

def _write_audit_log(source: str, path: Path, rows: int, checksum: str):
    """Append ingestion record to audit CSV (append-only, never deleted)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "ingestion_audit.csv"
    entry = pd.DataFrame([{
        "ingested_at":  now_utc().isoformat(),
        "source":       source,
        "file":         path.name,
        "rows":         rows,
        "size_bytes":   path.stat().st_size,
        "md5":          checksum,
        "status":       "SUCCESS",
        "dataset":      DATASET_NAME,
    }])
    header = not log_path.exists()
    entry.to_csv(log_path, mode="a", header=header, index=False)
    log.debug(f"Audit log updated → {log_path.name}")


# ── Source 3: REST API ingestion ──────────────────────────────────────────────

API_BASE_URL  = "http://127.0.0.1:8080"
API_PAGE_LIMIT = 100   # records per API page


def ingest_api_products() -> Path | None:
    """
    Ingest product catalog from the RecoMart REST API (Source Type 2).

    Calls the local Flask API server (api_server.py) with pagination,
    collecting all available products across multiple pages.

    API endpoint: GET /api/products?page=N&limit=100
    Output: data/raw/products/api_products_YYYYMMDD_HHMMSS.csv

    Note: Start api_server.py before running this function.
    """
    log.info("=== Ingesting product catalog via REST API ===")
    log.info(f"API base URL: {API_BASE_URL}")
    ts   = timestamp()
    dest = PRODUCTS_DIR / f"api_products_{ts}.csv"
    PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Health check ──────────────────────────────────────────────────────────
    try:
        health = requests.get(
            f"{API_BASE_URL}/api/health", timeout=5)
        health.raise_for_status()
        health_data = health.json()
        log.info(f"API health: {health_data.get('status')} | "
                 f"Products available: {health_data.get('products_loaded'):,}")
    except Exception as e:
        log.warning(f"API server not reachable: {e}")
        log.warning("Start server with: python src/ingestion/api_server.py")
        log.warning("Skipping REST API ingestion — using JSONL source only")
        return None

    # ── Paginated fetch ───────────────────────────────────────────────────────
    all_records = []
    page        = 1
    total_pages = 1   # updated after first response

    while page <= total_pages:
        for attempt in range(1, 4):
            try:
                resp = requests.get(
                    f"{API_BASE_URL}/api/products",
                    params={"page": page, "limit": API_PAGE_LIMIT},
                    timeout=30,
                )
                resp.raise_for_status()
                data       = resp.json()
                records    = data.get("data", [])
                pagination = data.get("pagination", {})
                total_pages = pagination.get("pages", 1)

                all_records.extend(records)
                log.info(f"API page {page}/{total_pages} → "
                         f"{len(records)} records fetched")
                break

            except Exception as e:
                log.warning(f"API page {page} attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    time.sleep(3)
                else:
                    log.error(f"Page {page} failed after 3 attempts — stopping")
                    total_pages = 0   # exit loop

        page += 1

    if not all_records:
        log.error("No records received from API")
        return None

    # ── Save to CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    df.to_csv(dest, index=False)
    checksum = md5(dest)

    log.info(f"API ingestion complete: {len(df):,} products")
    log.info(f"Columns: {list(df.columns)}")
    log.info(f"Saved → {dest.name} | MD5: {checksum}")
    _write_audit_log("rest_api_products", dest, len(df), checksum)

    # ── Fetch catalog stats from API ──────────────────────────────────────────
    try:
        stats_resp = requests.get(f"{API_BASE_URL}/api/stats", timeout=10)
        stats      = stats_resp.json().get("data", {})
        log.info(f"Catalog stats from API:")
        log.info(f"  Total products   : {stats.get('total_products'):,}")
        log.info(f"  Total categories : {stats.get('total_categories')}")
        log.info(f"  Avg price        : ${stats.get('avg_price')}")
        log.info(f"  Top category     : {stats.get('top_category')}")
    except Exception:
        pass

    return dest


# ── Pipeline Runner ───────────────────────────────────────────────────────────

def run_ingestion(include_api: bool = True):
    """
    Run full ingestion — all sources.

    Sources:
      Type 1 (File-based): ratings JSONL + products JSONL
      Type 2 (REST API):   product catalog API (if server is running)

    Args:
        include_api: If True, also ingest from REST API source.
                     Set False if api_server.py is not running.
    """
    log.info("========== RecoMart Ingestion Pipeline START ==========")
    log.info(f"Dataset: {DATASET_NAME}")
    log.info(f"Sources: JSONL (file) + {'REST API' if include_api else 'REST API skipped'}")
    start = time.time()

    # Source Type 1 — File-based JSONL
    r_path = ingest_ratings()
    p_path = ingest_products()

    # Source Type 2 — REST API
    api_path = None
    if include_api:
        api_path = ingest_api_products()

    elapsed = time.time() - start
    if r_path and p_path:
        log.info(f"========== Ingestion COMPLETE in {elapsed:.1f}s ==========")
        log.info(f"  Source 1a (JSONL) Ratings  → {r_path.name}")
        log.info(f"  Source 1b (JSONL) Products → {p_path.name}")
        if api_path:
            log.info(f"  Source 2  (API)   Products → {api_path.name}")
        else:
            log.info(f"  Source 2  (API)   Products → skipped/unavailable")
    else:
        log.error(f"========== Ingestion PARTIAL/FAILED in {elapsed:.1f}s ==========")

    return r_path, p_path, api_path


def schedule_ingestion(include_api: bool = True):
    """Run ingestion now, then on schedule every N hours."""
    run_ingestion(include_api=include_api)
    schedule.every(INGESTION_INTERVAL_HOURS).hours.do(
        run_ingestion, include_api=include_api)
    log.info(f"Scheduler active — next run in {INGESTION_INTERVAL_HOURS}h. Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(60)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="RecoMart Data Ingestion — Video Games\n"
                    "Sources: JSONL files (Type 1) + REST API (Type 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run continuously on schedule (every 24h)")
    parser.add_argument(
        "--no-api", action="store_true",
        help="Skip REST API ingestion (use if api_server.py is not running)")
    args = parser.parse_args()

    include_api = not args.no_api

    if args.schedule:
        schedule_ingestion(include_api=include_api)
    else:
        run_ingestion(include_api=include_api)
