"""
Task 8: Data Versioning & Lineage — RecoMart Pipeline
Tool: DVC (Data Version Control)

What this script does:
  1. Initializes DVC in the project (dvc init)
  2. Configures local DVC remote storage
  3. Adds key data files to DVC tracking
  4. Creates .dvc files for versioned datasets
  5. Records metadata lineage (source, date, transformations)
  6. Generates versioning documentation

DVC tracks:
  data/raw/interactions/     ← raw ratings JSONL output
  data/raw/products/         ← raw products JSONL output
  data/processed/            ← cleaned datasets
  feature_store/             ← SQLite feature DB

Lineage tracked:
  source → ingestion → validation → cleaning → features → model

Run this script AFTER:
  - ingest_data.py       (Task 2)
  - validate_data.py     (Task 4)
  - eda_notebook.ipynb   (Task 5)
  - feature_engineering.py (Task 6)
"""

import sys
import json
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    BASE_DIR, RAW_DIR, PROCESSED_DIR,
    FEATURE_STORE, REPORTS_DIR, DATASET_NAME
)
from src.logger import get_logger

log = get_logger("dvc_versioning")

DVC_REMOTE = BASE_DIR / ".dvc_remote"   # local remote storage folder


# ── Shell command runner ──────────────────────────────────────────────────────

def run(cmd: str, cwd: Path = BASE_DIR,
        check: bool = True) -> tuple[int, str, str]:
    """Run a shell command, log output, return (returncode, stdout, stderr)."""
    log.info(f"$ {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=str(cwd),
        capture_output=True, text=True
    )
    if result.stdout.strip():
        log.info(result.stdout.strip())
    if result.stderr.strip():
        # DVC often writes info to stderr — only log as error if failed
        if result.returncode != 0 and check:
            log.error(result.stderr.strip())
        else:
            log.debug(result.stderr.strip())
    return result.returncode, result.stdout, result.stderr


# ── Step 1: Git init (if needed) ──────────────────────────────────────────────

def ensure_git():
    """Ensure git is initialized — DVC requires git."""
    git_dir = BASE_DIR / ".git"
    if git_dir.exists():
        log.info("Git already initialized ✅")
        return True

    log.info("Initializing git repository...")
    rc, _, _ = run("git init")
    if rc != 0:
        log.error("git init failed — install git first")
        return False

    # Create .gitignore
    gitignore = BASE_DIR / ".gitignore"
    gitignore.write_text(
        "# Python\n__pycache__/\n*.pyc\n*.pyo\n.env\n\n"
        "# DVC\n/dvc_remote/\n\n"
        "# Data (large files — tracked by DVC)\n"
        "data/raw/interactions/*.jsonl\n"
        "data/raw/products/*.jsonl\n\n"
        "# Models\n*.pkl\n\n"
        "# Jupyter\n.ipynb_checkpoints/\n\n"
        "# MacOS\n.DS_Store\n"
    )
    run("git add .gitignore")
    run('git commit -m "Initial commit — add .gitignore"')
    log.info("Git initialized ✅")
    return True


# ── Step 2: DVC init ──────────────────────────────────────────────────────────

def ensure_dvc():
    """Initialize DVC if not already done."""
    dvc_dir = BASE_DIR / ".dvc"
    if dvc_dir.exists():
        log.info("DVC already initialized ✅")
        return True

    log.info("Initializing DVC...")
    rc, _, _ = run("dvc init")
    if rc != 0:
        log.error("dvc init failed — install with: pip install dvc")
        return False

    run("git add .dvc .dvcignore")
    run('git commit -m "Initialize DVC"')
    log.info("DVC initialized ✅")
    return True


# ── Step 3: Configure local remote ───────────────────────────────────────────

def setup_remote():
    """Configure a local DVC remote for storing cached data."""
    DVC_REMOTE.mkdir(parents=True, exist_ok=True)
    log.info(f"Setting up DVC remote → {DVC_REMOTE}")

    run(f'dvc remote add -d localremote "{DVC_REMOTE}"')
    run("git add .dvc/config")
    run('git commit -m "Configure DVC local remote"')
    log.info("DVC remote configured ✅")


# ── Step 4: Track data files ──────────────────────────────────────────────────

def track_data_files():
    """Add key data files and folders to DVC tracking."""
    log.info("Adding files to DVC tracking...")

    # Find latest timestamped files
    def latest_csv(directory: Path, prefix: str) -> Path | None:
        expected_len = len(prefix) + 1 + 8 + 1 + 6
        files = sorted([
            f for f in directory.glob(f"{prefix}_*.csv")
            if len(f.stem) == expected_len
            and f.stem[len(prefix)+1:].replace("_","").isdigit()
        ], reverse=True)
        return files[0] if files else None

    tracked = []

    # Track ratings CSV
    ratings_file = latest_csv(
        BASE_DIR / "data" / "raw" / "interactions", "ratings")
    if ratings_file:
        rc, _, _ = run(f'dvc add "{ratings_file.relative_to(BASE_DIR)}"')
        if rc == 0:
            tracked.append(str(ratings_file.relative_to(BASE_DIR)))
            log.info(f"Tracked: {ratings_file.name}")

    # Track products CSV
    products_file = latest_csv(
        BASE_DIR / "data" / "raw" / "products", "products")
    if products_file:
        rc, _, _ = run(f'dvc add "{products_file.relative_to(BASE_DIR)}"')
        if rc == 0:
            tracked.append(str(products_file.relative_to(BASE_DIR)))
            log.info(f"Tracked: {products_file.name}")

    # Track processed data folder
    processed = BASE_DIR / "data" / "processed"
    if any(processed.glob("*.csv")):
        rc, _, _ = run('dvc add data/processed')
        if rc == 0:
            tracked.append("data/processed")
            log.info("Tracked: data/processed/")

    # Track feature store DB
    db_file = FEATURE_STORE / "recomart_features.db"
    if db_file.exists():
        rc, _, _ = run(
            f'dvc add "{db_file.relative_to(BASE_DIR)}"')
        if rc == 0:
            tracked.append(str(db_file.relative_to(BASE_DIR)))
            log.info(f"Tracked: {db_file.name}")

    if tracked:
        run("git add .")
        run(f'git commit -m "Track data files with DVC: {len(tracked)} files"')
        log.info(f"DVC tracking: {len(tracked)} files/folders ✅")
    else:
        log.warning("No files found to track — run ingestion pipeline first")

    return tracked


# ── Step 5: DVC push ──────────────────────────────────────────────────────────

def push_to_remote():
    """Push tracked files to DVC remote cache."""
    log.info("Pushing data to DVC remote...")
    rc, _, _ = run("dvc push")
    if rc == 0:
        log.info("DVC push complete ✅")
    else:
        log.warning("DVC push had issues — check remote config")


# ── Step 6: Lineage metadata ──────────────────────────────────────────────────

def write_lineage_metadata(tracked: list):
    """
    Write pipeline lineage JSON — tracks each transformation step,
    its inputs, outputs, and the tool/script used.
    """
    now = datetime.now(timezone.utc).isoformat()

    lineage = {
        "pipeline":   "RecoMart Recommendation Pipeline",
        "dataset":    DATASET_NAME,
        "version":    "v1.0",
        "created_at": now,
        "tracked_files": tracked,
        "stages": [
            {
                "stage":      "1_ingestion",
                "task":       "Task 2",
                "script":     "src/ingestion/ingest_data.py",
                "inputs":     ["Video_Games.jsonl", "meta_Video_Games.jsonl"],
                "outputs":    ["data/raw/interactions/ratings_*.csv",
                               "data/raw/products/products_*.csv"],
                "tool":       "pandas, requests, schedule",
                "description":"Ingested Amazon Reviews 2023 Video Games dataset. "
                              "Sampled top 10,000 users. "
                              "Synthetic product catalog fallback available.",
            },
            {
                "stage":      "2_validation",
                "task":       "Task 4",
                "script":     "src/validation/validate_data.py",
                "inputs":     ["data/raw/interactions/ratings_*.csv",
                               "data/raw/products/products_*.csv"],
                "outputs":    ["reports/data_quality_report.md",
                               "reports/data_quality_summary.csv",
                               "reports/ge_results_ratings.csv",
                               "reports/ge_results_products.csv"],
                "tool":       "pandas, great_expectations",
                "description":"12 pandas checks + 18 GE expectations. "
                              "8/12 pandas passed, 16/18 GE passed. "
                              "Key issues: duplicate ratings (deduped in prep), "
                              "54.8% missing price (imputed in prep).",
            },
            {
                "stage":      "3_preparation",
                "task":       "Task 5",
                "script":     "src/eda/eda_notebook.ipynb",
                "inputs":     ["data/raw/interactions/ratings_*.csv",
                               "data/raw/products/products_*.csv"],
                "outputs":    ["data/processed/cleaned_ratings.csv",
                               "data/processed/cleaned_products.csv",
                               "reports/eda_plots/*.png"],
                "tool":       "pandas, matplotlib, seaborn, scikit-learn",
                "description":"Deduplication: 5,548 exact + 9,923 user-item pairs. "
                              "Timestamp filter: 16 out-of-window records dropped. "
                              "Price imputation: category median. "
                              "Final: 258,524 interactions, 137,269 products.",
            },
            {
                "stage":      "4_feature_engineering",
                "task":       "Task 6",
                "script":     "src/features/feature_engineering.py",
                "inputs":     ["data/processed/cleaned_ratings.csv",
                               "data/processed/cleaned_products.csv"],
                "outputs":    ["feature_store/recomart_features.db",
                               "feature_store/feature_metadata.json"],
                "tool":       "pandas, numpy, scikit-learn, sqlite3",
                "description":"Built 19 features across 4 tables. "
                              "LabelEncoder: category (508 classes), brand. "
                              "MinMaxScaler: price. "
                              "Co-occurrence matrix: top 500 items, 1,500 pairs.",
            },
            {
                "stage":      "5_feature_store",
                "task":       "Task 7",
                "script":     "src/features/feature_store.py",
                "inputs":     ["feature_store/recomart_features.db",
                               "feature_store/feature_metadata.json"],
                "outputs":    ["feature_store/recomart_features.db (version table)",
                               "reports/feature_store_documentation.md"],
                "tool":       "sqlite3, pandas",
                "description":"Custom SQLite feature store with versioned retrieval. "
                              "Version v1.0 registered. "
                              "Training set: 258,524 rows × 19 features. "
                              "Inference API: user × 100 candidate items.",
            },
            {
                "stage":      "6_model_training",
                "task":       "Task 9",
                "script":     "src/model/train_model.py",
                "inputs":     ["feature_store/recomart_features.db"],
                "outputs":    ["models/svd_model.pkl",
                               "models/model_metadata.json"],
                "tool":       "scikit-surprise, mlflow, scikit-learn",
                "description":"Pending — SVD collaborative filtering + "
                              "content-based filtering. MLflow tracking.",
            },
            {
                "stage":      "7_orchestration",
                "task":       "Task 10",
                "script":     "src/orchestration/pipeline_dag.py",
                "inputs":     ["All pipeline stages"],
                "outputs":    ["Automated end-to-end execution"],
                "tool":       "prefect",
                "description":"Pending — Prefect DAG wiring all stages.",
            },
        ],
    }

    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    lineage_path = FEATURE_STORE / "pipeline_lineage.json"
    with open(lineage_path, "w") as f:
        json.dump(lineage, f, indent=2)
    log.info(f"Lineage metadata → {lineage_path}")
    return lineage_path


# ── Step 7: DVC status check ──────────────────────────────────────────────────

def check_dvc_status():
    """Show current DVC status — which files are tracked/modified."""
    log.info("\n--- DVC Status ---")
    run("dvc status", check=False)

    log.info("\n--- Git Log (last 5 commits) ---")
    run("git log --oneline -5", check=False)

    log.info("\n--- DVC Tracked Files ---")
    run("dvc list . --dvc-only -R", check=False)


# ── Step 8: Versioning documentation ─────────────────────────────────────────

def write_versioning_doc(tracked: list, lineage_path: Path):
    """Generate versioning workflow documentation for PDF report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Data Versioning & Lineage — RecoMart Pipeline",
        "**Task 8 | DMML Assignment 1 | Group 37**",
        f"Generated: {now}",
        "",
        "---",
        "",
        "## Tool: DVC (Data Version Control)",
        "",
        "DVC extends Git to version large data files and ML models.",
        "It stores file metadata (`.dvc` files) in Git while keeping",
        "actual data in a separate remote storage.",
        "",
        "## Repository Structure",
        "",
        "```",
        "recomart/",
        "├── .git/                    ← Git repository",
        "├── .dvc/                    ← DVC configuration",
        "│   ├── config               ← remote storage config",
        "│   └── cache/               ← local DVC cache",
        "├── .dvc_remote/             ← local remote storage",
        "├── .dvcignore               ← files DVC should ignore",
        "└── [tracked files].dvc      ← DVC pointer files",
        "```",
        "",
        "## Files Tracked by DVC",
        "",
        "| File/Folder | Description | Stage |",
        "|-------------|-------------|-------|",
    ]

    stage_map = {
        "ratings": ("Raw user-item interactions (sampled)", "Ingestion"),
        "products": ("Raw product catalog", "Ingestion"),
        "processed": ("Cleaned datasets", "Preparation"),
        "recomart_features": ("Feature store SQLite DB", "Feature Engineering"),
    }

    for f in tracked:
        name = Path(f).name
        key  = next((k for k in stage_map if k in name), name)
        desc, stage = stage_map.get(key, (f, "—"))
        lines.append(f"| `{f}` | {desc} | {stage} |")

    lines += [
        "",
        "## Versioning Workflow",
        "",
        "```bash",
        "# 1. Initialize (one-time)",
        "git init && dvc init",
        "dvc remote add -d localremote .dvc_remote/",
        "",
        "# 2. Track data files after each pipeline run",
        "dvc add data/raw/interactions/ratings_*.csv",
        "dvc add data/raw/products/products_*.csv",
        "dvc add data/processed/",
        "dvc add feature_store/recomart_features.db",
        "",
        "# 3. Commit .dvc pointer files to Git",
        'git add . && git commit -m "Update dataset v1.0"',
        "",
        "# 4. Push data to remote",
        "dvc push",
        "",
        "# 5. Reproduce any version",
        "git checkout <commit-hash>",
        "dvc pull",
        "```",
        "",
        "## Pipeline Lineage",
        "",
        "| Stage | Task | Script | Key Transformation |",
        "|-------|------|--------|-------------------|",
        "| Ingestion | T2 | ingest_data.py | JSONL → CSV, 10K user sample |",
        "| Validation | T4 | validate_data.py | 12 pandas + 18 GE checks |",
        "| Preparation | T5 | eda_notebook.ipynb | Dedup, impute, normalize |",
        "| Feature Eng. | T6 | feature_engineering.py | 19 features → SQLite |",
        "| Feature Store | T7 | feature_store.py | Versioned retrieval API |",
        "| Model Training | T9 | train_model.py | SVD + MLflow tracking |",
        "| Orchestration | T10 | pipeline_dag.py | Prefect end-to-end DAG |",
        "",
        "## Metadata Tracked per Version",
        "",
        "| Metadata field | Value (v1.0) |",
        "|----------------|-------------|",
        "| Dataset | Amazon Reviews 2023 — Video Games |",
        "| Raw interactions | 4,624,615 total → 268,463 sampled |",
        "| After cleaning | 258,524 interactions |",
        "| Users | 10,000 |",
        "| Products (catalog) | 137,269 |",
        "| Products (rated) | 45,914 |",
        "| Features | 19 across 4 tables |",
        "| Sparsity | 99.94% |",
        "",
        f"Full lineage metadata: `feature_store/pipeline_lineage.json`",
    ]

    path = REPORTS_DIR / "versioning_documentation.md"
    path.write_text("\n".join(lines))
    log.info(f"Versioning docs → {path}")
    return path


# ── Entry point ───────────────────────────────────────────────────────────────

def run_dvc_setup():
    log.info("========== DVC Versioning Setup START ==========")

    # Steps 1–4
    if not ensure_git():
        return False
    if not ensure_dvc():
        return False
    setup_remote()
    tracked = track_data_files()

    # Step 5 — push to remote
    push_to_remote()

    # Step 6 — lineage metadata
    lineage_path = write_lineage_metadata(tracked)

    # Step 7 — status check
    check_dvc_status()

    # Step 8 — documentation
    doc_path = write_versioning_doc(tracked, lineage_path)

    log.info("========== DVC Versioning Setup COMPLETE ==========")
    log.info(f"Tracked files : {len(tracked)}")
    log.info(f"Lineage JSON  : {lineage_path}")
    log.info(f"Docs          : {doc_path}")
    return True


if __name__ == "__main__":
    run_dvc_setup()
