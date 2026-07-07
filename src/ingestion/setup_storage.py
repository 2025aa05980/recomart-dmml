"""
Task 3: Storage Setup — RecoMart Data Lake
Creates folder structure, README, storage docs, requirements.txt
Dataset: Amazon Reviews 2023 — Video Games (JSONL)
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

def now_utc():
    return datetime.now(timezone.utc)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BASE_DIR, REPORTS_DIR, DATASET_NAME, DATASET_CATEGORY
from src.logger import get_logger

log = get_logger("storage_setup")


def create_structure():
    dirs = [
        BASE_DIR / "data" / "raw" / "interactions",
        BASE_DIR / "data" / "raw" / "products",
        BASE_DIR / "data" / "raw" / "logs",
        BASE_DIR / "data" / "processed",
        BASE_DIR / "src" / "ingestion",
        BASE_DIR / "src" / "validation",
        BASE_DIR / "src" / "eda",
        BASE_DIR / "src" / "features",
        BASE_DIR / "src" / "model",
        BASE_DIR / "src" / "orchestration",
        BASE_DIR / "feature_store",
        BASE_DIR / "models",
        BASE_DIR / "reports",
        BASE_DIR / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").touch()
    log.info(f"Created {len(dirs)} directories")


def write_readme():
    content = f"""# RecoMart Recommendation Pipeline
**DMML Assignment 1 | Group 37**
Student: Thanigaivel S | `2025aa05980@wilp.bits-pilani.ac.in`
Generated: {now_utc().strftime("%Y-%m-%d %H:%M UTC")}

## Dataset
**{DATASET_NAME}**
- Interactions: `Video_Games.jsonl` — userId, productId, rating, timestamp
- Metadata:     `meta_Video_Games.jsonl` — productId, title, price, brand, category

## Project Structure
```
recomart/
├── config.py                      # Central config (paths, params, field mappings)
├── requirements.txt               # All dependencies
├── data/
│   ├── raw/
│   │   ├── interactions/          # ratings_YYYYMMDD_HHMMSS.csv
│   │   ├── products/              # products_YYYYMMDD_HHMMSS.csv
│   │   └── logs/                  # ingestion_audit.csv
│   └── processed/                 # cleaned_ratings.csv, cleaned_products.csv
├── src/
│   ├── ingestion/                 # Tasks 2–3
│   ├── validation/                # Task 4
│   ├── eda/                       # Task 5
│   ├── features/                  # Task 6
│   ├── model/                     # Task 9
│   └── orchestration/             # Task 10 (Prefect DAG)
├── feature_store/                 # Task 7 (SQLite registry)
├── models/                        # Trained model artifacts
├── reports/                       # DQ reports, EDA plots
└── logs/                          # pipeline.log
```

## Quick Start
```bash
pip install -r requirements.txt

# Place dataset files:
# data/raw/interactions/Video_Games.jsonl
# data/raw/products/meta_Video_Games.jsonl

python src/ingestion/setup_storage.py   # Task 3
python src/ingestion/ingest_data.py     # Task 2
python src/validation/validate_data.py  # Task 4
python src/orchestration/pipeline_dag.py  # Task 10 (full pipeline)
```

## File Naming Convention
```
ratings_YYYYMMDD_HHMMSS.csv    ← sampled interaction snapshot
products_YYYYMMDD_HHMMSS.csv   ← parsed product catalog snapshot
ingestion_audit.csv            ← append-only audit trail
```
"""
    (BASE_DIR / "README.md").write_text(content)
    log.info("README.md written")


def write_storage_doc():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    content = f"""# Storage Structure Documentation
**RecoMart Pipeline — Task 3**
Dataset: {DATASET_NAME}
Generated: {now_utc().strftime("%Y-%m-%d %H:%M UTC")}

## Data Lake Layout

| Path | Content | Format | Partition Key |
|------|---------|--------|---------------|
| `data/raw/interactions/` | User-item ratings | CSV | Ingest timestamp |
| `data/raw/products/` | Product metadata catalog | CSV | Ingest timestamp |
| `data/raw/logs/` | Ingestion audit trail | CSV (append-only) | — |
| `data/processed/` | Cleaned + merged datasets | CSV | Pipeline run |

## Source Files (place before running)
| File | Location | Description |
|------|----------|-------------|
| `Video_Games.jsonl` | `data/raw/interactions/` | Amazon 2023 ratings JSONL |
| `meta_Video_Games.jsonl` | `data/raw/products/` | Amazon 2023 metadata JSONL |

## JSONL Field Mapping

### Ratings (Video_Games.jsonl)
| Raw Field | Pipeline Field | Description |
|-----------|---------------|-------------|
| `user_id` | `userId` | Reviewer ID |
| `parent_asin` | `productId` | Product ASIN |
| `rating` | `rating` | Star rating 1.0–5.0 |
| `timestamp` | `timestamp` | Unix epoch (ms → s) |

### Metadata (meta_Video_Games.jsonl)
| Raw Field | Pipeline Field | Description |
|-----------|---------------|-------------|
| `parent_asin` | `productId` | Product ASIN |
| `title` | `title` | Product title |
| `price` | `price` | Numeric price (USD) |
| `store` | `brand` | Brand/store name |
| `categories` | `category` | Category path string |
| `description` | `description` | Text (truncated 300 chars) |

## Retention Policy
- Raw JSONL source files: retained locally (not uploaded to Drive — too large)
- Sampled CSV outputs: uploaded to Google Drive `02_dataset/raw/`
- Audit log: append-only, never deleted
- Processed data: versioned via DVC (Task 8)
"""
    (REPORTS_DIR / "storage_structure.md").write_text(content)
    log.info("storage_structure.md written")


def write_requirements():
    content = """# RecoMart Pipeline — Dependencies
# DMML Assignment 1 | Group 37

# Core data
pandas>=2.0
numpy>=1.24
scipy>=1.11

# Ingestion & scheduling
requests>=2.31
schedule>=1.2

# Data generation (synthetic fallback)
faker>=19.0

# Validation & profiling
great-expectations>=0.18
ydata-profiling>=4.5

# Visualisation & EDA
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
ipykernel>=6.0

# ML & recommendation
scikit-learn>=1.3
scikit-surprise>=1.1
mlflow>=2.8

# Feature store
sqlalchemy>=2.0

# Versioning
dvc>=3.0

# Orchestration
prefect>=2.14
"""
    (BASE_DIR / "requirements.txt").write_text(content)
    log.info("requirements.txt written")


if __name__ == "__main__":
    log.info("=== RecoMart Storage Setup START ===")
    create_structure()
    write_readme()
    write_storage_doc()
    write_requirements()
    log.info("=== Storage Setup COMPLETE ===")
