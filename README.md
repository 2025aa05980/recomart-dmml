# RecoMart Recommendation Pipeline
**DMML Assignment 1 | Group 37**
Student: Thanigaivel S | `2025aa05980@wilp.bits-pilani.ac.in`
Generated: 2026-07-07 02:58 UTC

## Dataset
**Amazon Reviews 2023 — Video Games**
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
