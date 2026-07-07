# Data Versioning & Lineage — RecoMart Pipeline
**Task 8 | DMML Assignment 1 | Group 37**
Generated: 2026-07-07 05:17 UTC

---

## Tool: DVC (Data Version Control)

DVC extends Git to version large data files and ML models.
It stores file metadata (`.dvc` files) in Git while keeping
actual data in a separate remote storage.

## Repository Structure

```
recomart/
├── .git/                    ← Git repository
├── .dvc/                    ← DVC configuration
│   ├── config               ← remote storage config
│   └── cache/               ← local DVC cache
├── .dvc_remote/             ← local remote storage
├── .dvcignore               ← files DVC should ignore
└── [tracked files].dvc      ← DVC pointer files
```

## Files Tracked by DVC

| File/Folder | Description | Stage |
|-------------|-------------|-------|
| `data/raw/interactions/ratings_20260707_025900.csv` | Raw user-item interactions (sampled) | Ingestion |
| `data/raw/products/products_20260707_025959.csv` | Raw product catalog | Ingestion |
| `data/processed` | Cleaned datasets | Preparation |
| `feature_store/recomart_features.db` | Feature store SQLite DB | Feature Engineering |

## Versioning Workflow

```bash
# 1. Initialize (one-time)
git init && dvc init
dvc remote add -d localremote .dvc_remote/

# 2. Track data files after each pipeline run
dvc add data/raw/interactions/ratings_*.csv
dvc add data/raw/products/products_*.csv
dvc add data/processed/
dvc add feature_store/recomart_features.db

# 3. Commit .dvc pointer files to Git
git add . && git commit -m "Update dataset v1.0"

# 4. Push data to remote
dvc push

# 5. Reproduce any version
git checkout <commit-hash>
dvc pull
```

## Pipeline Lineage

| Stage | Task | Script | Key Transformation |
|-------|------|--------|-------------------|
| Ingestion | T2 | ingest_data.py | JSONL → CSV, 10K user sample |
| Validation | T4 | validate_data.py | 12 pandas + 18 GE checks |
| Preparation | T5 | eda_notebook.ipynb | Dedup, impute, normalize |
| Feature Eng. | T6 | feature_engineering.py | 19 features → SQLite |
| Feature Store | T7 | feature_store.py | Versioned retrieval API |
| Model Training | T9 | train_model.py | SVD + MLflow tracking |
| Orchestration | T10 | pipeline_dag.py | Prefect end-to-end DAG |

## Metadata Tracked per Version

| Metadata field | Value (v1.0) |
|----------------|-------------|
| Dataset | Amazon Reviews 2023 — Video Games |
| Raw interactions | 4,624,615 total → 268,463 sampled |
| After cleaning | 258,524 interactions |
| Users | 10,000 |
| Products (catalog) | 137,269 |
| Products (rated) | 45,914 |
| Features | 19 across 4 tables |
| Sparsity | 99.94% |

Full lineage metadata: `feature_store/pipeline_lineage.json`