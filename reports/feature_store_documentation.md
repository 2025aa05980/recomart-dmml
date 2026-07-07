# Feature Store Documentation — RecoMart Pipeline
**Task 7 | DMML Assignment 1 | Group 37**
Generated: 2026-07-07 04:59 UTC

---

## Overview

The RecoMart feature store is implemented using:
- **SQLite** backend (`recomart_features.db`) for feature data storage
- **JSON metadata registry** (`feature_metadata.json`) for feature documentation
- **Version table** in SQLite for tracking feature versions
- **Python API** (`feature_store.py`) for retrieval at training and inference time

## Feature Store API

| Function | Purpose | Use case |
|----------|---------|----------|
| `register_version()` | Log feature version with metadata | After each pipeline run |
| `list_versions()` | Show all registered versions | Audit, reproducibility |
| `get_user_features()` | Retrieve user features by ID | Training + inference |
| `get_item_features()` | Retrieve item features by ID | Training + inference |
| `get_training_set()` | Joined features for model training | Task 9 model training |
| `get_inference_features()` | User × candidate items for scoring | Real-time recommendation |
| `get_cooccurrence()` | Top-N co-occurring items | Item-based recommendation |
| `get_feature_stats()` | Descriptive stats for any table | Monitoring, debugging |
| `get_feature_metadata()` | Feature definitions from registry | Documentation, lineage |

## Registered Versions

|   version_id | version_tag   | feature_set                                                   |   n_users |   n_items |   n_interactions | dataset                           | created_at                       | notes                                                            |
|-------------:|:--------------|:--------------------------------------------------------------|----------:|----------:|-----------------:|:----------------------------------|:---------------------------------|:-----------------------------------------------------------------|
|            1 | v1.0          | user_features,item_features,interaction_features,cooccurrence |     10000 |     45914 |           258524 | Amazon Reviews 2023 — Video Games | 2026-07-07T04:59:46.315810+00:00 | Initial feature set — Amazon Video Games 2023, 10K users sampled |

## Training Set Schema

Shape: 25,852 rows × 19 features

| Column | Source | Description |
|--------|--------|-------------|
| userId | interaction | Amazon reviewer ID |
| productId | interaction | Amazon ASIN |
| rating | interaction | Raw star rating (1–5) |
| rating_normalized | interaction | Rating scaled to [0,1] |
| days_since_epoch | interaction | Timestamp in days |
| user_item_rating_diff | interaction | User rating minus item avg |
| user_rating_count | user | Total ratings given |
| user_avg_rating | user | Mean rating given |
| user_rating_std | user | Rating consistency |
| user_unique_items | user | Distinct products rated |
| user_recency_days | user | Days since last rating |
| user_category_diversity | user | Categories explored |
| item_rating_count | item | Total ratings received |
| item_avg_rating | item | Mean rating received |
| item_rating_std | item | Rating controversy score |
| item_unique_users | item | Reach |
| price_normalized | item | MinMaxScaled price [0,1] |
| category_encoded | item | LabelEncoded category |
| brand_encoded | item | LabelEncoded brand |

## Inference Flow

```
user_id → get_inference_features(user_id)
        → joins user_features × top-100 candidate items
        → returns DataFrame ready for model.predict()
        → rank by predicted_rating → top-K recommendations
```

Sample inference shape: (100, 18) (1 user × 100 candidate items × 18 features)

## Co-occurrence Sample

| co_productId   |   cooc_score |
|:---------------|-------------:|
| B0000296O5     |           42 |
| B00004Y57G     |           41 |
| B08VFQ3XJX     |           39 |

## Versioning Strategy

Each pipeline run registers a new version with:
- Unique version tag (e.g. v1.0, v1.1)
- Row counts for all feature tables
- Dataset name and creation timestamp
- Human-readable notes describing changes

This enables reproducible training — any historical version
can be reconstructed by re-running the pipeline with the
same source data snapshot (tracked via DVC in Task 8).