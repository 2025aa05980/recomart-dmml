# Feature Engineering Summary — RecoMart Pipeline
**Task 6 | DMML Assignment 1 | Group 37**
Generated: 2026-07-10 02:55 UTC

---

## Feature Tables

| Table | Rows | Features | Storage |
|-------|------|----------|---------|
| user_features | 10,000 | 6 | SQLite |
| item_features | 45,914 | 12 | SQLite |
| interaction_features | 258,524 | 5 | SQLite |
| cooccurrence | 1,500 | 3 | SQLite |

## User Features

| Feature | Logic | Rationale |
|---------|-------|-----------|
| user_rating_count | COUNT(ratings) per user | Activity level — power vs casual users |
| user_avg_rating | MEAN(rating) per user | Preference tendency — optimist vs critic |
| user_rating_std | STD(rating) per user | Consistency — polarized vs moderate rater |
| user_unique_items | NUNIQUE(productId) per user | Breadth of engagement |
| user_recency_days | (max_ts - last_ts) / 86400 | Engagement recency — lower = more active |
| user_category_diversity | NUNIQUE(category) per user | Cross-category interest |

## Item Features

| Feature | Source | Logic | Rationale |
|---------|--------|-------|-----------|
| item_rating_count | ratings (sampled) | COUNT(ratings) per item | Popularity from 10K user sample |
| item_avg_rating | ratings (sampled) | MEAN(rating) per item | Quality signal from sampled users |
| item_rating_std | ratings (sampled) | STD(rating) per item | Controversy score — polarising items |
| item_unique_users | ratings (sampled) | NUNIQUE(userId) per item | Reach from sampled users |
| price_normalized | products (JSONL) | MinMaxScaler(price) → [0,1] | Normalized for model input |
| category_encoded | products (JSONL) | LabelEncoder(category) | Categorical → numeric for ML |
| brand_encoded | products (JSONL) | LabelEncoder(brand) | Categorical → numeric for ML |
| api_rating_count | REST API (Source 2) | rating_count from API | Full catalog popularity (all Amazon users) |
| api_avg_rating | REST API (Source 2) | avg_rating from API | Full catalog quality signal (all Amazon users) |

## Interaction Features

| Feature | Logic | Rationale |
|---------|-------|-----------|
| rating_normalized | (rating - 1) / 4 → [0,1] | Normalized rating for model |
| days_since_epoch | timestamp / 86400 | Temporal feature (days) |
| user_item_rating_diff | rating - item_avg_rating | Relative sentiment above/below crowd |

## Co-occurrence Features

Item-item co-occurrence is computed as the number of shared users
who rated both items. Built from the top 500 most-rated items
using a binary user-item pivot matrix and dot product.
Used for item-based collaborative filtering and 'customers also bought' recommendations.

## Encoders Applied (Task 5 gap closure)

| Column | Encoder | Output |
|--------|---------|--------|
| category | sklearn LabelEncoder | Integer index 0..N_categories |
| brand | sklearn LabelEncoder | Integer index 0..N_brands |
| price | sklearn MinMaxScaler | Float in [0.0, 1.0] |
| rating | Manual scaling (r-1)/4 | Float in [0.0, 1.0] |

## SQL Schema (SQLite)

```sql
-- user_features
CREATE TABLE user_features (
    userId                  TEXT PRIMARY KEY,
    user_rating_count       INTEGER,
    user_avg_rating         REAL,
    user_rating_std         REAL,
    user_unique_items       INTEGER,
    user_recency_days       REAL,
    user_category_diversity INTEGER
);

-- item_features
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
    price_normalized    REAL
);

-- interaction_features
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

-- cooccurrence
CREATE TABLE cooccurrence (
    productId       TEXT,
    co_productId    TEXT,
    cooc_score      REAL,
    PRIMARY KEY (productId, co_productId)
);
```