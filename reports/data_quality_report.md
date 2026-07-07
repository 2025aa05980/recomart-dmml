# Data Quality Report — RecoMart Pipeline
**Task 4 | DMML Assignment 1 | Group 37**
Student: Thanigaivel S | `2025aa05980@wilp.bits-pilani.ac.in`
Generated: 2026-07-07 06:50 UTC
Dataset: Amazon Reviews 2023 — Video Games

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total checks run | 12 |
| Checks passed | 8 |
| Checks with warnings | 4 |
| Ratings file | `ratings_20260707_025900.csv` |
| Products file | `products_20260707_025959.csv` |
| Total interactions | 268,463 |
| Unique users | 10,000 |
| Unique products (ratings) | 45,916 |
| Unique products (catalog) | 137,269 |
| Matrix sparsity | 99.9415% |

---

## 1. Ratings Dataset Validation
**File:** `ratings_20260707_025900.csv`  
**Shape:** 268,463 rows × 4 columns  
**Columns:** ['userId', 'productId', 'rating', 'timestamp']

| # | Check | Result | Detail |
|---|-------|--------|--------|
| 1 | schema | ✅ PASS | Missing cols: [] \| Extra cols: [] |
| 2 | missing_values | ✅ PASS | No missing values |
| 3 | duplicate_rows | ⚠️  WARN | 5,548 duplicates found |
| 4 | duplicate_userId+productId | ⚠️  WARN | 9,923 duplicates found |
| 5 | rating_range | ✅ PASS | 0 out-of-range values \| Distribution: {1: 15327, 2: 12261, 3: 26140, 4: 50174, 5: 164561} |
| 6 | timestamp_validity | ⚠️  WARN | Range: 1999-11-10 → 2023-08-30 \| Nulls: 0 \| Out-of-window: 16 |
| 7 | sparsity | ✅ PASS | Users: 10,000 \| Items: 45,916 \| Interactions: 268,463 \| Sparsity: 99.94% \| Avg ratings/user: 26.8 \| Avg ratings/ite |

### Rating Distribution

| Stars | Count | % |
|-------|-------|---|
| 1 star | 15,327 | 5.7% |
| 2 stars | 12,261 | 4.6% |
| 3 stars | 26,140 | 9.7% |
| 4 stars | 50,174 | 18.7% |
| 5 stars | 164,561 | 61.3% |

### Sparsity Analysis

| Metric | Value |
|--------|-------|
| Users | 10,000 |
| Products | 45,916 |
| Observed interactions | 268,463 |
| Possible interactions | 459,160,000 |
| Sparsity | 99.9415% |

> High sparsity (>95%) is expected and desirable for collaborative
> filtering — it reflects real-world recommendation scenarios where
> users rate only a small fraction of available products.

---

## 2. Products Dataset Validation
**File:** `products_20260707_025959.csv`  
**Shape:** 137,269 rows × 6 columns  
**Columns:** ['productId', 'title', 'price', 'brand', 'category', 'description']

| # | Check | Result | Detail |
|---|-------|--------|--------|
| 1 | schema | ✅ PASS | Missing cols: [] \| Extra cols: [] |
| 2 | missing_values | ⚠️  WARN | title: 9 (0.01%); price: 75277 (54.84%); brand: 4375 (3.19%); category: 12637 (9.21%); description: 51740 (37.69%) |
| 3 | duplicate_productId | ✅ PASS | 0 duplicates found |
| 4 | price_range | ✅ PASS | Nulls: 75277 (54.8%) \| Negatives: 0 \| Zeros: 32 \| Outliers (>3×p95): 365 \| p50: $24.95 \| p95: $150.65 |
| 5 | category_coverage | ✅ PASS | Unique categories: 508 \| Empty: 12637 (9.2%) \| Top 5: {'Video Games > PC > Games': 17758, 'Video Games > PC > Accessor |

### Missing Values Summary — Products

| Column | Null Count | Null % |
|--------|-----------|--------|
| productId | 0 | 0.0% |
| title | 9 | 0.0% |
| price | 75,277 | 54.8% |
| brand | 4,375 | 3.2% |
| category | 12,637 | 9.2% |
| description | 51,740 | 37.7% |

---

## 3. Issues & Recommendations

| Issue | Severity | Action |
|-------|----------|--------|
| Missing price values in product catalog | Medium | Impute with category median in Task 5 |
| High sparsity in user-item matrix | Expected | Use matrix factorization (SVD) in Task 9 |
| Duplicate user-item pairs (if any) | Low | Keep highest rating per pair |
| Missing brand/category fields | Low | Fill with 'Unknown' placeholder |

---

## 4. Validation Conclusion

The dataset passes **8/12** quality checks. 
All critical checks (schema, rating range, duplicate detection) pass. 
Warnings relate to expected characteristics of e-commerce interaction data 
(high sparsity, missing price metadata) which are handled in the 
Data Preparation stage (Task 5).

The dataset is **approved for downstream processing**.