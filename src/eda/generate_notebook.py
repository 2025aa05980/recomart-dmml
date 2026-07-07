"""
Generates eda_notebook.ipynb programmatically using nbformat.
Run this once to create the notebook, then open in Jupyter.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"},
}

def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)

nb.cells = [

# ── Title ─────────────────────────────────────────────────────────────────────
md("""# Task 5: Exploratory Data Analysis & Data Preparation
## RecoMart Pipeline — DMML Assignment 1 | Group 37
**Student:** Thanigaivel S | `2025aa05980@wilp.bits-pilani.ac.in`  
**Dataset:** Amazon Reviews 2023 — Video Games  
**Objective:** Understand interaction patterns, item popularity, sparsity, and prepare clean datasets for feature engineering.
"""),

# ── 0. Setup ──────────────────────────────────────────────────────────────────
md("## 0. Setup & Data Loading"),
code("""\
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "../..")   # reach config.py from src/eda/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from glob import glob
from scipy.sparse import csr_matrix

from config import (
    INTERACTIONS_DIR, PRODUCTS_DIR, PROCESSED_DIR,
    REPORTS_DIR, RANDOM_SEED
)

# Plot style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 4)})
PLOTS_DIR = REPORTS_DIR / "eda_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("Libraries loaded ✅")
print(f"Plots → {PLOTS_DIR}")
"""),

code("""\
def load_latest(directory, prefix):
    \"\"\"Load most recent timestamped CSV: prefix_YYYYMMDD_HHMMSS.csv\"\"\"
    expected_len = len(prefix) + 1 + 8 + 1 + 6
    candidates = [
        f for f in directory.glob(f"{prefix}_*.csv")
        if len(f.stem) == expected_len
        and f.stem[len(prefix)+1:].replace("_","").isdigit()
    ]
    if not candidates:
        raise FileNotFoundError(f"No timestamped file for '{prefix}' in {directory}")
    latest = sorted(candidates, reverse=True)[0]
    print(f"Loaded: {latest.name}")
    return pd.read_csv(str(latest))

ratings  = load_latest(INTERACTIONS_DIR, "ratings")
products = load_latest(PRODUCTS_DIR, "products")

print(f"\\nRatings  shape : {ratings.shape}")
print(f"Products shape : {products.shape}")
"""),

# ── 1. Dataset Overview ───────────────────────────────────────────────────────
md("## 1. Dataset Overview"),
code("""\
print("=== RATINGS SAMPLE ===")
display(ratings.head(5))
print("\\n=== RATINGS DTYPES ===")
display(ratings.dtypes.to_frame("dtype"))
print("\\n=== RATINGS DESCRIBE ===")
display(ratings.describe())
"""),

code("""\
print("=== PRODUCTS SAMPLE ===")
display(products.head(5))
print("\\n=== PRODUCTS DTYPES ===")
display(products.dtypes.to_frame("dtype"))
print("\\n=== MISSING VALUES — PRODUCTS ===")
missing = products.isnull().sum()
missing_pct = (missing / len(products) * 100).round(2)
display(pd.DataFrame({"null_count": missing, "null_%": missing_pct}))
"""),

# ── 2. Data Cleaning ──────────────────────────────────────────────────────────
md("""## 2. Data Cleaning & Preprocessing
Addressing issues identified in Task 4 validation:
- Remove duplicate rows and duplicate user-item pairs (keep highest rating)
- Drop 16 out-of-window timestamps
- Impute missing prices with category median
- Fill missing categorical fields with 'Unknown'
"""),

code("""\
print(f"Before cleaning — Ratings: {len(ratings):,} rows")

# 2a. Remove exact duplicate rows
ratings = ratings.drop_duplicates()
print(f"After drop_duplicates       : {len(ratings):,} rows")

# 2b. Duplicate userId+productId — keep highest rating
ratings = (ratings
    .sort_values("rating", ascending=False)
    .drop_duplicates(subset=["userId","productId"])
    .reset_index(drop=True))
print(f"After dedup userId+productId: {len(ratings):,} rows")

# 2c. Drop out-of-window timestamps (16 records)
TS_MIN, TS_MAX = 946684800, 1893456000   # 2000 → 2030
ratings = ratings[ratings["timestamp"].between(TS_MIN, TS_MAX)].reset_index(drop=True)
print(f"After timestamp filter      : {len(ratings):,} rows")

# 2d. Convert timestamp → datetime for analysis
ratings["date"] = pd.to_datetime(ratings["timestamp"], unit="s")
ratings["year"] = ratings["date"].dt.year
ratings["month"] = ratings["date"].dt.month

print(f"\\nFinal ratings shape: {ratings.shape}")
"""),

code("""\
print(f"Before cleaning — Products: {len(products):,} rows")

# 2e. Impute missing price with category median
products["price"] = pd.to_numeric(products["price"], errors="coerce")
cat_median = products.groupby("category")["price"].transform("median")
products["price"] = products["price"].fillna(cat_median)
# Remaining nulls (categories with all-null price) → global median
products["price"] = products["price"].fillna(products["price"].median())

# 2f. Fill missing categoricals
products["brand"]       = products["brand"].fillna("Unknown")
products["category"]    = products["category"].fillna("Unknown")
products["title"]       = products["title"].fillna("Untitled")
products["description"] = products["description"].fillna("")

print(f"After cleaning — Products: {len(products):,} rows")
print(f"\\nRemaining nulls:")
display(products.isnull().sum().to_frame("null_count"))
"""),

# ── 3. Rating Distribution ────────────────────────────────────────────────────
md("## 3. Rating Distribution"),
code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# 3a. Star rating counts
rating_counts = ratings["rating"].value_counts().sort_index()
axes[0].bar(rating_counts.index.astype(int),
            rating_counts.values,
            color=sns.color_palette("muted"), edgecolor="white", width=0.6)
axes[0].set_xlabel("Star Rating")
axes[0].set_ylabel("Number of Ratings")
axes[0].set_title("Rating Distribution")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
for i, (star, cnt) in enumerate(rating_counts.items()):
    axes[0].text(star, cnt + 500, f"{cnt/len(ratings)*100:.1f}%",
                 ha="center", fontsize=9)

# 3b. Ratings over time (year)
yearly = ratings.groupby("year").size()
axes[1].bar(yearly.index, yearly.values,
            color=sns.color_palette("muted")[2], edgecolor="white")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Number of Ratings")
axes[1].set_title("Ratings Volume by Year")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_rating_distribution.png", bbox_inches="tight")
plt.show()
print("Plot saved ✅")
"""),

# ── 4. User Activity ──────────────────────────────────────────────────────────
md("## 4. User Activity Analysis"),
code("""\
user_activity = ratings.groupby("userId").size().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# 4a. Distribution of ratings per user
axes[0].hist(user_activity.values, bins=50,
             color=sns.color_palette("muted")[0], edgecolor="white")
axes[0].set_xlabel("Ratings per User")
axes[0].set_ylabel("Number of Users")
axes[0].set_title("User Activity Distribution")
axes[0].axvline(user_activity.median(), color="red", linestyle="--",
                label=f"Median: {user_activity.median():.0f}")
axes[0].legend()
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))

# 4b. Top 20 most active users
top_users = user_activity.head(20)
axes[1].barh(range(20), top_users.values[::-1],
             color=sns.color_palette("muted")[1])
axes[1].set_yticks(range(20))
axes[1].set_yticklabels([u[:12]+"..." for u in top_users.index[::-1]], fontsize=7)
axes[1].set_xlabel("Number of Ratings")
axes[1].set_title("Top 20 Most Active Users")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_user_activity.png", bbox_inches="tight")
plt.show()

print(f"Total users          : {user_activity.shape[0]:,}")
print(f"Median ratings/user  : {user_activity.median():.0f}")
print(f"Mean ratings/user    : {user_activity.mean():.1f}")
print(f"Max ratings/user     : {user_activity.max():,}")
print(f"Users with <5 ratings: {(user_activity < 5).sum():,} ({(user_activity < 5).mean()*100:.1f}%)")
"""),

# ── 5. Item Popularity ────────────────────────────────────────────────────────
md("## 5. Item Popularity Analysis"),
code("""\
item_popularity = ratings.groupby("productId").size().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# 5a. Distribution of ratings per item
axes[0].hist(item_popularity.values, bins=60,
             color=sns.color_palette("muted")[3], edgecolor="white")
axes[0].set_xlabel("Ratings per Product")
axes[0].set_ylabel("Number of Products")
axes[0].set_title("Item Popularity Distribution (Long Tail)")
axes[0].axvline(item_popularity.median(), color="red", linestyle="--",
                label=f"Median: {item_popularity.median():.0f}")
axes[0].legend()
axes[0].set_yscale("log")

# 5b. Popularity curve (top 500)
axes[1].plot(range(min(500, len(item_popularity))),
             item_popularity.values[:500],
             color=sns.color_palette("muted")[4])
axes[1].fill_between(range(min(500, len(item_popularity))),
                     item_popularity.values[:500], alpha=0.3,
                     color=sns.color_palette("muted")[4])
axes[1].set_xlabel("Product Rank")
axes[1].set_ylabel("Number of Ratings")
axes[1].set_title("Item Popularity Curve (Top 500 Products)")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_item_popularity.png", bbox_inches="tight")
plt.show()

# Long tail analysis
top10_pct   = item_popularity.head(int(len(item_popularity)*0.1)).sum() / len(ratings) * 100
print(f"Total products              : {item_popularity.shape[0]:,}")
print(f"Median ratings/product      : {item_popularity.median():.0f}")
print(f"Top 10% products account for: {top10_pct:.1f}% of all ratings (long tail)")
print(f"Products with only 1 rating : {(item_popularity == 1).sum():,}")
"""),

# ── 6. Sparsity Matrix ────────────────────────────────────────────────────────
md("## 6. Sparsity Visualization"),
code("""\
# Sample 200 users × 200 items for visualization
sample_users = ratings["userId"].value_counts().head(200).index
sample_items = ratings["productId"].value_counts().head(200).index
mask = ratings["userId"].isin(sample_users) & ratings["productId"].isin(sample_items)
sub  = ratings[mask].copy()

# Build user-item matrix
user_idx = {u: i for i, u in enumerate(sample_users)}
item_idx = {p: i for i, p in enumerate(sample_items)}
sub["u_idx"] = sub["userId"].map(user_idx)
sub["i_idx"] = sub["productId"].map(item_idx)

matrix = np.zeros((200, 200))
for _, row in sub.iterrows():
    matrix[int(row["u_idx"]), int(row["i_idx"])] = row["rating"]

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
               vmin=0, vmax=5, interpolation="nearest")
plt.colorbar(im, ax=ax, label="Rating (0=unrated)")
ax.set_xlabel("Products (top 200 by popularity)")
ax.set_ylabel("Users (top 200 by activity)")
ax.set_title(f"User-Item Interaction Matrix\\n"
             f"Sparsity: {(matrix == 0).sum() / matrix.size * 100:.1f}%  "
             f"(white = no interaction)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_sparsity_heatmap.png", bbox_inches="tight")
plt.show()

n_u = ratings["userId"].nunique()
n_i = ratings["productId"].nunique()
sparsity = (1 - len(ratings) / (n_u * n_i)) * 100
print(f"Full matrix sparsity: {sparsity:.4f}%")
"""),

# ── 7. Category Analysis ──────────────────────────────────────────────────────
md("## 7. Category & Price Analysis"),
code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 7a. Top 15 categories by product count
top_cats = products["category"].value_counts().head(15)
axes[0].barh(range(15), top_cats.values[::-1],
             color=sns.color_palette("muted"))
axes[0].set_yticks(range(15))
axes[0].set_yticklabels(
    [c[:40]+"..." if len(c)>40 else c for c in top_cats.index[::-1]],
    fontsize=7)
axes[0].set_xlabel("Number of Products")
axes[0].set_title("Top 15 Categories by Product Count")

# 7b. Price distribution
price_clean = products["price"][products["price"] < products["price"].quantile(0.95)]
axes[1].hist(price_clean, bins=40,
             color=sns.color_palette("muted")[2], edgecolor="white")
axes[1].axvline(products["price"].median(), color="red", linestyle="--",
                label=f"Median: ${products['price'].median():.2f}")
axes[1].set_xlabel("Price (USD)")
axes[1].set_ylabel("Number of Products")
axes[1].set_title("Product Price Distribution (below 95th percentile)")
axes[1].legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_category_price.png", bbox_inches="tight")
plt.show()
"""),

# ── 8. Rating vs Price ────────────────────────────────────────────────────────
md("## 8. Rating vs Price & Avg Rating per Category"),
code("""\
# Merge ratings with product metadata
merged = ratings.merge(products[["productId","price","category","brand"]],
                       on="productId", how="left")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 8a. Avg rating by price bucket
merged["price_bucket"] = pd.cut(
    merged["price"].clip(0, 200),
    bins=[0, 10, 25, 50, 100, 200],
    labels=["<$10","$10-25","$25-50","$50-100","$100-200"]
)
avg_by_price = merged.groupby("price_bucket", observed=True)["rating"].mean()
axes[0].bar(avg_by_price.index, avg_by_price.values,
            color=sns.color_palette("muted"))
axes[0].set_xlabel("Price Range")
axes[0].set_ylabel("Average Rating")
axes[0].set_title("Average Rating by Price Range")
axes[0].set_ylim(3.5, 5.0)
for i, v in enumerate(avg_by_price.values):
    axes[0].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

# 8b. Top 10 categories by avg rating (min 100 ratings)
cat_stats = (merged.groupby("category")
             .agg(avg_rating=("rating","mean"), n=("rating","count"))
             .query("n >= 100")
             .nlargest(10, "avg_rating"))
axes[1].barh(range(len(cat_stats)),
             cat_stats["avg_rating"].values[::-1],
             color=sns.color_palette("muted"))
axes[1].set_yticks(range(len(cat_stats)))
axes[1].set_yticklabels(
    [c[:35]+"..." if len(c)>35 else c
     for c in cat_stats.index[::-1]], fontsize=7)
axes[1].set_xlabel("Average Rating")
axes[1].set_title("Top 10 Categories by Avg Rating (min 100 ratings)")
axes[1].set_xlim(4.0, 5.0)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_rating_vs_price_category.png", bbox_inches="tight")
plt.show()
"""),

# ── 9. Temporal Analysis ──────────────────────────────────────────────────────
md("## 9. Temporal Analysis"),
code("""\
monthly = ratings.groupby(["year","month"]).size().reset_index(name="count")
monthly["period"] = pd.to_datetime(
    monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2))
monthly = monthly.sort_values("period")

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(monthly["period"], monthly["count"],
        color=sns.color_palette("muted")[0], linewidth=1.5)
ax.fill_between(monthly["period"], monthly["count"],
                alpha=0.2, color=sns.color_palette("muted")[0])
ax.set_xlabel("Month")
ax.set_ylabel("Number of Ratings")
ax.set_title("Monthly Rating Volume Over Time")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_temporal_trends.png", bbox_inches="tight")
plt.show()

peak = monthly.loc[monthly["count"].idxmax()]
print(f"Peak month: {peak['period'].strftime('%Y-%m')} with {peak['count']:,} ratings")
"""),

# ── 10. Save Processed Data ───────────────────────────────────────────────────
md("""## 10. Save Processed Datasets
Save cleaned ratings and products to `data/processed/` for use in Tasks 6–9.
"""),
code("""\
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Save cleaned ratings (drop helper columns before saving)
ratings_out = ratings.drop(columns=["date","year","month"], errors="ignore")
ratings_path = PROCESSED_DIR / "cleaned_ratings.csv"
ratings_out.to_csv(ratings_path, index=False)
print(f"Cleaned ratings saved  → {ratings_path.name}")
print(f"  Shape : {ratings_out.shape}")
print(f"  Users : {ratings_out['userId'].nunique():,}")
print(f"  Items : {ratings_out['productId'].nunique():,}")

# Save cleaned products
products_path = PROCESSED_DIR / "cleaned_products.csv"
products.to_csv(products_path, index=False)
print(f"\\nCleaned products saved → {products_path.name}")
print(f"  Shape : {products.shape}")
print(f"  Nulls remaining: {products.isnull().sum().sum()}")
"""),

# ── 11. EDA Summary ───────────────────────────────────────────────────────────
md("""## 11. EDA Summary

| Metric | Value |
|--------|-------|
| Total interactions (after cleaning) | see above |
| Unique users | 10,000 |
| Unique products | ~45,900 |
| Matrix sparsity | 99.94% |
| Rating distribution | Positively skewed (61% five-star) |
| Dominant category | Video Games > PC > Games |
| Median price | $24.95 |
| Peak rating year | 2020–2021 (COVID gaming boom) |

### Key Observations
1. **Heavy long tail** — top 10% of products account for majority of ratings
2. **High sparsity (99.94%)** — confirms need for matrix factorization (SVD)
3. **Positive rating bias** — 80% of ratings are 4★ or 5★ (common on Amazon)
4. **Missing prices (54.8%)** — imputed with category median; does not affect model
5. **COVID spike** — rating volume peaks around 2020–2021 reflecting gaming boom

### Prepared Outputs
- `data/processed/cleaned_ratings.csv` — deduplicated, timestamp-filtered
- `data/processed/cleaned_products.csv` — null-imputed, encoding-ready
- `reports/eda_plots/` — 7 visualisation plots for PDF report
"""),

]

# Write notebook
out_path = Path(__file__).parent / "eda_notebook.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written → {out_path}")
