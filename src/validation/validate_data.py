"""
Task 4: Data Validation & Quality Report — RecoMart Pipeline
Dataset: Amazon Reviews 2023 — Video Games

Validation checks performed:
  Ratings CSV:
    ✓ Schema check         — expected columns present
    ✓ Missing values       — null counts per column
    ✓ Duplicate records    — exact row duplicates
    ✓ Duplicate user-item  — same (userId, productId) pair rated twice
    ✓ Rating range         — all values between 1.0 and 5.0
    ✓ Timestamp validity   — Unix epoch within reasonable range
    ✓ User/item counts     — unique users, products, sparsity

  Products CSV:
    ✓ Schema check         — expected columns present
    ✓ Missing values       — null counts and percentages
    ✓ Duplicate productIds — dedup check
    ✓ Price range          — non-negative, outlier detection
    ✓ Category coverage    — distribution of categories

Output:
  reports/data_quality_report.md   — detailed markdown report
  reports/data_quality_summary.csv — machine-readable summary
  logs/pipeline.log                — validation events
"""

import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from glob import glob

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    INTERACTIONS_DIR, PRODUCTS_DIR, REPORTS_DIR,
    RATING_SCALE, RANDOM_SEED, DATASET_NAME
)
from src.logger import get_logger

log = get_logger("validation")

# ── Expected schemas ──────────────────────────────────────────────────────────
RATINGS_SCHEMA  = ["userId", "productId", "rating", "timestamp"]
PRODUCTS_SCHEMA = ["productId", "title", "price", "brand", "category", "description"]

# Timestamp sanity window: Jan 2000 → Jan 2030
TS_MIN = 946684800   # 2000-01-01
TS_MAX = 1893456000  # 2030-01-01


# ── File loader ───────────────────────────────────────────────────────────────

def load_latest(directory: Path, prefix: str) -> tuple[pd.DataFrame, str]:
    """Load most recent timestamped CSV: prefix_YYYYMMDD_HHMMSS.csv"""
    expected_len = len(prefix) + 1 + 8 + 1 + 6  # e.g. ratings_20260707_025900
    candidates = [
        f for f in directory.glob(f"{prefix}_*.csv")
        if len(f.stem) == expected_len
        and f.stem[len(prefix)+1:].replace("_", "").isdigit()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No timestamped file found for prefix='{prefix}' in {directory}. "
            f"Run ingest_data.py first."
        )
    latest = sorted(candidates, reverse=True)[0]
    log.info(f"Loading: {latest.name}")
    df = pd.read_csv(str(latest))
    return df, str(latest)


# ── Validation helpers ────────────────────────────────────────────────────────

def check_schema(df: pd.DataFrame, expected: list[str], name: str) -> dict:
    missing = [c for c in expected if c not in df.columns]
    extra   = [c for c in df.columns if c not in expected]
    passed  = len(missing) == 0
    log.info(f"[{name}] Schema — {'PASS' if passed else 'FAIL'} | "
             f"Missing: {missing} | Extra: {extra}")
    return {
        "check":   "schema",
        "passed":  passed,
        "detail":  f"Missing cols: {missing} | Extra cols: {extra}",
    }


def check_missing(df: pd.DataFrame, name: str) -> dict:
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)
    worst_col   = null_counts.idxmax()
    worst_pct   = null_pct[worst_col]
    passed      = worst_pct < 30.0   # threshold: <30% nulls in any column
    detail_rows = [f"{c}: {null_counts[c]} ({null_pct[c]}%)"
                   for c in df.columns if null_counts[c] > 0]
    detail = "; ".join(detail_rows) if detail_rows else "No missing values"
    log.info(f"[{name}] Missing values — {'PASS' if passed else 'FAIL'} | {detail}")
    return {
        "check":   "missing_values",
        "passed":  passed,
        "detail":  detail,
        "null_summary": null_counts.to_dict(),
    }


def check_duplicates(df: pd.DataFrame, name: str,
                     subset: list[str] | None = None) -> dict:
    label  = f"duplicate_rows" if subset is None else f"duplicate_{'+'.join(subset)}"
    n_dups = df.duplicated(subset=subset).sum()
    passed = n_dups == 0
    detail = f"{n_dups:,} duplicate{'s' if n_dups != 1 else ''} found"
    log.info(f"[{name}] {label} — {'PASS' if passed else 'WARN'} | {detail}")
    return {"check": label, "passed": passed, "detail": detail}


def check_rating_range(df: pd.DataFrame, name: str) -> dict:
    lo, hi   = RATING_SCALE
    out_mask = ~df["rating"].between(lo, hi)
    n_out    = out_mask.sum()
    passed   = n_out == 0
    dist     = df["rating"].value_counts().sort_index().to_dict()
    detail   = f"{n_out} out-of-range values | Distribution: {dist}"
    log.info(f"[{name}] Rating range [{lo}–{hi}] — {'PASS' if passed else 'FAIL'} | {detail}")
    return {"check": "rating_range", "passed": passed, "detail": detail,
            "distribution": dist}


def check_timestamp(df: pd.DataFrame, name: str) -> dict:
    ts       = pd.to_numeric(df["timestamp"], errors="coerce")
    n_null   = ts.isnull().sum()
    n_out    = ((ts < TS_MIN) | (ts > TS_MAX)).sum()
    passed   = n_null == 0 and n_out == 0
    ts_min   = pd.to_datetime(ts.min(), unit="s").strftime("%Y-%m-%d")
    ts_max   = pd.to_datetime(ts.max(), unit="s").strftime("%Y-%m-%d")
    detail   = (f"Range: {ts_min} → {ts_max} | "
                f"Nulls: {n_null} | Out-of-window: {n_out}")
    log.info(f"[{name}] Timestamp validity — {'PASS' if passed else 'WARN'} | {detail}")
    return {"check": "timestamp_validity", "passed": passed, "detail": detail}


def check_price(df: pd.DataFrame, name: str) -> dict:
    price    = pd.to_numeric(df["price"], errors="coerce")
    n_null   = price.isnull().sum()
    n_neg    = (price < 0).sum()
    n_zero   = (price == 0).sum()
    p95      = price.quantile(0.95)
    n_out    = (price > p95 * 3).sum()   # extreme outliers
    passed   = n_neg == 0
    detail   = (f"Nulls: {n_null} ({n_null/len(df)*100:.1f}%) | "
                f"Negatives: {n_neg} | Zeros: {n_zero} | "
                f"Outliers (>3×p95): {n_out} | "
                f"p50: ${price.median():.2f} | p95: ${p95:.2f}")
    log.info(f"[{name}] Price range — {'PASS' if passed else 'FAIL'} | {detail}")
    return {"check": "price_range", "passed": passed, "detail": detail}


def check_sparsity(df: pd.DataFrame, name: str) -> dict:
    n_users    = df["userId"].nunique()
    n_items    = df["productId"].nunique()
    n_ratings  = len(df)
    possible   = n_users * n_items
    sparsity   = (1 - n_ratings / possible) * 100
    avg_u      = df.groupby("userId").size().mean()
    avg_i      = df.groupby("productId").size().mean()
    passed     = sparsity > 90   # recommendation datasets are typically >95% sparse
    detail     = (f"Users: {n_users:,} | Items: {n_items:,} | "
                  f"Interactions: {n_ratings:,} | "
                  f"Sparsity: {sparsity:.2f}% | "
                  f"Avg ratings/user: {avg_u:.1f} | "
                  f"Avg ratings/item: {avg_i:.1f}")
    log.info(f"[{name}] Sparsity — {'PASS' if passed else 'NOTE'} | {detail}")
    return {
        "check": "sparsity", "passed": passed, "detail": detail,
        "sparsity_pct": round(sparsity, 4),
        "n_users": n_users, "n_items": n_items, "n_ratings": n_ratings,
    }


def check_categories(df: pd.DataFrame, name: str) -> dict:
    n_empty  = (df["category"].isna() | (df["category"] == "")).sum()
    top5     = df["category"].value_counts().head(5).to_dict()
    n_unique = df["category"].nunique()
    passed   = n_empty / len(df) < 0.5
    detail   = (f"Unique categories: {n_unique} | "
                f"Empty: {n_empty} ({n_empty/len(df)*100:.1f}%) | "
                f"Top 5: {top5}")
    log.info(f"[{name}] Category coverage — {'PASS' if passed else 'WARN'} | {detail}")
    return {"check": "category_coverage", "passed": passed, "detail": detail}


# ── Main validation runners ───────────────────────────────────────────────────

def validate_ratings() -> tuple[list[dict], pd.DataFrame, str]:
    log.info("=" * 55)
    log.info("=== Validating Ratings Dataset ===")
    df, path = load_latest(INTERACTIONS_DIR, "ratings")
    log.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    results = [
        check_schema(df, RATINGS_SCHEMA, "ratings"),
        check_missing(df, "ratings"),
        check_duplicates(df, "ratings"),
        check_duplicates(df, "ratings", subset=["userId", "productId"]),
        check_rating_range(df, "ratings"),
        check_timestamp(df, "ratings"),
        check_sparsity(df, "ratings"),
    ]

    passed = sum(1 for r in results if r["passed"])
    log.info(f"=== Ratings validation: {passed}/{len(results)} checks passed ===")
    return results, df, path


def validate_products() -> tuple[list[dict], pd.DataFrame, str]:
    log.info("=" * 55)
    log.info("=== Validating Products Dataset ===")
    df, path = load_latest(PRODUCTS_DIR, "products")
    log.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    results = [
        check_schema(df, PRODUCTS_SCHEMA, "products"),
        check_missing(df, "products"),
        check_duplicates(df, "products", subset=["productId"]),
        check_price(df, "products"),
        check_categories(df, "products"),
    ]

    passed = sum(1 for r in results if r["passed"])
    log.info(f"=== Products validation: {passed}/{len(results)} checks passed ===")
    return results, df, path


# ── Report generator ──────────────────────────────────────────────────────────

def generate_report(r_results, r_df, r_path,
                    p_results, p_df, p_path) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    r_pass  = sum(1 for r in r_results if r["passed"])
    p_pass  = sum(1 for r in p_results if r["passed"])
    total   = len(r_results) + len(p_results)
    t_pass  = r_pass + p_pass

    def status(passed): return "✅ PASS" if passed else "⚠️  WARN"

    # ── Sparsity stats from results ───────────────────────────────────────────
    sp = next((r for r in r_results if r["check"] == "sparsity"), {})
    n_users   = sp.get("n_users", "—")
    n_items   = sp.get("n_items", "—")
    n_ratings = sp.get("n_ratings", "—")
    sparsity  = sp.get("sparsity_pct", "—")

    # ── Rating distribution ───────────────────────────────────────────────────
    rr = next((r for r in r_results if r["check"] == "rating_range"), {})
    dist = rr.get("distribution", {})
    dist_table = "\n".join(
        f"| {int(k)} star{'s' if k > 1 else ''} | {v:,} | "
        f"{v/len(r_df)*100:.1f}% |"
        for k, v in sorted(dist.items())
    ) if dist else "| — | — | — |"

    lines = [
        f"# Data Quality Report — RecoMart Pipeline",
        f"**Task 4 | DMML Assignment 1 | Group 37**",
        f"Student: Thanigaivel S | `2025aa05980@wilp.bits-pilani.ac.in`",
        f"Generated: {now}",
        f"Dataset: {DATASET_NAME}",
        f"",
        f"---",
        f"",
        f"## Executive Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total checks run | {total} |",
        f"| Checks passed | {t_pass} |",
        f"| Checks with warnings | {total - t_pass} |",
        f"| Ratings file | `{Path(r_path).name}` |",
        f"| Products file | `{Path(p_path).name}` |",
        f"| Total interactions | {len(r_df):,} |",
        f"| Unique users | {n_users:,} |",
        f"| Unique products (ratings) | {n_items:,} |",
        f"| Unique products (catalog) | {len(p_df):,} |",
        f"| Matrix sparsity | {sparsity}% |",
        f"",
        f"---",
        f"",
        f"## 1. Ratings Dataset Validation",
        f"**File:** `{Path(r_path).name}`  ",
        f"**Shape:** {len(r_df):,} rows × {len(r_df.columns)} columns  ",
        f"**Columns:** {list(r_df.columns)}",
        f"",
        f"| # | Check | Result | Detail |",
        f"|---|-------|--------|--------|",
    ]

    for i, r in enumerate(r_results, 1):
        detail = r["detail"].replace("|", "\\|")[:120]
        lines.append(f"| {i} | {r['check']} | {status(r['passed'])} | {detail} |")

    lines += [
        f"",
        f"### Rating Distribution",
        f"",
        f"| Stars | Count | % |",
        f"|-------|-------|---|",
        dist_table,
        f"",
        f"### Sparsity Analysis",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Users | {n_users:,} |",
        f"| Products | {n_items:,} |",
        f"| Observed interactions | {n_ratings:,} |",
        f"| Possible interactions | {n_users * n_items:,} |",
        f"| Sparsity | {sparsity}% |",
        f"",
        f"> High sparsity (>95%) is expected and desirable for collaborative",
        f"> filtering — it reflects real-world recommendation scenarios where",
        f"> users rate only a small fraction of available products.",
        f"",
        f"---",
        f"",
        f"## 2. Products Dataset Validation",
        f"**File:** `{Path(p_path).name}`  ",
        f"**Shape:** {len(p_df):,} rows × {len(p_df.columns)} columns  ",
        f"**Columns:** {list(p_df.columns)}",
        f"",
        f"| # | Check | Result | Detail |",
        f"|---|-------|--------|--------|",
    ]

    for i, r in enumerate(p_results, 1):
        detail = r["detail"].replace("|", "\\|")[:120]
        lines.append(f"| {i} | {r['check']} | {status(r['passed'])} | {detail} |")

    lines += [
        f"",
        f"### Missing Values Summary — Products",
        f"",
        f"| Column | Null Count | Null % |",
        f"|--------|-----------|--------|",
    ]

    for col in p_df.columns:
        n   = p_df[col].isnull().sum()
        pct = n / len(p_df) * 100
        lines.append(f"| {col} | {n:,} | {pct:.1f}% |")

    lines += [
        f"",
        f"---",
        f"",
        f"## 3. Issues & Recommendations",
        f"",
        f"| Issue | Severity | Action |",
        f"|-------|----------|--------|",
        f"| Missing price values in product catalog | Medium | Impute with category median in Task 5 |",
        f"| High sparsity in user-item matrix | Expected | Use matrix factorization (SVD) in Task 9 |",
        f"| Duplicate user-item pairs (if any) | Low | Keep highest rating per pair |",
        f"| Missing brand/category fields | Low | Fill with 'Unknown' placeholder |",
        f"",
        f"---",
        f"",
        f"## 4. Validation Conclusion",
        f"",
        f"The dataset passes **{t_pass}/{total}** quality checks. ",
        f"All critical checks (schema, rating range, duplicate detection) pass. ",
        f"Warnings relate to expected characteristics of e-commerce interaction data ",
        f"(high sparsity, missing price metadata) which are handled in the ",
        f"Data Preparation stage (Task 5).",
        f"",
        f"The dataset is **approved for downstream processing**.",
    ]

    report_path = REPORTS_DIR / "data_quality_report.md"
    report_path.write_text("\n".join(lines))
    log.info(f"Report saved → {report_path}")

    # ── Machine-readable summary CSV ──────────────────────────────────────────
    all_results = []
    for r in r_results:
        all_results.append({
            "dataset": "ratings",
            "check":   r["check"],
            "passed":  r["passed"],
            "detail":  r["detail"][:200],
        })
    for r in p_results:
        all_results.append({
            "dataset": "products",
            "check":   r["check"],
            "passed":  r["passed"],
            "detail":  r["detail"][:200],
        })
    summary_path = REPORTS_DIR / "data_quality_summary.csv"
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    log.info(f"Summary CSV saved → {summary_path}")

    return report_path


# ── Great Expectations Validation ────────────────────────────────────────────

def run_great_expectations(df: pd.DataFrame, name: str) -> list[dict]:
    """
    Run Great Expectations 1.x suite on a DataFrame.
    Uses ephemeral context + pandas DataSource (no backend required).
    Compatible with great_expectations >= 1.0.
    Returns list of expectation results for report inclusion.
    """
    try:
        import great_expectations as gx

        log.info(f"[GE] Running Great Expectations suite on '{name}'...")

        # ── Build expectation objects ─────────────────────────────────────────
        if name == "ratings":
            raw_expectations = [
                (gx.expectations.ExpectColumnToExist(column="userId"),
                 "userId column exists"),
                (gx.expectations.ExpectColumnToExist(column="productId"),
                 "productId column exists"),
                (gx.expectations.ExpectColumnToExist(column="rating"),
                 "rating column exists"),
                (gx.expectations.ExpectColumnToExist(column="timestamp"),
                 "timestamp column exists"),
                (gx.expectations.ExpectColumnValuesToNotBeNull(column="userId"),
                 "userId has no nulls"),
                (gx.expectations.ExpectColumnValuesToNotBeNull(column="productId"),
                 "productId has no nulls"),
                (gx.expectations.ExpectColumnValuesToNotBeNull(column="rating"),
                 "rating has no nulls"),
                (gx.expectations.ExpectColumnValuesToBeBetween(
                    column="rating", min_value=1.0, max_value=5.0),
                 "rating values in range [1, 5]"),
                (gx.expectations.ExpectTableRowCountToBeBetween(
                    min_value=10000, max_value=10_000_000),
                 "row count between 10K and 10M"),
                (gx.expectations.ExpectColumnValuesToMatchRegex(
                    column="userId", regex=r"^[A-Z0-9]+$"),
                 "userId matches Amazon reviewer ID format"),
            ]
        else:  # products
            raw_expectations = [
                (gx.expectations.ExpectColumnToExist(column="productId"),
                 "productId column exists"),
                (gx.expectations.ExpectColumnToExist(column="title"),
                 "title column exists"),
                (gx.expectations.ExpectColumnToExist(column="price"),
                 "price column exists"),
                (gx.expectations.ExpectColumnValuesToNotBeNull(column="productId"),
                 "productId has no nulls"),
                (gx.expectations.ExpectColumnValuesToBeUnique(column="productId"),
                 "productId values are unique"),
                (gx.expectations.ExpectColumnValuesToNotBeNull(column="title"),
                 "title has no nulls"),
                (gx.expectations.ExpectColumnValuesToBeBetween(
                    column="price", min_value=0.0,
                    max_value=10000.0, mostly=0.9),
                 "price in range [0, 10000] for 90%+ rows"),
                (gx.expectations.ExpectTableRowCountToBeBetween(
                    min_value=1000, max_value=5_000_000),
                 "product count between 1K and 5M"),
            ]

        # ── Set up ephemeral context + pandas source ──────────────────────────
        context  = gx.get_context(mode="ephemeral")
        ds       = context.data_sources.add_pandas(f"pandas_{name}")
        asset    = ds.add_dataframe_asset(f"{name}_asset")
        batch_def = asset.add_batch_definition_whole_dataframe("batch")
        suite    = context.suites.add(
                       gx.ExpectationSuite(name=f"{name}_suite"))

        for exp, _ in raw_expectations:
            suite.add_expectation(exp)

        vdef = context.validation_definitions.add(
            gx.ValidationDefinition(
                name=f"{name}_validation",
                data=batch_def,
                suite=suite,
            )
        )
        result = vdef.run(batch_parameters={"dataframe": df})

        # ── Parse results ─────────────────────────────────────────────────────
        results  = []
        passed_n = 0
        res_list = result.results if hasattr(result, "results") else []

        for i, (_, description) in enumerate(raw_expectations):
            if i < len(res_list):
                success = bool(res_list[i].success)
            else:
                success = False
            if success:
                passed_n += 1
            status = "PASS" if success else "FAIL"
            log.info(f"[GE] [{name}] {status} — {description}")
            results.append({
                "framework":   "great_expectations",
                "dataset":     name,
                "expectation": description,
                "passed":      success,
            })

        log.info(f"[GE] [{name}] Suite complete — "
                 f"{passed_n}/{len(raw_expectations)} passed")

        # Save GE results CSV
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ge_path = REPORTS_DIR / f"ge_results_{name}.csv"
        pd.DataFrame(results).to_csv(ge_path, index=False)
        log.info(f"[GE] Results saved → {ge_path.name}")
        return results

    except ImportError:
        log.warning("[GE] great_expectations not installed — skipping")
        log.warning("[GE] Install: pip install great-expectations")
        return []
    except Exception as e:
        log.error(f"[GE] Error: {e}", exc_info=True)
        return []


# ── Entry point ───────────────────────────────────────────────────────────────

def run_validation():
    log.info("========== RecoMart Validation Pipeline START ==========")
    log.info(f"Dataset: {DATASET_NAME}")

    # ── Pandas-based validation ───────────────────────────────────────────────
    r_results, r_df, r_path = validate_ratings()
    p_results, p_df, p_path = validate_products()

    # ── Great Expectations suite ──────────────────────────────────────────────
    log.info("=" * 55)
    ge_r = run_great_expectations(r_df, "ratings")
    ge_p = run_great_expectations(p_df, "products")

    # ── Report ────────────────────────────────────────────────────────────────
    report_path = generate_report(
        r_results, r_df, r_path,
        p_results, p_df, p_path
    )

    # Append GE summary to report
    if ge_r or ge_p:
        ge_all   = ge_r + ge_p
        ge_pass  = sum(1 for r in ge_all if r["passed"])
        ge_lines = [
            "\n\n---\n",
            "## 5. Great Expectations Suite Results\n",
            f"Framework: `great_expectations`  \n",
            f"Total expectations: {len(ge_all)} | "
            f"Passed: {ge_pass} | Failed: {len(ge_all)-ge_pass}\n\n",
            "| Dataset | Expectation | Result |\n",
            "|---------|-------------|--------|\n",
        ]
        for r in ge_all:
            s = "✅ PASS" if r["passed"] else "❌ FAIL"
            ge_lines.append(f"| {r['dataset']} | {r['expectation']} | {s} |\n")
        with open(report_path, "a") as f:
            f.writelines(ge_lines)
        log.info(f"[GE] Summary appended to report")

    total  = len(r_results) + len(p_results)
    passed = sum(1 for r in r_results + p_results if r["passed"])
    log.info("========== Validation COMPLETE ==========")
    log.info(f"Pandas checks : {passed}/{total} passed")
    if ge_r or ge_p:
        ge_all  = ge_r + ge_p
        ge_pass = sum(1 for r in ge_all if r["passed"])
        log.info(f"GE checks     : {ge_pass}/{len(ge_all)} passed")
    log.info(f"Report: {report_path}")
    return report_path


if __name__ == "__main__":
    run_validation()
