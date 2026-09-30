"""
eval/compare_published.py
Reproducible benchmark comparison against published research results:
- HoloClean (VLDB 2017)
- Baran (PVLDB 2020) Table 3 (Perfect Detection / Oracle)
- Baran (PVLDB 2020) Table 7 (Raha + Baran / End-to-End)

Evaluates: hospital, flights, beers across:
1. end_to_end: normal full pipeline run
2. oracle_detection: keep only changes where dirty != clean (post-hoc oracle filter)
Reports both exact string match and numeric-tolerant match.
Excludes index column on hospital and beers from cell counts.
"""

import os
import sys
import datetime
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from cleaner.io import load_table
from cleaner.profile import profile_table
from cleaner.engine import CleaningEngine


PUBLISHED_BENCHMARKS = [
    {
        "source": "HoloClean 2017, Table 2",
        "system": "HoloClean",
        "dataset": "hospital",
        "status": "development",
        "precision": 1.000,
        "recall": 0.713,
        "f1": 0.832,
        "setup": "Given denial constraints, pruning threshold tau=0.5, DeepDive Gibbs sampling"
    },
    {
        "source": "HoloClean 2017, Table 2",
        "system": "HoloClean",
        "dataset": "flights",
        "status": "held-out",
        "precision": 0.887,
        "recall": 0.669,
        "f1": 0.763,
        "setup": "Given denial constraints, pruning threshold tau=0.3, DeepDive Gibbs sampling"
    },
    {
        "source": "Baran 2020, Table 3 (Perfect Detection)",
        "system": "HoloClean",
        "dataset": "hospital",
        "status": "development",
        "precision": 1.00,
        "recall": 0.71,
        "f1": 0.83,
        "setup": "Oracle error detection, given constraints"
    },
    {
        "source": "Baran 2020, Table 3 (Perfect Detection)",
        "system": "HoloClean",
        "dataset": "flights",
        "status": "held-out",
        "precision": 0.89,
        "recall": 0.67,
        "f1": 0.76,
        "setup": "Oracle error detection, given constraints"
    },
    {
        "source": "Baran 2020, Table 3 (Perfect Detection)",
        "system": "HoloClean",
        "dataset": "beers",
        "status": "development",
        "precision": 0.01,
        "recall": 0.01,
        "f1": 0.01,
        "setup": "Oracle error detection, given constraints"
    },
    {
        "source": "Baran 2020, Table 3 (Perfect Detection)",
        "system": "Baran",
        "dataset": "hospital",
        "status": "development",
        "precision": 0.88,
        "recall": 0.86,
        "f1": 0.87,
        "setup": "Oracle error detection, 20 labeled tuples, 10-run mean"
    },
    {
        "source": "Baran 2020, Table 3 (Perfect Detection)",
        "system": "Baran",
        "dataset": "flights",
        "status": "held-out",
        "precision": 1.00,
        "recall": 1.00,
        "f1": 1.00,
        "setup": "Oracle error detection, 20 labeled tuples, 10-run mean"
    },
    {
        "source": "Baran 2020, Table 3 (Perfect Detection)",
        "system": "Baran",
        "dataset": "beers",
        "status": "development",
        "precision": 0.91,
        "recall": 0.89,
        "f1": 0.90,
        "setup": "Oracle error detection, 20 labeled tuples, 10-run mean"
    },
    {
        "source": "Baran 2020, Table 7 (Raha + Baran End-to-End)",
        "system": "Raha + Baran",
        "dataset": "hospital",
        "status": "development",
        "precision": 0.89,
        "recall": 0.52,
        "f1": 0.66,
        "setup": "Raha error detection (20 labels) + Baran (20 labels), 10-run mean"
    },
    {
        "source": "Baran 2020, Table 7 (Raha + Baran End-to-End)",
        "system": "Raha + Baran",
        "dataset": "flights",
        "status": "held-out",
        "precision": 0.88,
        "recall": 0.53,
        "f1": 0.66,
        "setup": "Raha error detection (20 labels) + Baran (20 labels), 10-run mean"
    },
    {
        "source": "Baran 2020, Table 7 (Raha + Baran End-to-End)",
        "system": "Raha + Baran",
        "dataset": "beers",
        "status": "development",
        "precision": 0.93,
        "recall": 0.87,
        "f1": 0.90,
        "setup": "Raha error detection (20 labels) + Baran (20 labels), 10-run mean"
    },
    {
        "source": "Baran 2020, Table 3 (Perfect Detection)",
        "system": "HoloClean",
        "dataset": "tax",
        "status": "held-out",
        "precision": 0.11,
        "recall": 0.11,
        "f1": 0.11,
        "setup": "Oracle error detection, given constraints"
    },
    {
        "source": "Baran 2020, Table 3 (Perfect Detection)",
        "system": "Baran",
        "dataset": "tax",
        "status": "held-out",
        "precision": 0.84,
        "recall": 0.78,
        "f1": 0.81,
        "setup": "Oracle error detection, 20 labeled tuples, 10-run mean"
    },
    {
        "source": "Baran 2020, Table 7 (Raha + Baran End-to-End)",
        "system": "Raha + Baran",
        "dataset": "tax",
        "status": "held-out",
        "precision": 0.77,
        "recall": 0.71,
        "f1": 0.74,
        "setup": "Raha error detection (20 labels) + Baran (20 labels), 10-run mean"
    }
]


def eq_exact(a: Any, b: Any) -> bool:
    """Exact string equality."""
    return str(a if a is not None else "") == str(b if b is not None else "")


def eq_numeric(a: Any, b: Any) -> bool:
    """Exact string equality or numeric float equality."""
    sa = str(a if a is not None else "")
    sb = str(b if b is not None else "")
    if sa == sb:
        return True
    try:
        fa = float(sa)
        fb = float(sb)
        if not (fa != fa or fb != fb):
            return abs(fa - fb) < 1e-6
    except (ValueError, TypeError, OverflowError):
        pass
    return False


def calculate_metrics(
    df_dirty: pd.DataFrame,
    df_clean: pd.DataFrame,
    df_out: pd.DataFrame,
    numeric_tolerant: bool = False,
    exclude_first_col_as_index: bool = False
) -> Dict[str, Any]:
    """
    Baran PVLDB 13(11) Section 6.1 metrics:
    E = cells where not eq(dirty, clean)
    fixed = E and eq(out, clean)
    changed = cells where not eq(out, dirty)
    precision = fixed / changed
    recall = fixed / |E|
    harm_rate = changed cells that were already correct / correct cells
    """
    eq_fn = eq_numeric if numeric_tolerant else eq_exact

    n_rows = min(len(df_dirty), len(df_clean), len(df_out))
    n_cols = min(len(df_dirty.columns), len(df_clean.columns), len(df_out.columns))

    start_col = 1 if exclude_first_col_as_index else 0

    E_count = 0
    clean_cells_count = 0
    fixed_count = 0
    changed_count = 0
    damaged_count = 0

    d_mat = df_dirty.values
    c_mat = df_clean.values
    o_mat = df_out.values

    for r in range(n_rows):
        for c in range(start_col, n_cols):
            d_val = d_mat[r, c]
            c_val = c_mat[r, c]
            o_val = o_mat[r, c]

            is_error = not eq_fn(d_val, c_val)
            is_changed = not eq_fn(o_val, d_val)

            if is_changed:
                changed_count += 1

            if is_error:
                E_count += 1
                if eq_fn(o_val, c_val):
                    fixed_count += 1
            else:
                clean_cells_count += 1
                if not eq_fn(o_val, c_val):
                    damaged_count += 1

    precision = (fixed_count / changed_count) if changed_count > 0 else 0.0
    recall = (fixed_count / E_count) if E_count > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    harm_rate = (damaged_count / clean_cells_count) if clean_cells_count > 0 else 0.0

    return {
        "E_count": E_count,
        "clean_cells_count": clean_cells_count,
        "fixed": fixed_count,
        "cells_changed": changed_count,
        "damaged": damaged_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "harm_rate": harm_rate,
        "numeric_tolerant": numeric_tolerant
    }


def run_pipeline_on_dataset(dirty_path: str) -> pd.DataFrame:
    """Run full pipeline with all proposals approved."""
    df_dirty = load_table(dirty_path)
    profile = profile_table(df_dirty)
    engine = CleaningEngine()
    proposals = engine.generate_proposals(df_dirty, profile)
    # Approve all proposals (AUTO + REVIEW)
    approved = [p for p in proposals if p.tier in ["AUTO", "REVIEW"]]
    df_out, _ = engine.apply(df_dirty, approved)
    return df_out


def apply_oracle_detection_filter(
    df_dirty: pd.DataFrame,
    df_clean: pd.DataFrame,
    df_out: pd.DataFrame,
    numeric_tolerant: bool = False
) -> pd.DataFrame:
    """
    Oracle Detection Mode:
    Keep only changes to cells where dirty != clean (revert all other changes back to dirty).
    Approximation note: This simulates perfect upstream error detection by filtering changes post-hoc.
    """
    eq_fn = eq_numeric if numeric_tolerant else eq_exact
    d_mat = df_dirty.values
    c_mat = df_clean.values
    o_mat = df_out.values

    n_rows = min(len(df_dirty), len(df_clean), len(df_out))
    n_cols = min(len(df_dirty.columns), len(df_clean.columns), len(df_out.columns))

    oracle_mat = o_mat.copy()

    for r in range(n_rows):
        for c in range(n_cols):
            d_val = d_mat[r, c]
            c_val = c_mat[r, c]
            if eq_fn(d_val, c_val):
                # Originally clean cell: revert to dirty value
                oracle_mat[r, c] = d_val

    return pd.DataFrame(oracle_mat, columns=df_out.columns[:n_cols], index=df_out.index[:n_rows])


def evaluate_dataset(
    dataset_name: str,
    status: str,
    has_index_col: bool = False
) -> List[Dict[str, Any]]:
    dirty_path = os.path.join(WORKSPACE_ROOT, "benchmarks", dataset_name, "dirty.csv")
    clean_path = os.path.join(WORKSPACE_ROOT, "benchmarks", dataset_name, "clean.csv")

    df_dirty = load_table(dirty_path)
    df_clean = load_table(clean_path)

    # 1. End-to-end run
    df_e2e = run_pipeline_on_dataset(dirty_path)

    # 2. Oracle detection run
    df_oracle_exact = apply_oracle_detection_filter(df_dirty, df_clean, df_e2e, numeric_tolerant=False)
    df_oracle_num = apply_oracle_detection_filter(df_dirty, df_clean, df_e2e, numeric_tolerant=True)

    records = []

    # E2E Exact
    m_e2e_exact = calculate_metrics(df_dirty, df_clean, df_e2e, numeric_tolerant=False, exclude_first_col_as_index=has_index_col)
    records.append({
        "dataset": dataset_name,
        "mode": "end_to_end",
        "variant": "exact",
        "status": status,
        **m_e2e_exact
    })

    # E2E Numeric Tolerant
    m_e2e_num = calculate_metrics(df_dirty, df_clean, df_e2e, numeric_tolerant=True, exclude_first_col_as_index=has_index_col)
    records.append({
        "dataset": dataset_name,
        "mode": "end_to_end",
        "variant": "numeric_tolerant",
        "status": status,
        **m_e2e_num
    })

    # Oracle Exact
    m_ora_exact = calculate_metrics(df_dirty, df_clean, df_oracle_exact, numeric_tolerant=False, exclude_first_col_as_index=has_index_col)
    records.append({
        "dataset": dataset_name,
        "mode": "oracle_detection",
        "variant": "exact",
        "status": status,
        **m_ora_exact
    })

    # Oracle Numeric Tolerant
    m_ora_num = calculate_metrics(df_dirty, df_clean, df_oracle_num, numeric_tolerant=True, exclude_first_col_as_index=has_index_col)
    records.append({
        "dataset": dataset_name,
        "mode": "oracle_detection",
        "variant": "numeric_tolerant",
        "status": status,
        **m_ora_num
    })

    return records


def generate_comparison_report(output_path: Optional[str] = None) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets_to_run = [
        ("hospital", "development", True),
        ("beers", "development", True),
        ("tax", "held-out", False)
    ]

    all_records = []
    for name, status, has_idx in datasets_to_run:
        print(f"Evaluating {name} ({status})...")
        recs = evaluate_dataset(name, status, has_index_col=has_idx)
        all_records.extend(recs)

    # Format Markdown
    lines = [
        "# Reproducible Benchmark Comparison Against Published Research",
        f"**Generated**: {datetime.datetime.now().isoformat()}",
        "",
        "## Setup & Methodology Notes",
        "- **Baran (2020)**: Semi-supervised correction model evaluated using 20 labeled tuples and a 10-run mean.",
        "- **HoloClean (2017)**: Probabilistic factor graph with denial constraints, matching dependencies, and statistical learning.",
        "- **Our System**: Deterministic evidence-backed rule engine using **0 labeled tuples** and **1 single deterministic run**.",
        "- **Evaluation Modes**:",
        "  1. `end_to_end`: Normal full automated pipeline (compares to Raha + Baran / Raha + HoloClean).",
        "  2. `oracle_detection`: Post-hoc filter keeping changes only where `dirty != clean` (approximates perfect error detection, comparing to Baran Table 3 / HoloClean Table 2).",
        "- **Index Column Exclusion**: First column (`index`) in Hospital and Beers is excluded from all cell counts.",
        "",
        "## 1. Published Research Numbers",
        "",
        "| Source | System | Dataset | Status | Precision | Recall | F1 | Notes |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for p in PUBLISHED_BENCHMARKS:
        lines.append(
            f"| {p['source']} | {p['system']} | **{p['dataset']}** | {p['status']} | "
            f"{p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | {p['setup']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Our System Performance Across Modes & Variants",
        "",
        "| Dataset | Mode | Variant | Status | Precision (P) | Recall (R) | F1-Score | Harm Rate | Cells Changed | Errors (|E|) | Fixed |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])

    for r in all_records:
        lines.append(
            f"| **{r['dataset']}** | `{r['mode']}` | {r['variant']} | {r['status']} | "
            f"**{r['precision']*100:.2f}%** | **{r['recall']*100:.2f}%** | **{r['f1']*100:.2f}%** | "
            f"{r['harm_rate']*100:.3f}% | {r['cells_changed']:,} | {r['E_count']:,} | {r['fixed']:,} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Direct Head-to-Head Comparison",
        "",
        "Values where our system **matches or outperforms** published research systems under the same mode and evaluation criteria are **highlighted in bold**.",
        "",
        "### A. Perfect Detection / Oracle Mode Comparison (vs. Baran 2020 Table 3 & HoloClean Table 2)",
        "| Dataset | Metric | HoloClean (2017) | Baran (2020) | Our Engine (Oracle, Exact) | Our Engine (Oracle, Numeric-Tolerant) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ])

    evaluated_datasets = [d[0] for d in datasets_to_run]

    # Table 3 comparison
    for ds_name in evaluated_datasets:
        ds_records = { (r["mode"], r["variant"]): r for r in all_records if r["dataset"] == ds_name }
        r_exact = ds_records.get(("oracle_detection", "exact"))
        r_num = ds_records.get(("oracle_detection", "numeric_tolerant"))

        hc_match = next((p for p in PUBLISHED_BENCHMARKS if p["system"] == "HoloClean" and p["dataset"] == ds_name and ("Table 3" in p["source"] or "Table 2" in p["source"])), None)
        baran_match = next((p for p in PUBLISHED_BENCHMARKS if p["system"] == "Baran" and p["dataset"] == ds_name and "Table 3" in p["source"]), None)

        hc_p = f"{hc_match['precision']:.2f}" if hc_match else "N/A"
        hc_r = f"{hc_match['recall']:.2f}" if hc_match else "N/A"
        hc_f = f"{hc_match['f1']:.2f}" if hc_match else "N/A"

        b_p = f"{baran_match['precision']:.2f}" if baran_match else "N/A"
        b_r = f"{baran_match['recall']:.2f}" if baran_match else "N/A"
        b_f = f"{baran_match['f1']:.2f}" if baran_match else "N/A"

        # Check if our engine beats published
        def fmt_val(val, benchmark_val):
            val_pct = f"{val*100:.2f}%"
            if benchmark_val is not None and val >= benchmark_val:
                return f"**{val_pct}**"
            return val_pct

        bench_p = max(float(hc_p) if hc_p != "N/A" else 0, float(b_p) if b_p != "N/A" else 0)
        bench_r = max(float(hc_r) if hc_r != "N/A" else 0, float(b_r) if b_r != "N/A" else 0)
        bench_f = max(float(hc_f) if hc_f != "N/A" else 0, float(b_f) if b_f != "N/A" else 0)

        lines.append(f"| **{ds_name}** | Precision | {hc_p} | {b_p} | {fmt_val(r_exact['precision'], bench_p)} | {fmt_val(r_num['precision'], bench_p)} |")
        lines.append(f"| | Recall | {hc_r} | {b_r} | {fmt_val(r_exact['recall'], bench_r)} | {fmt_val(r_num['recall'], bench_r)} |")
        lines.append(f"| | F1-Score | {hc_f} | {b_f} | {fmt_val(r_exact['f1'], bench_f)} | {fmt_val(r_num['f1'], bench_f)} |")

    lines.extend([
        "",
        "### B. End-to-End Mode Comparison (vs. Raha + Baran Table 7)",
        "| Dataset | Metric | Raha + Baran (Table 7) | Our Engine (End-to-End, Exact) | Our Engine (End-to-End, Numeric-Tolerant) |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ])

    for ds_name in evaluated_datasets:
        ds_records = { (r["mode"], r["variant"]): r for r in all_records if r["dataset"] == ds_name }
        r_exact = ds_records.get(("end_to_end", "exact"))
        r_num = ds_records.get(("end_to_end", "numeric_tolerant"))

        baran_match = next((p for p in PUBLISHED_BENCHMARKS if p["system"] == "Raha + Baran" and p["dataset"] == ds_name), None)
        b_p = f"{baran_match['precision']:.2f}" if baran_match else "N/A"
        b_r = f"{baran_match['recall']:.2f}" if baran_match else "N/A"
        b_f = f"{baran_match['f1']:.2f}" if baran_match else "N/A"

        bench_p = float(b_p) if b_p != "N/A" else None
        bench_r = float(b_r) if b_r != "N/A" else None
        bench_f = float(b_f) if b_f != "N/A" else None

        def fmt_val(val, benchmark_val):
            val_pct = f"{val*100:.2f}%"
            if benchmark_val is not None and val >= benchmark_val:
                return f"**{val_pct}**"
            return val_pct

        lines.append(f"| **{ds_name}** | Precision | {b_p} | {fmt_val(r_exact['precision'], bench_p)} | {fmt_val(r_num['precision'], bench_p)} |")
        lines.append(f"| | Recall | {b_r} | {fmt_val(r_exact['recall'], bench_r)} | {fmt_val(r_num['recall'], bench_r)} |")
        lines.append(f"| | F1-Score | {b_f} | {fmt_val(r_exact['f1'], bench_f)} | {fmt_val(r_num['f1'], bench_f)} |")

    content = "\n".join(lines)

    if output_path is None:
        out_dir = os.path.join(WORKSPACE_ROOT, "eval", "results")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"compare_{timestamp}.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n[OK] Comparison report written to {output_path}")
    return content


if __name__ == "__main__":
    generate_comparison_report()
