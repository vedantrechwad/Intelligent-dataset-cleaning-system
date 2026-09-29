"""
Benchmark Harness for Dataset Cleaning System
Measures: Recall, Precision, Harm Rate, Issues/Proposals count, and Changes per rule.

Inputs: dirty D, clean C, output O.
Loaded as dtype=str, keep_default_na=False.
Row alignment by position for rows that survive dedupe (first N rows if duplicates present).
Column alignment by position.
eq(a, b): equal if strings match or both parse as floats and are numerically equal.

E = cells where not eq(D, C)
fixed   = E and eq(O, C)
missed  = E and eq(O, D)
wrong   = E and not eq(O, C) and not eq(O, D)
damaged = (not E) and not eq(O, C)

recall    = fixed / |E|
precision = fixed / (fixed + wrong + damaged)
harm_rate = damaged / (non-error cells)
"""

import os
import sys
import argparse
import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

# Workspace root
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)


def eq(a: Any, b: Any, allow_bool_equiv: bool = False) -> bool:
    """
    Check equality between two cell values.
    Returns True if strings match exactly, or if both parse as floats and are numerically equal.
    If allow_bool_equiv is True, treats Yes/No and True/False as equivalent (used for legacy baseline).
    """
    if a is None:
        a = ""
    if b is None:
        b = ""
    str_a = str(a)
    str_b = str(b)
    if str_a == str_b:
        return True

    if allow_bool_equiv:
        bm = {"true": "1", "yes": "1", "false": "0", "no": "0"}
        al = str_a.strip().lower()
        bl = str_b.strip().lower()
        if al in bm and bl in bm and bm[al] == bm[bl]:
            return True

    try:
        fa = float(str_a)
        fb = float(str_b)
        if not (fa != fa or fb != fb):  # Neither is NaN
            return abs(fa - fb) < 1e-6
    except (ValueError, TypeError, OverflowError):
        pass

    return False


def load_table(path: str) -> pd.DataFrame:
    """Load table strictly as string with keep_default_na=False."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def evaluate_cleaning(
    D: pd.DataFrame,
    C: pd.DataFrame,
    O: pd.DataFrame,
    allow_bool_equiv: bool = False,
    rule_stats: Optional[Dict[str, int]] = None,
    issues_count: int = 0
) -> Dict[str, Any]:
    """
    Evaluate output O against dirty D and clean ground truth C.
    """
    # Number of ground-truth rows
    N = len(C)
    n_cols = min(len(C.columns), len(D.columns), len(O.columns))

    E_count = 0
    non_error_count = 0
    fixed = 0
    missed = 0
    wrong = 0
    damaged = 0

    # Align rows:
    # If O has len >= N, compare the first N rows.
    # If O deduplicated down to N rows, row i of O aligns with row i of C.
    rows_to_eval = min(N, len(O))

    for r in range(rows_to_eval):
        for c in range(n_cols):
            d_val = D.iat[r, c]
            c_val = C.iat[r, c]
            o_val = O.iat[r, c]

            is_error = not eq(d_val, c_val)
            if is_error:
                E_count += 1
                if eq(o_val, c_val, allow_bool_equiv=allow_bool_equiv):
                    fixed += 1
                elif eq(o_val, d_val, allow_bool_equiv=allow_bool_equiv):
                    missed += 1
                else:
                    wrong += 1
            else:
                non_error_count += 1
                if not eq(o_val, c_val, allow_bool_equiv=allow_bool_equiv):
                    damaged += 1

    recall = (fixed / E_count) if E_count > 0 else 0.0
    denom_prec = fixed + wrong + damaged
    precision = (fixed / denom_prec) if denom_prec > 0 else 0.0
    harm_rate = (damaged / non_error_count) if non_error_count > 0 else 0.0

    return {
        "errors_total": E_count,
        "non_errors_total": non_error_count,
        "fixed": fixed,
        "missed": missed,
        "wrong": wrong,
        "damaged": damaged,
        "recall": recall,
        "precision": precision,
        "harm_rate": harm_rate,
        "issues_or_proposals": issues_count,
        "cells_changed_per_rule": rule_stats or {}
    }


def run_legacy_pipeline(
    dirty_path: str,
    mode: str = "AUTO"  # "AUTO" or "AUTO+REVIEW"
) -> Tuple[pd.DataFrame, Dict[str, int], int]:
    """
    Run the existing/legacy pipeline on the dataset.
    """
    from src.ingestion.file_loader import load_dataset
    from src.detection.anomaly_detector import detect_all_issues
    from src.cleaning.pipeline import execute_cleaning_pipeline

    orig_df, work_df, meta = load_dataset(dirty_path)
    raw_issues = detect_all_issues(work_df)

    user_decisions = {}
    if mode == "AUTO+REVIEW":
        # In legacy, HUMAN_REVIEW_REQUIRED issues could be approved if suggested_value exists
        for iss in raw_issues:
            if iss.get("routing_decision") in ["HUMAN_REVIEW_REQUIRED", "SUGGEST_REVIEW"]:
                iid = iss.get("issue_id")
                s_val = iss.get("suggested_value")
                if s_val is not None:
                    user_decisions[iid] = {
                        "status": "accepted",
                        "corrected_value": s_val,
                        "row": iss.get("row"),
                        "column": iss.get("column"),
                        "original_value": iss.get("original_value"),
                        "issue_type": iss.get("issue_type")
                    }

    cleaned_df, audit_trail = execute_cleaning_pipeline(
        work_df, raw_issues, user_corrections=user_decisions if user_decisions else None
    )

    # Format O as string dataframe
    O = cleaned_df.astype(str).fillna("")

    # Collect stats per rule / issue type
    rule_stats: Dict[str, int] = {}
    for entry in audit_trail:
        action_kind = entry.get("issue_type") or entry.get("action", "unknown")
        rule_stats[action_kind] = rule_stats.get(action_kind, 0) + 1

    return O, rule_stats, len(raw_issues)


def run_rule_engine(
    dirty_path: str,
    mode: str = "AUTO"
) -> Tuple[pd.DataFrame, Dict[str, int], int]:
    """
    Run the new rule-engine pipeline (cleaner package).
    """
    try:
        from cleaner.io import load_table as cleaner_load
        from cleaner.profile import profile_table
        from cleaner.engine import CleaningEngine
        import cleaner.rules as rule_pkg
    except ImportError as e:
        raise ImportError(f"cleaner modules not fully implemented yet: {e}")

    df = cleaner_load(dirty_path)
    profile = profile_table(df)
    engine = CleaningEngine()
    proposals = engine.generate_proposals(df, profile)

    if mode == "AUTO":
        approved = [p for p in proposals if p.tier == "AUTO"]
    else:  # AUTO+REVIEW
        approved = [p for p in proposals if p.tier in ["AUTO", "REVIEW"]]

    cleaned_df, diff_records = engine.apply(df, approved)

    rule_stats: Dict[str, int] = {}
    for rec in diff_records:
        r_kind = rec.get("rule_kind", rec.get("rule_id", "unknown"))
        rule_stats[r_kind] = rule_stats.get(r_kind, 0) + 1

    return cleaned_df, rule_stats, len(proposals)


def run_benchmark(
    datasets: Optional[List[str]] = None,
    use_legacy: bool = False,
    modes: Optional[List[str]] = None,
    output_md: Optional[str] = None
) -> str:
    """
    Execute the benchmark suite across specified datasets and modes.
    """
    if datasets is None:
        datasets = ["beers", "hospital", "flights", "rayyan", "telco"]
    if modes is None:
        modes = ["AUTO", "AUTO+REVIEW"]

    results = []

    for name in datasets:
        dirty_path = os.path.join(WORKSPACE_ROOT, "benchmarks", name, "dirty.csv")
        clean_path = os.path.join(WORKSPACE_ROOT, "benchmarks", name, "clean.csv")

        if not os.path.exists(dirty_path) or not os.path.exists(clean_path):
            print(f"Skipping {name}: benchmark files missing.")
            continue

        D = load_table(dirty_path)
        C = load_table(clean_path)

        for mode in modes:
            print(f"\n--- Running benchmark on {name} [{mode}] (legacy={use_legacy}) ---")
            if use_legacy:
                O, rule_stats, issues_or_proposals = run_legacy_pipeline(dirty_path, mode=mode)
            else:
                O, rule_stats, issues_or_proposals = run_rule_engine(dirty_path, mode=mode)

            # Evaluate (for telco on legacy, show both bool_equiv and strict)
            allow_bool = (name == "telco" and use_legacy)
            metrics = evaluate_cleaning(
                D, C, O,
                allow_bool_equiv=allow_bool,
                rule_stats=rule_stats,
                issues_count=issues_or_proposals
            )

            metrics["dataset"] = name
            metrics["mode"] = mode
            metrics["allow_bool"] = allow_bool
            results.append(metrics)

            print(
                f"[{name}] {mode} -> Recall: {metrics['recall']*100:.2f}%, "
                f"Precision: {metrics['precision']*100:.2f}%, Harm: {metrics['harm_rate']*100:.2f}% "
                f"(Fixed: {metrics['fixed']}, Damaged: {metrics['damaged']}, Wrong: {metrics['wrong']}, Issues/Props: {issues_or_proposals})"
            )
            if rule_stats:
                print(f"  Changes by rule/type: {rule_stats}")

            # If legacy and telco, also record strict version
            if name == "telco" and use_legacy and allow_bool:
                strict_metrics = evaluate_cleaning(
                    D, C, O,
                    allow_bool_equiv=False,
                    rule_stats=rule_stats,
                    issues_count=issues_or_proposals
                )
                strict_metrics["dataset"] = "telco (strict)"
                strict_metrics["mode"] = mode
                strict_metrics["allow_bool"] = False
                results.append(strict_metrics)

    # Format Markdown Table
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    engine_name = "Legacy Pipeline (Baseline-v0)" if use_legacy else "New Rule Engine"
    
    md_lines = [
        f"# Benchmark Results ({engine_name})",
        f"**Timestamp**: {datetime.datetime.now().isoformat()}",
        "",
        "| Dataset | Setting | Issues / Proposals | Errors (|E|) | Fixed | Missed | Wrong | Damaged | Recall | Precision | Harm Rate |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for r in results:
        rec_pct = f"{r['recall']*100:.2f}%"
        prec_pct = f"{r['precision']*100:.2f}%"
        harm_pct = f"{r['harm_rate']*100:.2f}%"
        md_lines.append(
            f"| {r['dataset']} | {r['mode']} | {r['issues_or_proposals']} | {r['errors_total']} | "
            f"{r['fixed']} | {r['missed']} | {r['wrong']} | {r['damaged']} | "
            f"{rec_pct} | {prec_pct} | {harm_pct} |"
        )

    md_lines.append("")
    md_lines.append("### Changes Applied Per Rule / Issue Type")
    for r in results:
        if r.get("cells_changed_per_rule"):
            md_lines.append(f"- **{r['dataset']} ({r['mode']})**: `{r['cells_changed_per_rule']}`")

    md_content = "\n".join(md_lines)

    if output_md is None:
        out_dir = os.path.join(WORKSPACE_ROOT, "eval", "results")
        os.makedirs(out_dir, exist_ok=True)
        output_md = os.path.join(out_dir, f"{ts}.md")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[OK] Benchmark results written to {output_md}")
    return md_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset cleaning benchmark harness.")
    parser.add_argument("--legacy", action="store_true", help="Run legacy existing pipeline instead of new rule engine.")
    parser.add_argument("--datasets", type=str, default="beers,hospital,flights,rayyan,telco", help="Comma-separated dataset names.")
    parser.add_argument("--modes", type=str, default="AUTO,AUTO+REVIEW", help="Comma-separated modes: AUTO, AUTO+REVIEW")
    parser.add_argument("--output", type=str, default=None, help="Output markdown path.")

    args = parser.parse_args()
    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    mode_list = [m.strip() for m in args.modes.split(",") if m.strip()]

    run_benchmark(datasets=ds_list, use_legacy=args.legacy, modes=mode_list, output_md=args.output)
