from typing import Dict, Any, List
import pandas as pd

def compute_before_after_comparison(
    before_profile: Dict[str, Any],
    after_profile: Dict[str, Any],
    before_issues: List[Dict[str, Any]],
    after_issues: List[Dict[str, Any]],
    before_score: Dict[str, Any],
    after_score: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Computes rigorous before-and-after comparison metrics.
    Quantifies resolved vs unresolved issues without false claims.
    """
    # Summary of issues by type before
    b_type_counts = {}
    for i in before_issues:
        t = i.get("issue_type", "other")
        b_type_counts[t] = b_type_counts.get(t, 0) + 1

    # Summary of issues by type after
    a_type_counts = {}
    for i in after_issues:
        t = i.get("issue_type", "other")
        a_type_counts[t] = a_type_counts.get(t, 0) + 1

    all_types = sorted(list(set(b_type_counts.keys()).union(set(a_type_counts.keys()))))
    issue_type_comparison = []
    for t in all_types:
        bc = b_type_counts.get(t, 0)
        ac = a_type_counts.get(t, 0)
        resolved = max(0, bc - ac)
        res_rate = round((resolved / bc * 100.0), 1) if bc > 0 else 100.0
        issue_type_comparison.append({
            "issue_type": t,
            "before_count": bc,
            "after_count": ac,
            "resolved_count": resolved,
            "resolution_rate_pct": res_rate
        })

    # Overall metrics comparison
    metrics = {
        "rows": {
            "before": before_profile.get("row_count", 0),
            "after": after_profile.get("row_count", 0),
            "delta": after_profile.get("row_count", 0) - before_profile.get("row_count", 0)
        },
        "missing_cells": {
            "before": before_profile.get("missing_cells", 0),
            "after": after_profile.get("missing_cells", 0),
            "delta": after_profile.get("missing_cells", 0) - before_profile.get("missing_cells", 0)
        },
        "duplicate_rows": {
            "before": before_profile.get("duplicate_rows", 0),
            "after": after_profile.get("duplicate_rows", 0),
            "delta": after_profile.get("duplicate_rows", 0) - before_profile.get("duplicate_rows", 0)
        },
        "total_issues": {
            "before": len(before_issues),
            "after": len(after_issues),
            "delta": len(after_issues) - len(before_issues)
        },
        "quality_score": {
            "before": before_score.get("overall_score", 0.0),
            "after": after_score.get("overall_score", 0.0),
            "delta": round(after_score.get("overall_score", 0.0) - before_score.get("overall_score", 0.0), 2)
        }
    }

    dim_b = before_score.get("dimensions", {})
    dim_a = after_score.get("dimensions", {})
    dimension_deltas = {}
    for d in ["completeness", "consistency", "validity", "uniqueness", "anomaly_quality"]:
        vb = dim_b.get(d, 0.0)
        va = dim_a.get(d, 0.0)
        dimension_deltas[d] = {
            "before": vb,
            "after": va,
            "delta": round(va - vb, 2)
        }

    return {
        "metrics": metrics,
        "dimension_deltas": dimension_deltas,
        "issue_type_comparison": issue_type_comparison,
        "unresolved_issues": after_issues
    }
