from typing import List, Dict, Any

def evaluate_detection_performance(
    ground_truth: List[Dict[str, Any]],
    detected_issues: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes scientific detection evaluation metrics:
    True Positives (TP), False Positives (FP), False Negatives (FN),
    Precision, Recall, and F1 Score (overall and per issue type).
    """
    # Key mapping: (row, column) -> ground_truth_record
    gt_map = {}
    gt_by_type = {}
    for gt in ground_truth:
        r = gt.get("row")
        c = gt.get("column")
        ctype = gt.get("corruption_type")
        gt_map[(r, c)] = gt
        gt_by_type.setdefault(ctype, []).append((r, c))

    # Detected map: (row, column) -> detected_issue
    detected_map = {}
    detected_by_type = {}
    for issue in detected_issues:
        r = issue.get("row")
        c = issue.get("column")
        itype = issue.get("issue_type")
        detected_map[(r, c)] = issue
        detected_by_type.setdefault(itype, []).append((r, c))

    all_corruption_types = sorted(list(gt_by_type.keys()))
    type_metrics = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for ctype in all_corruption_types:
        gt_cells = set(gt_by_type[ctype])
        # Find detected issues matching this cell
        tp = 0
        fn = 0
        
        for cell in gt_cells:
            if cell in detected_map:
                tp += 1
            else:
                fn += 1

        # False positives for this type
        det_cells = set(detected_by_type.get(ctype, []))
        fp = len(det_cells - gt_cells)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        type_metrics[ctype] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1_score": round(f1, 3),
            "ground_truth_count": len(gt_cells)
        }

        total_tp += tp
        total_fn += fn

    # Overall false positives
    detected_cells = set(detected_map.keys())
    all_gt_cells = set(gt_map.keys())
    total_fp = len(detected_cells - all_gt_cells)

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0

    return {
        "overall": {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": round(overall_prec, 3),
            "recall": round(overall_rec, 3),
            "f1_score": round(overall_f1, 3),
            "total_injected_errors": len(ground_truth),
            "total_detected_issues": len(detected_issues)
        },
        "by_issue_type": type_metrics
    }
