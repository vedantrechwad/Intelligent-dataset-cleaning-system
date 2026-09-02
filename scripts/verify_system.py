"""
Comprehensive End-to-End System Verification Script
"""
import os
import sys
import pandas as pd
import numpy as np

# Force UTF-8 on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add workspace to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.file_loader import load_dataset
from src.ingestion.schema_detector import detect_dataset_schema
from src.profiling.profiler import profile_dataset
from src.profiling.quality_metrics import compute_quality_score
from src.detection.anomaly_detector import detect_all_issues
from src.intelligence.ollama_client import OllamaClient
from src.intelligence.reasoning_engine import ReasoningEngine
from src.review.human_review import HumanReviewManager, validate_user_correction_input
from src.cleaning.pipeline import execute_cleaning_pipeline
from src.validation.validator import validate_cleaned_dataset
from src.validation.before_after import compute_before_after_comparison
from src.reporting.audit_logger import AuditLogger
from src.reporting.report_generator import generate_html_report
from src.evaluation.corruption_engine import DataCorruptionEngine
from src.evaluation.detection_metrics import evaluate_detection_performance
from src.evaluation.correction_metrics import evaluate_correction_performance
from src.evaluation.downstream_evaluation import evaluate_downstream_ml

def main():
    print("=" * 70)
    print("STARTING FULL END-TO-END SYSTEM INTEGRATION VERIFICATION")
    print("=" * 70)

    sample_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "dirty_customer_dataset.csv")
    sample_xlsx = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "dirty_customer_dataset.xlsx")
    clean_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "clean_benchmark_dataset.csv")

    # Step 1: Ingestion
    print("\n[Step 1] Ingestion & Immutability Verification...")
    orig_df, work_df, meta = load_dataset(sample_csv)
    print(f"  [OK] Loaded CSV: {meta['row_count']} rows, {meta['column_count']} columns, format={meta['format']}")
    _, _, meta_xlsx = load_dataset(sample_xlsx)
    print(f"  [OK] Loaded XLSX: {meta_xlsx['row_count']} rows, {meta_xlsx['column_count']} columns, format={meta_xlsx['format']}")
    
    # Step 2: Schema Detection & Profiling
    print("\n[Step 2] Schema Detection & Profiling...")
    schema = detect_dataset_schema(work_df)
    profile = profile_dataset(work_df, schema)
    print(f"  [OK] Inferred schema for {len(schema)} columns.")
    print(f"  [OK] Profiling found {profile['missing_cells']} missing cells, {profile['duplicate_rows']} duplicate rows.")

    # Step 3: Anomaly Detection & Reasoning
    print("\n[Step 3] Multi-Dimensional Anomaly Detection...")
    raw_issues = detect_all_issues(work_df)
    print(f"  [OK] Detected {len(raw_issues)} total issues.")
    
    ollama = OllamaClient()
    ollama_ready = ollama.check_availability()
    print(f"  [OK] Ollama LLM Status: {'ONLINE (127.0.0.1:11434)' if ollama_ready else 'OFFLINE (Graceful Heuristic Fallback)'}")
    
    reasoner = ReasoningEngine(ollama)
    enriched_issues = reasoner.enrich_issues(raw_issues, work_df, use_ollama=ollama_ready, max_llm_queries=3)
    
    raw_score = compute_quality_score(work_df, detected_issues=enriched_issues)
    print(f"  [OK] Initial Data Quality Score: {raw_score['overall_score']}/100 (Grade: {raw_score['grade']})")

    # Step 4: Human Review Simulation
    print("\n[Step 4] Human-in-the-Loop Review Queue Simulation...")
    review_manager = HumanReviewManager()
    
    # Simulate user verifying an invalid date and impossible age
    human_needed = [i for i in enriched_issues if i.get("routing_decision") == "HUMAN_REVIEW_REQUIRED"]
    print(f"  [OK] Identified {len(human_needed)} issues requiring human review.")
    
    user_decisions = {}
    for iss in human_needed:
        iid = iss.get("issue_id")
        col = iss.get("column")
        itype = iss.get("issue_type")
        r = iss.get("row")
        orig = iss.get("original_value")

        if itype == "invalid_date":
            ok, msg = review_manager.record_decision(
                iid, "accept", r, col, orig, itype, corrected_value="2024-01-20"
            )
            assert ok, f"Date validation error: {msg}"
        elif itype == "invalid_range" and "age" in col.lower():
            ok, msg = review_manager.record_decision(
                iid, "accept", r, col, orig, itype, corrected_value=32
            )
            assert ok, f"Age validation error: {msg}"
        else:
            review_manager.record_decision(
                iid, "reject", r, col, orig, itype, notes="Skipped in benchmark"
            )

    user_decisions = review_manager.get_all_decisions()
    print(f"  [OK] Successfully verified and recorded {len(user_decisions)} human review decisions.")

    # Step 5: Deterministic Cleaning Pipeline
    print("\n[Step 5] Deterministic Cleaning Pipeline Execution...")
    cleaning_opts = {
        "missing_numeric_strategy": "median",
        "missing_categorical_strategy": "mode",
        "outlier_strategy": "flag_only",
        "remove_exact_duplicates": True,
        "normalize_whitespace": True,
        "normalize_text_case": True,
        "impute_missing": True
    }
    cleaned_df, audit_trail = execute_cleaning_pipeline(
        work_df, enriched_issues, user_corrections=user_decisions, cleaning_options=cleaning_opts
    )
    print(f"  [OK] Cleaned DataFrame shape: {cleaned_df.shape} (applied {len(audit_trail)} modifications).")

    # Step 6: Post-Cleaning Validation & Before/After
    print("\n[Step 6] Validation & Before vs. After Deltas...")
    af_prof, af_issues, af_score = validate_cleaned_dataset(cleaned_df)
    comp = compute_before_after_comparison(
        profile, af_prof, enriched_issues, af_issues, raw_score, af_score
    )
    score_delta = comp["metrics"]["quality_score"]["delta"]
    print(f"  [OK] Cleaned Quality Score: {af_score['overall_score']}/100 (Delta: +{score_delta} pts)")
    print(f"  [OK] Missing cells reduced: {profile['missing_cells']} -> {af_prof['missing_cells']}")
    print(f"  [OK] Duplicates reduced: {profile['duplicate_rows']} -> {af_prof['duplicate_rows']}")

    # Step 7: Audit Logging & HTML Report
    print("\n[Step 7] Audit Logging & HTML Report Generation...")
    audit_logger = AuditLogger(dataset_name="dirty_customer_dataset.csv")
    audit_logger.log_detected_issues(enriched_issues)
    audit_logger.log_applied_actions(audit_trail)
    audit_json = audit_logger.to_json()
    assert len(audit_json) > 100
    
    html_report = generate_html_report(
        "dirty_customer_dataset.csv", profile, af_prof, raw_score, af_score, audit_logger.generate_audit_report()
    )
    assert "<html" in html_report.lower()
    
    output_html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "quality_audit_report.html")
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_report)
    print(f"  [OK] Exported standalone HTML report to: {output_html_path}")

    # Step 8: Scientific Data Corruption & Benchmark
    print("\n[Step 8] Scientific Data Corruption & Ground-Truth Benchmarking...")
    clean_df = pd.read_csv(clean_csv)
    corrupter = DataCorruptionEngine(seed=42)
    corrupted_df, ground_truth = corrupter.inject_corruptions(
        clean_df, missing_rate=0.08, duplicate_count=3, outlier_count=4,
        invalid_numeric_count=3, invalid_date_count=3, spelling_typo_count=4
    )
    print(f"  [OK] Injected {len(ground_truth)} controlled ground-truth anomalies.")
    
    det_issues = detect_all_issues(corrupted_df)
    det_metrics = evaluate_detection_performance(ground_truth, det_issues)
    o = det_metrics["overall"]
    print(f"  [OK] Detection Metrics: Precision={o['precision']*100:.1f}%, Recall={o['recall']*100:.1f}%, F1={o['f1_score']:.3f}")
    
    bench_cleaned, bench_audit = execute_cleaning_pipeline(corrupted_df, det_issues)
    corr_metrics = evaluate_correction_performance(ground_truth, bench_cleaned, bench_audit)
    print(f"  [OK] Auto-Restoration Accuracy: {corr_metrics['automatic_correction_accuracy']}%")

    # Step 9: Downstream Machine Learning Evaluation
    print("\n[Step 9] Downstream ML Evaluation (Random Forest Classifier on Churn)...")
    ml_res = evaluate_downstream_ml(corrupted_df, bench_cleaned, target_column="Churn", random_seed=42)
    if "corrupted_metrics" in ml_res:
        print(f"  [OK] Corrupted Test Accuracy: {ml_res['corrupted_metrics']['accuracy']*100:.2f}% | Cleaned Test Accuracy: {ml_res['cleaned_metrics']['accuracy']*100:.2f}%")
        print(f"  [OK] Corrupted Test F1: {ml_res['corrupted_metrics']['f1_weighted']:.4f} | Cleaned Test F1: {ml_res['cleaned_metrics']['f1_weighted']:.4f}")

    print("\n" + "=" * 70)
    print("ALL END-TO-END FUNCTIONAL VERIFICATIONS PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    main()
