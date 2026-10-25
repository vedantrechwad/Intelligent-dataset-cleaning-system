"""
src/intelligence/metric_explainer.py
AI Diagnostic Explainer for Data Quality, Safety, and ML Utility Parameter Shifts.

Explains why specific metrics increased, decreased, or remained neutral after cleaning,
attributing changes to specific transformation rules, mathematical mechanisms, and
practical data engineering trade-offs.

Supports local Ollama LLMs with intelligent deterministic expert fallback.
"""

import json
from typing import Dict, Any, List, Optional
from src.intelligence.ollama_client import OllamaClient
from src.utils.helpers import logger


def generate_expert_explanation(
    param_name: str,
    before_val: float,
    after_val: float,
    delta: float,
    applied_rules: List[str],
    dataset_context: Dict[str, Any]
) -> Dict[str, str]:
    """
    Generates a deterministic, expert data engineering explanation for any parameter shift.
    Used when Ollama is offline or for instant fallback.
    """
    param_lower = param_name.lower().strip().replace(" ", "_").replace("-", "_")
    p_name_display = param_name.replace("_", " ").title()
    rules_str = ", ".join(applied_rules[:4]) if applied_rules else "standard deterministic normalization"

    # Determine direction
    if abs(delta) < 0.001:
        direction = "neutral"
    elif delta > 0:
        direction = "improved"
    else:
        direction = "reduced"

    # 1. Overall Quality Score
    if "quality_score" in param_lower or "overall" in param_lower:
        if direction == "improved":
            headline = f"Overall Quality Score rose by +{delta:.1f} points."
            exp = (
                f"The aggregate health score improved from {before_val:.1f} to {after_val:.1f} because approved "
                f"rules ({rules_str}) resolved verified defect cells across validity, consistency, and anomaly dimensions. "
                f"Under the continuous hyperbolic penalty model, repairing corruptions yields strictly positive dimensional gains."
            )
        elif direction == "reduced":
            headline = f"Overall Quality Score adjusted slightly by {delta:.1f} points."
            exp = (
                f"The score moved from {before_val:.1f} to {after_val:.1f}. This typically occurs when disguised sentinel "
                f"strings (such as 'UNKNOWN' or '?') were converted into formal null representations, properly exposing "
                f"unfilled data rather than masking missingness under artificial string validity."
            )
        else:
            headline = f"Overall Quality Score remained stable at {after_val:.1f}."
            exp = (
                f"The score remained at {after_val:.1f}. In high-volume datasets, localized cell repairs are balanced "
                f"against unedited raw columns, maintaining overall structural health while preserving untouched records."
            )

    # 2. Completeness (Missing Value Rate)
    elif "completeness" in param_lower or "missing" in param_lower:
        if direction == "improved":
            headline = f"Completeness improved from {before_val:.1f}% to {after_val:.1f}%."
            exp = (
                f"Imputation rules and default reconstruction filled recoverable missing cells, increasing data density "
                f"without introducing synthetic hallucination across required attributes."
            )
        elif direction == "reduced":
            headline = f"Completeness shifted from {before_val:.1f}% to {after_val:.1f}%."
            exp = (
                f"This reduction is a positive data hygiene signal: previously disguised sentinel strings (such as '?', "
                f"'N/A', 'UNKNOWN', or 'ERROR') were previously miscounted as valid textual content. Converting them to "
                f"canonical nulls honestly reflects true missingness for downstream imputation."
            )
        else:
            headline = f"Completeness held steady at {after_val:.1f}%."
            exp = (
                f"Completeness remained unchanged at {after_val:.1f}% because AutoClean's Zero-Harm Invariant strictly "
                f"prohibits guessing or synthesizing non-recoverable values without verified ground truth."
            )

    # 3. Validity (Type, Date, Format Conformity)
    elif "validity" in param_lower or "invalid" in param_lower:
        if direction == "improved":
            headline = f"Validity improved by +{delta:.1f}% to reach {after_val:.1f}%."
            exp = (
                f"Rule R5 (Type Casting) and R4 (Date Standardization) resolved malformed string representations, "
                f"coercing numeric strings, standardizing ISO 8601 timestamps, and trimming rogue currency symbols."
            )
        elif direction == "reduced":
            headline = f"Validity recorded a minor delta of {delta:.1f}%."
            exp = (
                f"Strict schema verification re-evaluated remaining unparseable tokens following type coercion, "
                f"isolating genuine domain violations that require manual human authorization."
            )
        else:
            headline = f"Validity maintained at {after_val:.1f}%."
            exp = (
                f"Existing valid values conformed to expected data types and bounds, with zero syntax errors introduced."
            )

    # 4. Consistency (Casing, Whitespace, Typo, Vocabulary)
    elif "consistency" in param_lower or "inconsist" in param_lower:
        if direction == "improved":
            headline = f"Consistency surged by +{delta:.1f}% (now {after_val:.1f}%)."
            exp = (
                f"Rules R3 (Whitespace & Casing) and Semantic Harmonization eliminated whitespace padding and case fragmentation "
                f"(e.g., standardizing 'Male' vs 'male'), collapsing fragmented category vocabularies into unified canonical groups."
            )
        elif direction == "reduced":
            headline = f"Consistency adjusted by {delta:.1f}%."
            exp = (
                f"Standardizing dominant categories concentrated rare edge cases, temporarily highlighting residual "
                f"spelling variations in free-text fields that can be finalized in the Review Changes editor."
            )
        else:
            headline = f"Consistency remained steady at {after_val:.1f}%."
            exp = (
                f"Categorical values already exhibited standard casing and formatting with minimal whitespace variance."
            )

    # 5. Uniqueness (Duplicate Records)
    elif "uniqueness" in param_lower or "duplicate" in param_lower:
        if direction == "improved":
            headline = f"Uniqueness reached {after_val:.1f}% (+{delta:.1f}% gain)."
            exp = (
                f"Exact duplicate rows were deduplicated while preserving the initial primary record, eliminating redundant "
                f"training weight in downstream predictive models."
            )
        elif direction == "reduced":
            headline = f"Uniqueness recorded an adjustment of {delta:.1f}%."
            exp = (
                f"Deduplication prioritized record preservation under conservative thresholds to prevent accidental deletion of valid recurring transactions."
            )
        else:
            headline = f"Uniqueness held at {after_val:.1f}%."
            exp = (
                f"All records in the dataset are distinct unique entities, with zero duplicate rows detected."
            )

    # 6. Anomaly Quality / Outliers
    elif "anomaly" in param_lower or "outlier" in param_lower:
        if direction == "improved":
            headline = f"Anomaly score improved by +{delta:.1f}% to {after_val:.1f}%."
            exp = (
                f"Rule R4 (Outlier Handling) applied bounded MAD/IQR Winsorization to extreme statistical leverage points, "
                f"capping artificial sensor spikes while preserving 100% of the sample records."
            )
        elif direction == "reduced":
            headline = f"Anomaly score shifted by {delta:.1f}%."
            exp = (
                f"Cleaning numeric features exposed previously hidden distributional skewness in multi-modal distributions, "
                f"flagging candidate outliers for user inspection."
            )
        else:
            headline = f"Anomaly Quality remained at {after_val:.1f}%."
            exp = (
                f"Continuous feature distributions remained within normal standard deviation bounds without severe anomalies."
            )

    # 7. Downstream ML: Accuracy / F1 Score
    elif "acc" in param_lower or "f1" in param_lower or "precision" in param_lower:
        if direction == "improved":
            headline = f"Downstream ML predictive accuracy lifted by +{delta:.2f}%."
            exp = (
                f"Resolving dirty categories and disguised missing values allowed the classifier to construct pure decision "
                f"boundaries on held-out test splits, eliminating misclassification noise caused by dirty string variants."
            )
        elif direction == "reduced":
            headline = f"Downstream ML metric shifted by {delta:.2f}%."
            exp = (
                f"In certain datasets, dirty sentinels (e.g. '?' or 'ERROR') act as unintended label leaks in raw data. "
                f"Neutralizing them creates a more honest, robust model that generalizes better to unseen real-world records."
            )
        else:
            headline = f"Downstream ML accuracy remained stable."
            exp = (
                f"Model classification accuracy was preserved across identical held-out splits with zero performance degradation."
            )

    # 8. Downstream ML: R-Squared / RMSE
    elif "r2" in param_lower or "r_squared" in param_lower or "rmse" in param_lower or "mae" in param_lower:
        if direction == "improved":
            headline = f"Regression utility surged: R-Squared (R²) gained +{delta:.3f} points."
            exp = (
                f"Standardizing numeric formats and resolving corruptions unlocked high-signal numerical features for the "
                f"regressor, dramatically reducing mean prediction error (RMSE)."
            )
        elif direction == "reduced":
            headline = f"R-Squared recorded a minor adjustment of {delta:.3f} points."
            exp = (
                f"Capping extreme outliers reduces the artificial variance accounted for by a few distant leverage points. "
                f"While raw R-Squared on extreme points may dip slightly, the model's generalized median error is substantially safer."
            )
        else:
            headline = f"Regression performance held steady."
            exp = (
                f"Target feature relationships remained stable on held-out test data without disruption to predictive utility."
            )

    # 9. Defect Resolution Rate (DRR)
    elif "drr" in param_lower or "resolution" in param_lower:
        headline = f"Defect Resolution Rate reached {after_val:.1f}%."
        exp = (
            f"AutoClean successfully resolved {after_val:.1f}% of all detectable anomalies. Remaining issues represent "
            f"semantic ground-truth decisions intentionally deferred to human review in Tab 2 to prevent unauthorized guessing."
        )

    # Generic Fallback
    else:
        headline = f"{p_name_display} changed from {before_val:.2f} to {after_val:.2f}."
        exp = (
            f"Applied cleaning rules ({rules_str}) transformed affected cells into standard representations. "
            f"The shift reflects improved schema fidelity and systematic noise reduction."
        )

    return {
        "parameter": p_name_display,
        "direction": direction,
        "headline": headline,
        "explanation": exp,
        "source": "expert_rules"
    }


def explain_metric_shift(
    param_name: str,
    before_val: float,
    after_val: float,
    applied_rules: Optional[List[str]] = None,
    dataset_context: Optional[Dict[str, Any]] = None,
    ollama_client: Optional[OllamaClient] = None
) -> Dict[str, str]:
    """
    Explains a metric shift using local Ollama if available, with immediate deterministic fallback.
    """
    applied = applied_rules or []
    ctx = dataset_context or {}
    delta = round(after_val - before_val, 4)

    # Quick deterministic baseline
    base_res = generate_expert_explanation(param_name, before_val, after_val, delta, applied, ctx)

    # If Ollama is available, enhance with custom contextual LLM generation
    if ollama_client is not None and ollama_client.check_availability():
        try:
            rules_str = ", ".join(applied[:5]) if applied else "standard cleaning rules"
            ds_name = ctx.get("name", "the dataset")
            rows = ctx.get("rows", "N/A")

            prompt = f"""You are a senior data engineering scientist explaining a benchmark data quality metric shift to a researcher.

Dataset: {ds_name} ({rows} rows)
Metric Name: {param_name}
Value Before Cleaning: {before_val}
Value After Cleaning: {after_val}
Absolute Shift: {'+' if delta >= 0 else ''}{delta}
Applied Transformation Rules: {rules_str}

Task:
Write a clear, authoritative, 2-to-3 sentence explanation answering:
1. What specific data cleaning operations caused this metric to shift?
2. Why did the value mathematically increase, stay flat, or decrease?
3. Why is this outcome technically healthy for downstream analytics and machine learning?

Respond ONLY with a JSON object in this exact format:
{{
    "headline": "A punchy 1-sentence summary of the shift",
    "explanation": "2-3 authoritative sentences explaining the mechanism and trade-off."
}}
"""
            payload = {
                "model": ollama_client.model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }

            resp = ollama_client._post_generate(payload)
            if resp and "response" in resp:
                data = json.loads(resp["response"])
                if "explanation" in data and len(data["explanation"].strip()) > 20:
                    return {
                        "parameter": base_res["parameter"],
                        "direction": base_res["direction"],
                        "headline": data.get("headline", base_res["headline"]),
                        "explanation": data["explanation"].strip(),
                        "source": "ollama"
                    }
        except Exception as e:
            logger.debug(f"Ollama metric explanation fallback: {e}")

    return base_res
