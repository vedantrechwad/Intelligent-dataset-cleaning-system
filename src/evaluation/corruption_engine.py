"""
src/evaluation/corruption_engine.py
Upgraded DataCorruptionEngine:
Scientifically corrupts clean tabular datasets with realistic error classes:
- Units, %, currency
- Compound fields
- Dependency violations
- Disguised missing tokens
- Mixed dates
- Near-miss category typos
- Trap columns (short codes, legitimate blanks)
Tracks exact ground-truth coordinates and original values.
"""

import random
import copy
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np


class DataCorruptionEngine:
    """
    Scientifically corrupts a clean tabular dataset with known synthetic anomalies
    to rigorously benchmark detection precision, recall, F1, harm rate, and restoration accuracy.
    Tracks exact ground-truth coordinates and original values.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.ground_truth: List[Dict[str, Any]] = []

    def inject_corruptions(
        self,
        clean_df: pd.DataFrame,
        missing_rate: float = 0.05,
        duplicate_count: int = 3,
        outlier_count: int = 4,
        invalid_numeric_count: int = 3,
        invalid_date_count: int = 3,
        spelling_typo_count: int = 4,
        casing_typo_count: int = 4,
        unit_count: int = 3,
        disguised_missing_count: int = 3
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Injects controlled, known corruptions into a deepcopy of the clean dataframe.
        Preserves backward compatibility with existing tests while adding unit & disguised missing.
        """
        corrupted_df = clean_df.copy(deep=True)
        self.ground_truth = []
        n_rows = len(corrupted_df)
        if n_rows == 0:
            return corrupted_df, []

        numeric_cols = [c for c in corrupted_df.columns if pd.api.types.is_numeric_dtype(corrupted_df[c]) and corrupted_df[c].nunique() > 2]
        date_cols = [c for c in corrupted_df.columns if any(k in c.lower() for k in ["date", "time", "dob", "birth", "join"])]
        str_cols = [c for c in corrupted_df.columns if corrupted_df[c].dtype == object and c not in date_cols and corrupted_df[c].nunique() < min(50, n_rows // 2)]

        # 1. Random Missing Values
        all_cells = [(r, c) for r in range(n_rows) for c in corrupted_df.columns]
        target_missing = int(len(all_cells) * missing_rate)
        sampled_missing = random.sample(all_cells, min(target_missing, len(all_cells)))
        for r, c in sampled_missing:
            orig = corrupted_df.at[r, c]
            corrupted_df.at[r, c] = np.nan
            self.ground_truth.append({
                "row": int(r),
                "column": c,
                "original_value": orig,
                "corrupted_value": np.nan,
                "corruption_type": "missing_value"
            })

        # 2. Exact Duplicates
        if n_rows > 5 and duplicate_count > 0:
            dup_source_indices = random.sample(range(n_rows), min(duplicate_count, n_rows))
            for src_idx in dup_source_indices:
                dup_row = corrupted_df.iloc[[src_idx]].copy()
                new_idx = len(corrupted_df)
                corrupted_df = pd.concat([corrupted_df, dup_row], ignore_index=True)
                self.ground_truth.append({
                    "row": int(new_idx),
                    "column": "__all__",
                    "original_value": f"Duplicate of row {src_idx}",
                    "corrupted_value": "Duplicate row",
                    "corruption_type": "exact_duplicate"
                })

        # Refresh row count after duplication
        n_rows = len(corrupted_df)

        # 3. Numerical Outliers
        if numeric_cols and outlier_count > 0:
            for _ in range(outlier_count):
                col = random.choice(numeric_cols)
                row = random.randint(0, n_rows - 1)
                orig = corrupted_df.at[row, col]
                if pd.notna(orig):
                    corrupted_val = round(float(orig) * random.choice([8.0, 12.0]), 2)
                    corrupted_df.at[row, col] = corrupted_val
                    self.ground_truth.append({
                        "row": int(row),
                        "column": col,
                        "original_value": orig,
                        "corrupted_value": corrupted_val,
                        "corruption_type": "outlier"
                    })

        # 4. Invalid Numerical / Range Violations
        age_cols = [c for c in numeric_cols if "age" in c.lower()]
        target_num_cols = age_cols or numeric_cols
        if target_num_cols and invalid_numeric_count > 0:
            for _ in range(invalid_numeric_count):
                col = random.choice(target_num_cols)
                row = random.randint(0, n_rows - 1)
                orig = corrupted_df.at[row, col]
                if pd.notna(orig):
                    corrupted_val = random.choice([-5, -12, 999])
                    corrupted_df.at[row, col] = corrupted_val
                    self.ground_truth.append({
                        "row": int(row),
                        "column": col,
                        "original_value": orig,
                        "corrupted_value": corrupted_val,
                        "corruption_type": "invalid_range"
                    })

        # 5. Invalid Dates
        if date_cols and invalid_date_count > 0:
            for _ in range(invalid_date_count):
                col = random.choice(date_cols)
                row = random.randint(0, n_rows - 1)
                orig = corrupted_df.at[row, col]
                invalid_date_val = random.choice(["2025-02-30", "22/13/2020", "2024-15-45"])
                corrupted_df.at[row, col] = invalid_date_val
                self.ground_truth.append({
                    "row": int(row),
                    "column": col,
                    "original_value": orig,
                    "corrupted_value": invalid_date_val,
                    "corruption_type": "invalid_date"
                })

        # 6. Categorical Spelling Typos
        if str_cols and spelling_typo_count > 0:
            for _ in range(spelling_typo_count):
                col = random.choice(str_cols)
                row = random.randint(0, n_rows - 1)
                orig = str(corrupted_df.at[row, col])
                if len(orig) >= 4 and pd.notna(orig):
                    pos = random.randint(1, len(orig) - 2)
                    typo = orig[:pos] + orig[pos+1] + orig[pos] + orig[pos+2:]
                    corrupted_df.at[row, col] = typo
                    self.ground_truth.append({
                        "row": int(row),
                        "column": col,
                        "original_value": orig,
                        "corrupted_value": typo,
                        "corruption_type": "spelling_typo"
                    })

        # 7. Casing Inconsistencies
        if str_cols and casing_typo_count > 0:
            for _ in range(casing_typo_count):
                col = random.choice(str_cols)
                row = random.randint(0, n_rows - 1)
                orig = str(corrupted_df.at[row, col])
                if len(orig) >= 2 and pd.notna(orig):
                    cased = orig.lower() if orig.istitle() else orig.upper()
                    corrupted_df.at[row, col] = cased
                    self.ground_truth.append({
                        "row": int(row),
                        "column": col,
                        "original_value": orig,
                        "corrupted_value": cased,
                        "corruption_type": "casing_inconsistency"
                    })

        # 8. Units / Currency Corruption
        if numeric_cols and unit_count > 0:
            for _ in range(unit_count):
                col = random.choice(numeric_cols)
                row = random.randint(0, n_rows - 1)
                orig = corrupted_df.at[row, col]
                if pd.notna(orig):
                    corrupted_df[col] = corrupted_df[col].astype(object)
                    unit_corrupt = f"{orig} {random.choice(['kg', 'USD', 'miles', 'pts'])}"
                    corrupted_df.at[row, col] = unit_corrupt
                    self.ground_truth.append({
                        "row": int(row),
                        "column": col,
                        "original_value": orig,
                        "corrupted_value": unit_corrupt,
                        "corruption_type": "unit_pollution"
                    })

        # 9. Disguised Missing Tokens
        if disguised_missing_count > 0:
            all_str_or_num = numeric_cols + str_cols
            if all_str_or_num:
                for _ in range(disguised_missing_count):
                    col = random.choice(all_str_or_num)
                    row = random.randint(0, n_rows - 1)
                    orig = corrupted_df.at[row, col]
                    if pd.notna(orig):
                        corrupted_df[col] = corrupted_df[col].astype(object)
                        token = random.choice(["N/A", "none", "null", "?", "-"])
                        corrupted_df.at[row, col] = token
                        self.ground_truth.append({
                            "row": int(row),
                            "column": col,
                            "original_value": orig,
                            "corrupted_value": token,
                            "corruption_type": "disguised_missing"
                        })

        return corrupted_df, self.ground_truth

    def inject_realistic_corruptions(
        self,
        clean_df: pd.DataFrame,
        cell_rate: float = 0.10,
        include_traps: bool = True
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Injects realistic error classes into 5-15% of cells:
        - units, %, currency
        - compound fields
        - dependency violations
        - disguised missing tokens
        - mixed dates
        - near-miss categories
        - trap columns (short codes, legitimate blanks)
        """
        corrupted = clean_df.astype(str).copy(deep=True)
        self.ground_truth = []
        n_rows, n_cols = corrupted.shape
        if n_rows == 0 or n_cols == 0:
            return corrupted, []

        total_cells = n_rows * n_cols
        target_corrupted_cells = max(1, int(total_cells * cell_rate))

        # Detect column types
        numeric_cols = []
        date_cols = []
        cat_cols = []
        short_code_cols = []

        for c in corrupted.columns:
            vals = [v.strip() for v in corrupted[c] if v.strip() != ""]
            if not vals:
                continue
            # Check date
            if any(k in c.lower() for k in ["date", "time", "dob"]) or all("-" in v and len(v) == 10 for v in vals[:5]):
                date_cols.append(c)
                continue
            # Check numeric
            is_num = sum(1 for v in vals if v.replace(".", "", 1).replace("-", "", 1).isdigit()) / len(vals) > 0.8
            if is_num:
                numeric_cols.append(c)
                continue
            # Check short codes
            median_len = sorted([len(v) for v in vals])[len(vals) // 2]
            if median_len <= 3:
                short_code_cols.append(c)
            else:
                cat_cols.append(c)

        # Add trap column if requested
        if include_traps and "TRAP_STATE" not in corrupted.columns:
            corrupted["TRAP_STATE"] = [random.choice(["AL", "CA", "NY", "TX", "FL"]) for _ in range(n_rows)]
            short_code_cols.append("TRAP_STATE")
            if "TRAP_LEGIT_BLANK" not in corrupted.columns:
                corrupted["TRAP_LEGIT_BLANK"] = ["" if i % 4 == 0 else f"Val_{i}" for i in range(n_rows)]

        corrupted_count = 0

        # 1. Units, %, Currency
        if numeric_cols and corrupted_count < target_corrupted_cells:
            num_to_corrupt = max(1, target_corrupted_cells // 6)
            for _ in range(num_to_corrupt):
                col = random.choice(numeric_cols)
                row = random.randint(0, n_rows - 1)
                orig = corrupted.at[row, col]
                if orig != "" and not orig.endswith("kg") and not orig.endswith("%"):
                    corrupted_val = random.choice([f"{orig} kg", f"${orig}", f"{orig}%"])
                    corrupted.at[row, col] = corrupted_val
                    self.ground_truth.append({
                        "row": int(row),
                        "column": col,
                        "original_value": orig,
                        "corrupted_value": corrupted_val,
                        "corruption_type": "unit_currency_pollution"
                    })
                    corrupted_count += 1

        # 2. Compound Field Splitting
        if cat_cols and len(cat_cols) >= 2 and corrupted_count < target_corrupted_cells:
            col_a, col_b = random.sample(cat_cols, 2)
            num_to_corrupt = max(1, target_corrupted_cells // 8)
            for _ in range(num_to_corrupt):
                row = random.randint(0, n_rows - 1)
                orig_a = corrupted.at[row, col_a]
                val_b = corrupted.at[row, col_b]
                if orig_a != "" and val_b != "" and val_b not in orig_a:
                    corrupted_val = f"{orig_a} {val_b}"
                    corrupted.at[row, col_a] = corrupted_val
                    self.ground_truth.append({
                        "row": int(row),
                        "column": col_a,
                        "original_value": orig_a,
                        "corrupted_value": corrupted_val,
                        "corruption_type": "compound_field"
                    })
                    corrupted_count += 1

        # 3. Disguised Missing Tokens
        if corrupted_count < target_corrupted_cells:
            num_to_corrupt = max(1, target_corrupted_cells // 6)
            candidate_cols = numeric_cols + cat_cols
            if candidate_cols:
                for _ in range(num_to_corrupt):
                    col = random.choice(candidate_cols)
                    row = random.randint(0, n_rows - 1)
                    orig = corrupted.at[row, col]
                    if orig != "":
                        token = random.choice(["N/A", "null", "none", "?", "missing"])
                        corrupted.at[row, col] = token
                        self.ground_truth.append({
                            "row": int(row),
                            "column": col,
                            "original_value": orig,
                            "corrupted_value": token,
                            "corruption_type": "disguised_missing"
                        })
                        corrupted_count += 1

        # 4. Mixed Dates
        if date_cols and corrupted_count < target_corrupted_cells:
            num_to_corrupt = max(1, target_corrupted_cells // 6)
            for _ in range(num_to_corrupt):
                col = random.choice(date_cols)
                row = random.randint(0, n_rows - 1)
                orig = corrupted.at[row, col]
                if "-" in orig and len(orig) == 10:
                    parts = orig.split("-")
                    if len(parts) == 3 and len(parts[0]) == 4:
                        # Convert YYYY-MM-DD to DD/MM/YYYY
                        corrupted_val = f"{parts[2]}/{parts[1]}/{parts[0]}"
                        corrupted.at[row, col] = corrupted_val
                        self.ground_truth.append({
                            "row": int(row),
                            "column": col,
                            "original_value": orig,
                            "corrupted_value": corrupted_val,
                            "corruption_type": "mixed_date"
                        })
                        corrupted_count += 1

        # 5. Near-Miss Category Typos
        if cat_cols and corrupted_count < target_corrupted_cells:
            num_to_corrupt = max(1, target_corrupted_cells // 6)
            for _ in range(num_to_corrupt):
                col = random.choice(cat_cols)
                row = random.randint(0, n_rows - 1)
                orig = corrupted.at[row, col]
                if len(orig) >= 6:
                    # Typo: swap two adjacent letters
                    idx = random.randint(1, len(orig) - 3)
                    typo = orig[:idx] + orig[idx+1] + orig[idx] + orig[idx+2:]
                    corrupted.at[row, col] = typo
                    self.ground_truth.append({
                        "row": int(row),
                        "column": col,
                        "original_value": orig,
                        "corrupted_value": typo,
                        "corruption_type": "near_miss_category"
                    })
                    corrupted_count += 1

        return corrupted, self.ground_truth
