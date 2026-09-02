import random
import copy
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np

class DataCorruptionEngine:
    """
    Scientifically corrupts a clean tabular dataset with known synthetic anomalies
    to rigorously benchmark detection precision, recall, F1, and restoration accuracy.
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
        casing_typo_count: int = 4
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Injects controlled, known corruptions into a deepcopy of the clean dataframe.
        """
        corrupted_df = clean_df.copy(deep=True)
        self.ground_truth = []
        n_rows = len(corrupted_df)
        if n_rows == 0:
            return corrupted_df, []

        numeric_cols = [c for c in corrupted_df.columns if pd.api.types.is_numeric_dtype(corrupted_df[c])]
        str_cols = [c for c in corrupted_df.columns if corrupted_df[c].dtype == object]
        date_cols = [c for c in corrupted_df.columns if any(k in c.lower() for k in ["date", "time", "dob", "birth", "join"])]

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
                    # Multiplied by 10 or 15 to ensure statistical extremity
                    corrupted_val = round(float(orig) * random.choice([8.0, 12.0]), 2)
                    corrupted_df.at[row, col] = corrupted_val
                    self.ground_truth.append({
                        "row": int(row),
                        "column": col,
                        "original_value": orig,
                        "corrupted_value": corrupted_val,
                        "corruption_type": "outlier"
                    })

        # 4. Invalid Numerical / Range Violations (e.g. negative age)
        age_cols = [c for c in numeric_cols if "age" in c.lower()]
        target_num_cols = age_cols or numeric_cols
        if target_num_cols and invalid_numeric_count > 0:
            for _ in range(invalid_numeric_count):
                col = random.choice(target_num_cols)
                row = random.randint(0, n_rows - 1)
                orig = corrupted_df.at[row, col]
                if pd.notna(orig):
                    # Negative or impossible value
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
                    # Swap two adjacent letters to create realistic typo
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

        return corrupted_df, self.ground_truth
