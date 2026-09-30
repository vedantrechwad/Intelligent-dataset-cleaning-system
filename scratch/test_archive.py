import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.file_loader import load_dataset
from src.profiling.profiler import profile_dataset
from src.detection.anomaly_detector import detect_all_issues
from src.cleaning.pipeline import execute_cleaning_pipeline
from src.validation.validator import validate_cleaned_dataset

def main():
    from cleaner.profile import profile_table as cleaner_profile_table
    from cleaner.engine import CleaningEngine
    
    archive_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "archive (2)")
    for filename in os.listdir(archive_dir):
        if not filename.endswith(".csv"):
            continue
        print(f"\n--- Testing {filename} with Evidence-Backed Engine ---")
        csv_path = os.path.join(archive_dir, filename)
        orig_df, work_df, meta = load_dataset(csv_path)
        
        raw_issues = detect_all_issues(work_df)
        print(f"Original issues count: {len(raw_issues)}")
        
        cleaner_df = work_df.astype(str).fillna("")
        cleaner_profile = cleaner_profile_table(cleaner_df)
        c_engine = CleaningEngine()
        proposals = c_engine.generate_proposals(cleaner_df, cleaner_profile)
        
        approved_props = [p for p in proposals if p.tier != "FLAG"]
        
        cleaned_df, diff_records = c_engine.apply(cleaner_df, approved_props)
        
        # FIX for type inconsistency explosion
        cleaned_df = cleaned_df.replace("", np.nan)
        for col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="ignore")
            if col in work_df.columns:
                try:
                    cleaned_df[col] = cleaned_df[col].astype(work_df[col].dtype)
                except:
                    pass
        
        af_prof, af_issues, af_score = validate_cleaned_dataset(cleaned_df)
        print(f"After cleaning issues count: {len(af_issues)}")
        
        if len(af_issues) > len(raw_issues):
            print("!!! ISSUE INCREASED !!!")
            raw_types = {}
            for r in raw_issues:
                raw_types[r['issue_type']] = raw_types.get(r['issue_type'], 0) + 1
            af_types = {}
            for a in af_issues:
                af_types[a['issue_type']] = af_types.get(a['issue_type'], 0) + 1
                
            for t, count in af_types.items():
                if count > raw_types.get(t, 0):
                    print(f"Increased issue type: {t}, Raw: {raw_types.get(t, 0)} -> After: {count}")

if __name__ == "__main__":
    main()
