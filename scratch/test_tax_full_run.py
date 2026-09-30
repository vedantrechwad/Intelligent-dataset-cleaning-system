import sys, os, time
sys.path.insert(0, ".")
import pandas as pd
from eval.compare_published import evaluate_dataset

t0 = time.time()
print("Starting full tax evaluate_dataset...")
records = evaluate_dataset("tax", status="held-out", has_index_col=False)
print(f"Finished in {time.time()-t0:.2f}s!")
for r in records:
    print(f"{r['mode']} ({r['variant']}): P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1']:.4f}, Fixed={r['fixed']}/{r['E_count']}, Changed={r['cells_changed']}, Damaged={r['damaged']}")
