import sys
sys.path.insert(0, ".")
from eval.compare_published import evaluate_dataset

for ds, has_idx in [('rayyan', False), ('movies_1', False)]:
    res = evaluate_dataset(ds, 'held-out', has_index_col=has_idx)
    print(f"=== {ds} ===")
    for r in res:
        m = r['mode']
        v = r['variant']
        p = r['precision']
        rec = r['recall']
        f1 = r['f1']
        fixed = r['fixed']
        e_cnt = r['E_count']
        changed = r['cells_changed']
        damaged = r['damaged']
        print(f"  {m} ({v}): P={p:.4f}, R={rec:.4f}, F1={f1:.4f} | Fixed={fixed}/{e_cnt} Changed={changed} Damaged={damaged}")
