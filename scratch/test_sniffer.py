import csv
import re
from typing import Tuple, Optional

def sniff_delimiter_and_header(sample: str) -> Tuple[str, Optional[int]]:
    lines = [l.strip() for l in sample.splitlines() if l.strip()][:10]
    if not lines:
        return ",", 0

    counts = {",": 0, ";": 0, "\t": 0, "|": 0}
    for l in lines:
        for delim in counts:
            counts[delim] += l.count(delim)
    
    best_delim = max(counts, key=counts.get)
    sep = best_delim if counts[best_delim] > 0 else ","

    # Header detection
    if len(lines) < 2:
        return sep, 0

    first_tokens = [t.strip().strip("'\"") for t in lines[0].split(sep)]
    second_tokens = [t.strip().strip("'\"") for t in lines[1].split(sep)]

    num_like_first = sum(1 for t in first_tokens if re.match(r"^[-+]?\d+(\.\d+)?$", t))
    num_ratio_first = num_like_first / max(1, len(first_tokens))

    # If first row has >= 20% numbers, it's almost certainly data (not a header)
    if num_ratio_first >= 0.20:
        return sep, None

    return sep, 0

if __name__ == "__main__":
    import os
    files = [
        "benchmarks/hospital/dirty.csv",
        "benchmarks/beers/dirty.csv",
        "new test datasets/bank+marketing/bank.csv",
        "new test datasets/student+performance/student-mat.csv",
        "new test datasets/adult (1)/adult.data",
        "new test datasets/breast+cancer+wisconsin+diagnostic/wdbc.data",
        "new test datasets/gender_submission.csv"
    ]
    for f in files:
        if os.path.exists(f):
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                sample = "".join([fp.readline() for _ in range(10)])
            sep, hdr = sniff_delimiter_and_header(sample)
            print(f"{f}: sep={repr(sep)}, header={hdr}")
