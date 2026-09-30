import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from cleaner.rules.base import Proposal, CellChange
from cleaner.exporter import generate_cleaning_script

# Test proposal
prop1 = Proposal(
    id="R1_missing_test",
    kind="R1_missing_tokens",
    tier="AUTO",
    columns=["City"],
    description="Convert disguised missing tokens",
    evidence="10 obs",
    changes=[CellChange(row=0, column="City", old_value="N/A", new_value="")]
)
prop2 = Proposal(
    id="R8_duplicates_test",
    kind="R8_exact_duplicates",
    tier="AUTO",
    columns=["__all__"],
    description="Drop duplicate rows",
    evidence="2 rows",
    dropped_rows=[1, 2]
)

script = generate_cleaning_script([prop1, prop2], "test.csv")
print("Generated Script Length:", len(script))
assert "def clean_dataset" in script
assert "drop_duplicates" in script
assert "R1_missing_test" in script

# Test compilation of script
compiled = compile(script, "<string>", "exec")
print("Compilation successful!")
