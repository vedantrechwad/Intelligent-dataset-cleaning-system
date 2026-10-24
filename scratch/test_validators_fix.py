import re
import pandas as pd
from cleaner.validators import EMAIL_RE, URL_RE, US_ZIP_RE, CA_POSTAL_RE, UK_POSTAL_RE, BOOLEAN_WORDS, US_STATE_CODES, ISO_COUNTRY_CODES

FLOAT_RE = re.compile(r"^[-+]?\d+\.\d+$")

def is_valid_phone_refined(val: str) -> bool:
    # Continuous floats (e.g. 25.38, 0.006399) are NOT phone numbers
    if FLOAT_RE.match(val):
        return False
    from cleaner.validators import PHONE_RE
    if not PHONE_RE.match(val):
        return False
    digits = sum(1 for c in val if c.isdigit())
    return 7 <= digits <= 15

print("Phone refined tests:")
print("  '25.38':", is_valid_phone_refined("25.38"))  # Should be False
print("  '0.006399':", is_valid_phone_refined("0.006399"))  # Should be False
print("  '+1 555-123-4567':", is_valid_phone_refined("+1 555-123-4567"))  # Should be True
