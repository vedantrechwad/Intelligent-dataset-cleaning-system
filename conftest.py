import sys
import os

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)
