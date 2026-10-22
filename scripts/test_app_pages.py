"""
scripts/test_app_pages.py
--------------------------
Tests every page of the Streamlit dashboard via AppTest.
Each page gets a fresh run to avoid widget state conflicts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAGES = ["Overview", "Video Analysis", "Live Monitor",
         "Event Log", "Model Performance", "About"]

print("=" * 60)
print("Testing all dashboard pages via AppTest")
print("=" * 60)

all_ok = True
for page in PAGES:
    try:
        # Use absolute path — Streamlit 1.65 resolves relative to calling file
        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)
        at.run()
        at.radio[0].set_value(page)
        at.run()
        errs = [e for e in at.exception]
        if errs:
            print(f"  FAIL [{page:20s}]: {errs[0]}")
            all_ok = False
        else:
            n_widgets = len(at.button) + len(at.metric) + len(at.markdown) + len(at.info)
            print(f"  PASS [{page:20s}]  (widgets: {n_widgets})")
    except Exception as e:
        print(f"  ERROR [{page:20s}]: {type(e).__name__}: {e}")
        all_ok = False

print()
print("Result:", "ALL PAGES PASSED" if all_ok else "SOME PAGES FAILED — see above")
