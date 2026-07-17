"""Dependency-free test runner: python3 precision_matching/tests/run_all.py
(also compatible with pytest if you have it installed)."""

import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

MODULES = [
    "precision_matching.tests.test_input_gen",
    "precision_matching.tests.test_dense_metrics",
    "precision_matching.tests.test_detection_matching",
    "precision_matching.tests.test_decision_agreement",
    "precision_matching.tests.test_postprocess",
    "precision_matching.tests.test_real_text_and_task_level",
]


def main():
    passed, failed = 0, 0
    for name in MODULES:
        mod = importlib.import_module(name)
        for attr in sorted(dir(mod)):
            if not attr.startswith("test_"):
                continue
            try:
                getattr(mod, attr)()
                passed += 1
                print(f"PASS {name.split('.')[-1]}::{attr}")
            except Exception:
                failed += 1
                print(f"FAIL {name.split('.')[-1]}::{attr}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
