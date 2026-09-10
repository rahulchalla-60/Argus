import inspect
import os
import sys
import traceback
import importlib
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEST_MODULES = [
    "tests.test_graph_engine",
    "tests.test_feature_engine",
    "tests.test_integration_phase2",
    "tests.test_rule_engine",
    "tests.test_risk_engine",
    "tests.test_realtime_scoring",
    "tests.test_alert_engine",
    "tests.test_risk_propagation",
    "tests.test_phase4",
    "tests.test_explainability",
]

def main():
    total_run = 0
    total_passed = 0
    failures = []

    print("=" * 60)
    print("ARGUS TEST RUNNER - RUNNING ALL TEST SUITES")
    print("=" * 60)

    for mod_name in TEST_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            test_funcs = [
                (name, fn)
                for name, fn in inspect.getmembers(mod, inspect.isfunction)
                if name.startswith("test_")
            ]
            print(f"\n[SUITE] {mod_name} ({len(test_funcs)} tests)")
            for name, fn in test_funcs:
                total_run += 1
                try:
                    fn()
                    total_passed += 1
                    print(f"  [PASS] {name}")
                except Exception as e:
                    print(f"  [FAIL] {name}: {e}")
                    failures.append((mod_name, name, traceback.format_exc()))
        except Exception as e:
            print(f"Failed to import {mod_name}: {e}")
            failures.append((mod_name, "IMPORT_ERROR", traceback.format_exc()))

    print("\n" + "=" * 60)
    print(f"SUMMARY: {total_passed}/{total_run} PASSED")
    print("=" * 60)

    if failures:
        print(f"\n[FAILURES] {len(failures)} FAILURES DETECTED:")
        for mod, name, tb in failures:
            print(f"\n--- {mod}.{name} ---")
            print(tb)
        sys.exit(1)
    else:
        print("\nALL TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
