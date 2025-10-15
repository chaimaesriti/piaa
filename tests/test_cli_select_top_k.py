"""
Test CLI --select --select-top-k functionality end-to-end
"""
import sys
import os
import subprocess
import pandas as pd

def test_select_top_k():
    """Test --select-top-k parameter in CLI"""
    print("=" * 60)
    print("CLI Feature Selection Top-K Test")
    print("=" * 60)

    input_file = "data/sample_train.csv"
    output_file = "data/test_output_top_k.csv"

    # Test 1: Select top 3
    print("\n" + "=" * 60)
    print("Test 1: Select top 3 features")
    print("=" * 60)

    cmd = [
        "python", "transform_data.py",
        input_file,
        "--target", "target",
        "--select",
        "--select-top-k", "3",
        "--output", output_file
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    # Check output file
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print(f"\nOutput shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Should have 3 features + target = 4 columns
        assert df.shape[1] == 4, f"Expected 4 columns (3 features + target), got {df.shape[1]}"
        assert 'target' in df.columns, "Target should be in output"

        print("✓ Test 1 passed: Top 3 features selected")
        os.remove(output_file)
    else:
        print("✗ Output file not created")
        return False

    # Test 2: Select top 5
    print("\n" + "=" * 60)
    print("Test 2: Select top 5 features")
    print("=" * 60)

    cmd = [
        "python", "transform_data.py",
        input_file,
        "--target", "target",
        "--select",
        "--select-top-k", "5",
        "--output", output_file
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print(f"Output shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Should have 5 features + target = 6 columns
        assert df.shape[1] == 6, f"Expected 6 columns (5 features + target), got {df.shape[1]}"
        assert 'target' in df.columns, "Target should be in output"

        print("✓ Test 2 passed: Top 5 features selected")
        os.remove(output_file)
    else:
        print("✗ Output file not created")
        return False

    # Test 3: With filtering + selection
    print("\n" + "=" * 60)
    print("Test 3: Filter + Select top 3")
    print("=" * 60)

    cmd = [
        "python", "transform_data.py",
        input_file,
        "--target", "target",
        "--filter",
        "--select",
        "--select-top-k", "3",
        "--output", output_file
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print(f"Output shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Should have at most 3 features + target
        assert df.shape[1] <= 4, f"Expected at most 4 columns, got {df.shape[1]}"
        assert 'target' in df.columns, "Target should be in output"

        print("✓ Test 3 passed: Filter + Select works together")
        os.remove(output_file)
    else:
        print("✗ Output file not created")
        return False

    # Test 4: Different selection methods
    print("\n" + "=" * 60)
    print("Test 4: Custom selection methods with top-k")
    print("=" * 60)

    cmd = [
        "python", "transform_data.py",
        input_file,
        "--target", "target",
        "--select",
        "--select-methods", "mutual_info", "correlation",
        "--select-top-k", "4",
        "--output", output_file
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print(f"Output shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Should have 4 features + target = 5 columns
        assert df.shape[1] == 5, f"Expected 5 columns, got {df.shape[1]}"
        assert 'target' in df.columns, "Target should be in output"

        print("✓ Test 4 passed: Custom methods with top-k works")
        os.remove(output_file)
    else:
        print("✗ Output file not created")
        return False

    print("\n" + "=" * 60)
    print("ALL CLI TOP-K TESTS PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_select_top_k()
    sys.exit(0 if success else 1)
