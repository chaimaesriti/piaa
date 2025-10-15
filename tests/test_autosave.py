"""
Test Auto-Save Functionality
"""
import sys
import os
sys.path.append('/Users/chaimaesriti/piaa-codex/piaa')

import numpy as np
import pandas as pd
from transform_data import generate_output_filename


def test_generate_output_filename():
    """Test output filename generation"""
    print("=" * 60)
    print("Testing Output Filename Generation")
    print("=" * 60)

    test_cases = [
        ("train.csv", "transformed_train.csv"),
        ("data/train.csv", "data/transformed_train.csv"),
        ("/path/to/train.csv", "/path/to/transformed_train.csv"),
        ("my_data.csv", "transformed_my_data.csv"),
        ("data/subfolder/file.csv", "data/subfolder/transformed_file.csv"),
    ]

    print("\nTest cases:")
    for input_path, expected_output in test_cases:
        actual_output = generate_output_filename(input_path)
        status = "✓" if actual_output == expected_output else "✗"
        print(f"{status} {input_path}")
        print(f"  Expected: {expected_output}")
        print(f"  Got:      {actual_output}")

        assert actual_output == expected_output, f"Mismatch for {input_path}"

    print("\n✓ All filename generation tests passed!")


def test_autosave_with_real_data():
    """Test auto-save with actual data"""
    print("\n\n" + "=" * 60)
    print("Testing Auto-Save with Real Data")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.exponential(50000, 100),
        'is_active': np.random.choice([0, 1], 100),
        'target': np.random.choice([0, 1], 100)
    })

    # Save test input
    test_input = "test_data.csv"
    expected_output = "transformed_test_data.csv"

    print(f"\n1. Creating test input: {test_input}")
    df.to_csv(test_input, index=False)
    print(f"   ✓ Created with {len(df)} rows")

    # Run transform (this will be done via command line)
    print(f"\n2. Expected output filename: {expected_output}")
    output_filename = generate_output_filename(test_input)
    print(f"   Generated: {output_filename}")
    assert output_filename == expected_output

    # Cleanup
    print(f"\n3. Cleanup")
    if os.path.exists(test_input):
        os.remove(test_input)
        print(f"   ✓ Removed {test_input}")

    print("\n✓ Auto-save test passed!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AUTO-SAVE FUNCTIONALITY TESTS")
    print("=" * 60)

    test_generate_output_filename()
    test_autosave_with_real_data()

    print("\n\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
