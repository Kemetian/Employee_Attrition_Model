import pandas as pd
import numpy as np
import pytest
import sys
sys.path.insert(0, "src")

from preprocessing import fill_missing_with_median, normalize_column, encode_categorical_column, encode_binary_column, remove_outliers, create_age_bins

def test_fill_missing_replaces_nulls():
    """Median fill should replace NaN values with the column median."""
    df = pd.DataFrame({
        "age": [20.0, 30.0, np.nan, 40.0, 50.0]
    })

    result = fill_missing_with_median(df, ["age"])

    assert result["age"].isna().sum() == 0, "There should be no missing values after filling"
    assert result["age"].iloc[2] == 35.0, "Missing value should be filled with median (35.0)"

def test_fill_missing_does_not_modify_original():
    """The original dataframe should not be changed."""
    df = pd.DataFrame({
        "age": [20.0, np.nan, 40.0]
    })
    original_null_count = df["age"].isna().sum()

    fill_missing_with_median(df, ["age"])

    assert df["age"].isna().sum() == original_null_count, \
        "Original dataframe should not be modified"

def test_fill_missing_handles_no_nulls():
    """If there are no missing values, the data should be unchanged."""
    df = pd.DataFrame({
        "age": [20.0, 30.0, 40.0]
    })

    result = fill_missing_with_median(df, ["age"])

    pd.testing.assert_frame_equal(result, df)

def test_fill_missing_multiple_columns():
    """Should handle filling multiple columns at once."""
    df = pd.DataFrame({
        "age": [20.0, np.nan, 40.0],
        "income": [50000.0, 60000.0, np.nan]
    })

    result = fill_missing_with_median(df, ["age", "income"])

    assert result["age"].isna().sum() == 0
    assert result["income"].isna().sum() == 0

def test_encode_binary_column():
    """Binary encoding should convert the positive value to 1 and the other to 0."""
    df = pd.DataFrame({
        "gender": ["male", "female", "male", "female"]
    })

    encode_binary_column(df, "gender", "male")

    assert (df["gender"] == [1, 0, 1, 0]).all(), "Binary encoding did not produce expected results"

def test_encode_binary_column_raises_on_bad_column():
    """Should raise ValueError if column does not exist."""
    df = pd.DataFrame({
        "gender": ["male", "female"]
    })

    with pytest.raises(ValueError, match="not found"):
        encode_binary_column(df, "nonexistent_column", "male")

def test_normalize_column():
    """Normalization should scale the column to have mean 0 and standard deviation 1."""
    df = pd.DataFrame({
        "age": [20.0, 30.0, 40.0, 50.0]
    })

    normalize_column(df, "age")

    assert np.isclose(df["age"].mean(), 0), "Normalized column should have mean 0"
    assert np.isclose(df["age"].std(), 1), "Normalized column should have standard deviation 1"

def test_remove_outliers():
    """Should remove values that are considered outliers."""
    df = pd.DataFrame({
        "age": [20.0, 30.0, 40.0, 1000.0]  # 1000 is an outlier
    })

    result = remove_outliers(df, "age")

    assert 1000.0 not in result["age"].values, "Outlier should be removed"
    assert len(result) == 3, "Only non-outlier rows should remain"

def test_encode_categorical_column():
    """Categorical encoding should convert categories to numerical codes."""
    df = pd.DataFrame({
        "color": ["red", "blue", "green", "red"]
    })

    encode_categorical_column(df, "color")

    assert (df["color"] == [0, 1, 2, 0]).all(), "Categorical encoding did not produce expected results"
