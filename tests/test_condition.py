import pytest
import pandas as pd
import numpy as np
from expcomp import Condition


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["foo", "bar", "baz", "qux", "quux"],
            "C": [1.1, 2.2, 3.3, 4.4, np.nan],
            "D": [True, False, True, False, True],
        }
    )


def test_init_valid():
    cond = Condition("A", "==", 1)
    assert cond.key == "A"
    assert cond.operator == "=="
    assert cond.value == 1

    def custom_fn(df, key, value):
        return df[df[key] == value]

    cond = Condition("A", "custom", 1, custom_fn)
    assert cond.custom_fn == custom_fn


def test_init_invalid():
    with pytest.raises(ValueError):
        Condition("A", "custom", 1)


def test_eq_operator(sample_df):
    cond = Condition("A", "==", 1)
    result = cond.apply(sample_df)
    assert len(result) == 1
    assert result.iloc[0]["A"] == 1


def test_neq_operator(sample_df):
    cond = Condition("A", "!=", 1)
    result = cond.apply(sample_df)
    assert len(result) == 4
    assert 1 not in result["A"].values


def test_gt_operator(sample_df):
    cond = Condition("A", ">", 3)
    result = cond.apply(sample_df)
    assert len(result) == 2
    assert all(val > 3 for val in result["A"])


def test_lt_operator(sample_df):
    cond = Condition("A", "<", 3)
    result = cond.apply(sample_df)
    assert len(result) == 2
    assert all(val < 3 for val in result["A"])


def test_gte_operator(sample_df):
    cond = Condition("A", ">=", 3)
    result = cond.apply(sample_df)
    assert len(result) == 3
    assert all(val >= 3 for val in result["A"])


def test_lte_operator(sample_df):
    cond = Condition("A", "<=", 3)
    result = cond.apply(sample_df)
    assert len(result) == 3
    assert all(val <= 3 for val in result["A"])


def test_contains_operator_string(sample_df):
    cond = Condition("B", "contains", "ba")
    result = cond.apply(sample_df)
    assert len(result) == 2
    assert all("ba" in val for val in result["B"])


def test_contains_operator_non_string(sample_df):
    cond = Condition("A", "contains", "1")
    result = cond.apply(sample_df)
    assert len(result) == 1
    assert result.iloc[0]["A"] == 1


def test_in_operator(sample_df):
    cond = Condition("A", "in", [1, 3, 5])
    result = cond.apply(sample_df)
    assert len(result) == 3
    assert all(val in [1, 3, 5] for val in result["A"])


def test_not_in_operator(sample_df):
    cond = Condition("A", "not in", [1, 3, 5])
    result = cond.apply(sample_df)
    assert len(result) == 2
    assert all(val not in [1, 3, 5] for val in result["A"])


def test_isna_operator(sample_df):
    cond = Condition("C", "isna", None)
    result = cond.apply(sample_df)
    assert len(result) == 1
    assert pd.isna(result.iloc[0]["C"])


def test_notna_operator(sample_df):
    cond = Condition("C", "notna", None)
    result = cond.apply(sample_df)
    assert len(result) == 4
    assert all(pd.notna(val) for val in result["C"])


def test_custom_operator(sample_df):
    def custom_filter(df, key, value):
        return df[df[key] % value == 0]

    cond = Condition("A", "custom", 2, custom_filter)
    result = cond.apply(sample_df)
    assert len(result) == 2
    assert all(val % 2 == 0 for val in result["A"])


def test_key_not_in_df(sample_df):
    cond = Condition("Z", "==", 1)
    with pytest.raises(KeyError):
        cond.apply(sample_df)


def test_unsupported_operator(sample_df):
    cond = Condition("A", "invalid_op", 1)
    with pytest.raises(ValueError):
        cond.apply(sample_df)
