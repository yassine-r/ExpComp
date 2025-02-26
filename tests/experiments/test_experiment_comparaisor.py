import pytest
import pandas as pd
import numpy as np
from typing import List

from expcomp import Condition
from expcomp.evaluation import Metric
from expcomp.experiments import Experiment, ExperimentConfig, ExperimentComparison

@pytest.fixture
def mock_experiments():
    """Fixture to create a standard set of mock experiments."""
    experiments = []
    
    for i in range(3):
        # Create config
        config = ExperimentConfig(
            id=f"exp_{i}",
            model_type=f"model_{i % 2}",
            learning_rate=0.01 * (i + 1),
            batch_size=32 * (i + 1)
        )
        
        # Create metrics
        metrics = [
            Metric(experiment_id=f"exp_{i}", name="accuracy", value=0.8 + 0.05 * i),
            Metric(experiment_id=f"exp_{i}", name="f1_score", value=0.75 + 0.05 * i),
            Metric(experiment_id=f"exp_{i}", name="precision", value=0.7 + 0.05 * i)
        ]
        
        # Create experiment
        experiment = Experiment(config=config, evaluations=metrics)
        experiments.append(experiment)
    
    return experiments

@pytest.fixture
def comparison(mock_experiments):
    """Fixture to create an ExperimentComparison instance."""
    return ExperimentComparison(mock_experiments)

def test_init(comparison, mock_experiments):
    """Test initialization of ExperimentComparison."""
    assert len(comparison.experiments) == 3
    assert isinstance(comparison.df, pd.DataFrame)

def test_create_comparison_dataframe(comparison):
    """Test that the DataFrame is created with proper columns."""
    df = comparison.df
    
    # Check that we have the right number of rows
    assert len(df) == 3
    
    # Check column structure
    assert "experiment_id" in df.columns
    assert "config_model_type" in df.columns
    assert "config_learning_rate" in df.columns
    assert "config_batch_size" in df.columns
    assert "metric_accuracy" in df.columns
    assert "metric_f1_score" in df.columns
    assert "metric_precision" in df.columns
    
    # Check data values
    assert df.iloc[0]["experiment_id"] == "exp_0"
    assert df.iloc[0]["config_learning_rate"] == pytest.approx(0.01)
    assert df.iloc[0]["metric_accuracy"] == pytest.approx(0.8)
    
    assert df.iloc[2]["experiment_id"] == "exp_2"
    assert df.iloc[2]["config_batch_size"] == 96
    assert df.iloc[2]["metric_f1_score"] == pytest.approx(0.85)

def test_filter_experiments_single_condition(comparison):
    """Test filtering with a single condition."""
    # Create a condition to filter model_type == 'model_0'
    condition = Condition(key="config_model_type", operator="==", value="model_0")
    
    filtered_df = comparison.filter_experiments([condition])
    
    # Should return only experiments with model_type = 'model_0'
    assert len(filtered_df) == 2  # exp_0 and exp_2
    assert filtered_df.iloc[0]["experiment_id"] == "exp_0"
    assert filtered_df.iloc[1]["experiment_id"] == "exp_2"

def test_filter_experiments_multiple_conditions(comparison):
    """Test filtering with multiple conditions."""
    # Create conditions: model_type == 'model_0' AND accuracy > 0.85
    condition1 = Condition(key="config_model_type", operator="==", value="model_0")
    condition2 = Condition(key="metric_accuracy", operator=">", value=0.85)
    
    filtered_df = comparison.filter_experiments([condition1, condition2])
    
    # Should only return exp_2
    assert len(filtered_df) == 1
    assert filtered_df.iloc[0]["experiment_id"] == "exp_2"

def test_filter_experiments_no_matches(comparison):
    """Test filtering with conditions that match no experiments."""
    # Create impossible condition
    condition = Condition(key="metric_accuracy", operator=">", value=0.95)
    
    filtered_df = comparison.filter_experiments([condition])
    
    # Should return empty DataFrame
    assert len(filtered_df) == 0

def test_sort_by_metric_ascending(comparison):
    """Test sorting by a metric in ascending order."""
    sorted_df = comparison.sort_by_metric("accuracy", ascending=True)
    
    # Check if sorted properly
    accuracy_values = sorted_df["metric_accuracy"].tolist()
    assert accuracy_values == sorted(accuracy_values)
    assert sorted_df.iloc[0]["experiment_id"] == "exp_0"
    assert sorted_df.iloc[2]["experiment_id"] == "exp_2"

def test_sort_by_metric_descending(comparison):
    """Test sorting by a metric in descending order."""
    sorted_df = comparison.sort_by_metric("f1_score", ascending=False)
    
    # Check if sorted properly
    f1_values = sorted_df["metric_f1_score"].tolist()
    assert f1_values == sorted(f1_values, reverse=True)
    assert sorted_df.iloc[0]["experiment_id"] == "exp_2"
    assert sorted_df.iloc[2]["experiment_id"] == "exp_0"

def test_sort_by_nonexistent_metric(comparison):
    """Test that sorting by a non-existent metric raises an error."""
    with pytest.raises(ValueError) as excinfo:
        comparison.sort_by_metric("nonexistent_metric")
    
    assert "not found in DataFrame columns" in str(excinfo.value)

def test_empty_experiments_list():
    """Test behavior with empty experiments list."""
    empty_comparison = ExperimentComparison([])
    
    # Should create an empty DataFrame
    assert len(empty_comparison.df) == 0
    
    # Filtering should raise an exception when a condition is applied to empty DataFrame
    condition = Condition(key="any_key", operator="==", value="any_value")
    with pytest.raises(KeyError):
        empty_comparison.filter_experiments([condition])

def test_custom_condition_with_function(comparison):
    """Test filtering with a custom condition using a function."""
    # Define a custom filtering function
    def custom_filter(df, key, value):
        return df[df["metric_accuracy"] > 0.8]
    
    # Create custom condition
    custom_condition = Condition(key="custom_filter", operator="custom", 
                                value=None, custom_fn=custom_filter)
    
    filtered_df = comparison.filter_experiments([custom_condition])
    
    # Should return exp_1 and exp_2 (accuracy > 0.8)
    assert len(filtered_df) == 2
    assert "exp_1" in filtered_df["experiment_id"].values
    assert "exp_2" in filtered_df["experiment_id"].values

# Parameterized test for different operators
@pytest.mark.parametrize("operator,value,expected_count", [
    ("==", "model_0", 2),      # exp_0 and exp_2
    (">", 0.85, 2),           # exp_1 and exp_2 (for accuracy)
    ("<=", 0.8, 1),           # only exp_0 (for accuracy)
    ("!=", "model_0", 1),     # only exp_1
])
def test_filter_with_different_operators(comparison, operator, value, expected_count):
    """Test filtering with different operators."""
    key = "config_model_type" if operator in ["==", "!="] else "metric_accuracy"
    condition = Condition(key=key, operator=operator, value=value)
    filtered_df = comparison.filter_experiments([condition])
    assert len(filtered_df) == expected_count

def test_experiment_with_missing_metric(mock_experiments):
    """Test handling an experiment with a missing metric."""
    # Create experiment with missing metric
    config = ExperimentConfig(id="exp_missing", model_type="model_x", learning_rate=0.05)
    metrics = [
        Metric(experiment_id="exp_missing", name="accuracy", value=0.9)
        # Missing f1_score and precision
    ]
    exp_missing = Experiment(config=config, evaluations=metrics)
    
    # Add to experiments and create new comparison
    experiments_with_missing = mock_experiments + [exp_missing]
    comparison = ExperimentComparison(experiments_with_missing)
    
    # Verify the DataFrame has NaN values for missing metrics
    df = comparison.df
    assert len(df) == 4
    assert pd.isna(df.loc[df["experiment_id"] == "exp_missing", "metric_f1_score"]).all()
    assert pd.isna(df.loc[df["experiment_id"] == "exp_missing", "metric_precision"]).all()

def test_experiments_with_different_config_parameters():
    """Test handling experiments with different config parameters."""
    # Create experiments with different parameters
    config1 = ExperimentConfig(id="exp_A", model_type="A", param_A=1.0)
    config2 = ExperimentConfig(id="exp_B", model_type="B", param_B=2.0)
    
    metrics1 = [Metric(experiment_id="exp_A", name="accuracy", value=0.9)]
    metrics2 = [Metric(experiment_id="exp_B", name="accuracy", value=0.8)]
    
    exp1 = Experiment(config=config1, evaluations=metrics1)
    exp2 = Experiment(config=config2, evaluations=metrics2)
    
    comparison = ExperimentComparison([exp1, exp2])
    df = comparison.df
    
    # Check that all parameters are included
    assert "config_param_A" in df.columns
    assert "config_param_B" in df.columns
    
    # Check that missing values are NaN
    assert pd.isna(df.loc[df["experiment_id"] == "exp_A", "config_param_B"]).all()
    assert pd.isna(df.loc[df["experiment_id"] == "exp_B", "config_param_A"]).all()

def test_experiments_with_different_metrics():
    """Test handling experiments with different metrics."""
    # Create experiments with different metrics
    config1 = ExperimentConfig(id="exp_C", model_type="C")
    config2 = ExperimentConfig(id="exp_D", model_type="D")
    
    metrics1 = [Metric(experiment_id="exp_C", name="accuracy", value=0.9)]
    metrics2 = [Metric(experiment_id="exp_D", name="f1_score", value=0.8)]
    
    exp1 = Experiment(config=config1, evaluations=metrics1)
    exp2 = Experiment(config=config2, evaluations=metrics2)
    
    comparison = ExperimentComparison([exp1, exp2])
    df = comparison.df
    
    # Check that all metrics are included
    assert "metric_accuracy" in df.columns
    assert "metric_f1_score" in df.columns
    
    # Check that missing values are NaN
    assert pd.isna(df.loc[df["experiment_id"] == "exp_C", "metric_f1_score"]).all()
    assert pd.isna(df.loc[df["experiment_id"] == "exp_D", "metric_accuracy"]).all()

def test_filter_with_contains_operator():
    """Test filtering with 'contains' operator."""
    # Create experiments
    config1 = ExperimentConfig(id="exp_model_A", model_type="neural_network")
    config2 = ExperimentConfig(id="exp_model_B", model_type="decision_tree")
    
    metrics1 = [Metric(experiment_id="exp_model_A", name="accuracy", value=0.9)]
    metrics2 = [Metric(experiment_id="exp_model_B", name="accuracy", value=0.8)]
    
    exp1 = Experiment(config=config1, evaluations=metrics1)
    exp2 = Experiment(config=config2, evaluations=metrics2)
    
    comparison = ExperimentComparison([exp1, exp2])
    
    # Filter for models containing "neural"
    condition = Condition(key="config_model_type", operator="contains", value="neural")
    filtered_df = comparison.filter_experiments([condition])
    
    # Should only return the neural_network experiment
    assert len(filtered_df) == 1
    assert filtered_df.iloc[0]["experiment_id"] == "exp_model_A"
