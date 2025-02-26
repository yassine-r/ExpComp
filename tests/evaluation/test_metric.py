import pytest
import json
from expcomp.evaluation import Metric

@pytest.fixture
def basic_metric():
    return Metric(experiment_id="exp_123", name="accuracy", value=0.95)

@pytest.fixture
def complex_metric():
    return Metric(
        experiment_id="exp_456",
        name="f1_score",
        value=0.87,
        units="percentage",
        threshold=0.8,
        direction="maximize",
        phase="validation"
    )

def test_init_basic():
    metric = Metric(experiment_id="test1", name="precision", value=0.82)
    assert metric.experiment_id == "test1"
    assert metric.name == "precision"
    assert metric.value == 0.82

def test_init_with_kwargs():
    metric = Metric(experiment_id="test1", name="loss", value=0.15, 
                   direction="minimize", baseline=0.2)
    assert metric.experiment_id == "test1"
    assert metric.name == "loss"
    assert metric.value == 0.15
    assert metric.direction == "minimize"
    assert metric.baseline == 0.2

def test_update_value(basic_metric):
    basic_metric.update_value(0.97)
    assert basic_metric.value == 0.97

def test_update(basic_metric):
    basic_metric.update(value=0.98, timestamp="2023-01-01", notes="Improved model")
    assert basic_metric.value == 0.98
    assert basic_metric.timestamp == "2023-01-01"
    assert basic_metric.notes == "Improved model"

def test_to_dict(complex_metric):
    metric_dict = complex_metric.to_dict()
    assert metric_dict["experiment_id"] == "exp_456"
    assert metric_dict["name"] == "f1_score"
    assert metric_dict["value"] == 0.87
    assert metric_dict["units"] == "percentage"
    assert metric_dict["threshold"] == 0.8
    assert metric_dict["direction"] == "maximize"
    assert metric_dict["phase"] == "validation"

def test_to_json_basic(basic_metric):
    json_str = basic_metric.to_json()
    parsed = json.loads(json_str)
    assert parsed["experiment_id"] == "exp_123"
    assert parsed["name"] == "accuracy"
    assert parsed["value"] == 0.95

def test_to_json_complex(complex_metric):
    json_str = complex_metric.to_json(indent=None)
    parsed = json.loads(json_str)
    assert parsed["experiment_id"] == "exp_456"
    assert parsed["name"] == "f1_score"
    assert parsed["value"] == 0.87
    assert parsed["units"] == "percentage"

def test_to_json_non_serializable():
    # Test with a non-serializable object
    class NonSerializable:
        def __init__(self):
            pass
    
    metric = Metric(experiment_id="test", name="custom", value=0.5, 
                  extra=NonSerializable())
    json_str = metric.to_json()
    parsed = json.loads(json_str)
    assert isinstance(parsed["extra"], str)

def test_from_dict():
    metric_dict = {
        "experiment_id": "exp_789",
        "name": "recall",
        "value": 0.76,
        "category": "classification"
    }
    metric = Metric.from_dict(metric_dict)
    assert metric.experiment_id == "exp_789"
    assert metric.name == "recall"
    assert metric.value == 0.76
    assert metric.category == "classification"

def test_repr(basic_metric):
    repr_str = repr(basic_metric)
    assert "Metric" in repr_str
    assert "experiment_id=exp_123" in repr_str
    assert "name=accuracy" in repr_str
    assert "value=0.95" in repr_str

def test_str(basic_metric):
    str_output = str(basic_metric)
    assert "Metric(experiment_id=exp_123, name=accuracy, value=0.95)" == str_output

def test_different_value_types():
    # Test with integer value
    int_metric = Metric(experiment_id="test", name="count", value=42)
    assert int_metric.value == 42
    
    # Test with list value
    list_metric = Metric(experiment_id="test", name="conf_matrix", value=[1, 2, 3, 4])
    assert list_metric.value == [1, 2, 3, 4]
    
    # Test with None value
    none_metric = Metric(experiment_id="test", name="pending", value=None)
    assert none_metric.value is None

def test_different_experiment_id_types():
    # Test with numeric experiment ID
    num_id_metric = Metric(experiment_id=123, name="accuracy", value=0.9)
    assert num_id_metric.experiment_id == 123
    
    # Test with tuple experiment ID
    tuple_id_metric = Metric(experiment_id=(1, "test"), name="accuracy", value=0.9)
    assert tuple_id_metric.experiment_id == (1, "test")