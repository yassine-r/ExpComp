import pytest
import json
from expcomp.experiments import ExperimentConfig


@pytest.fixture
def basic_config():
    return ExperimentConfig(id="test_exp", learning_rate=0.01, epochs=10)


@pytest.fixture
def complex_config():
    return ExperimentConfig(
        id=123,
        model_type="CNN",
        hyperparams={"dropout": 0.5, "batch_size": 32},
        metrics=["accuracy", "f1_score"],
        use_augmentation=True,
    )


def test_init_basic():
    config = ExperimentConfig(id="test1", param1=10, param2="value")
    assert config.id == "test1"
    assert config.param1 == 10
    assert config.param2 == "value"


def test_get(basic_config):
    assert basic_config.get("learning_rate") == 0.01
    assert basic_config.get("epochs") == 10
    assert basic_config.get("nonexistent") is None
    assert basic_config.get("nonexistent", "default") == "default"


def test_update(basic_config):
    basic_config.update(learning_rate=0.02, new_param="new_value")
    assert basic_config.learning_rate == 0.02
    assert basic_config.new_param == "new_value"
    assert basic_config.epochs == 10  # Original parameter should remain


def test_to_dict(complex_config):
    config_dict = complex_config.to_dict()
    assert config_dict["id"] == 123
    assert config_dict["model_type"] == "CNN"
    assert config_dict["hyperparams"]["dropout"] == 0.5
    assert config_dict["metrics"] == ["accuracy", "f1_score"]
    assert config_dict["use_augmentation"] is True


def test_to_json_basic(basic_config):
    json_str = basic_config.to_json()
    parsed = json.loads(json_str)
    assert parsed["id"] == "test_exp"
    assert parsed["learning_rate"] == 0.01
    assert parsed["epochs"] == 10


def test_to_json_complex(complex_config):
    json_str = complex_config.to_json(indent=None)
    parsed = json.loads(json_str)
    assert parsed["id"] == 123
    assert parsed["hyperparams"]["batch_size"] == 32
    assert parsed["metrics"] == ["accuracy", "f1_score"]


def test_to_json_non_serializable():
    # Test with a non-serializable object
    class NonSerializable:
        def __init__(self):
            pass

    config = ExperimentConfig(id="test", non_serializable=NonSerializable())
    json_str = config.to_json()
    parsed = json.loads(json_str)
    assert isinstance(parsed["non_serializable"], str)


def test_from_dict():
    config_dict = {"id": "from_dict_test", "learning_rate": 0.005, "batch_size": 64}
    config = ExperimentConfig.from_dict(config_dict)
    assert config.id == "from_dict_test"
    assert config.learning_rate == 0.005
    assert config.batch_size == 64


def test_repr(basic_config):
    repr_str = repr(basic_config)
    assert "ExperimentConfig" in repr_str
    assert "id=test_exp" in repr_str
    assert "learning_rate=0.01" in repr_str
    assert "epochs=10" in repr_str


def test_str(basic_config):
    str_output = str(basic_config)
    assert str_output == "ExperimentConfig_test_exp"


def test_str_numeric_id():
    config = ExperimentConfig(id=42)
    str_output = str(config)
    assert str_output == "ExperimentConfig_42"
