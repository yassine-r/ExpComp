import pytest
import json
import yaml
import time
from pathlib import Path
from typing import Any, Dict, List
from expcomp.evaluation import Metric
import wandb 

from expcomp.experiments import ExperimentLoader
from expcomp.experiments import ExperimentConfig
from expcomp.experiments import Experiment
import pytest
import json
import time
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Tuple, Union, Callable


@pytest.fixture
def single_experiment_dir(tmp_path) -> Path:
    """
    Creates a single experiment directory with one config file (JSON)
    and one metrics file (JSON).
    """
    exp_dir = tmp_path / "exp1"
    exp_dir.mkdir()

    # Write a config file
    config_data = {"id": "exp1", "learning_rate": 0.01}
    config_path = exp_dir / "config.json"
    config_path.write_text(json.dumps(config_data))

    # Write a metrics file
    metrics_data = [{"name": "accuracy", "value": 0.95}]
    metrics_path = exp_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_data))

    return tmp_path


@pytest.fixture
def multi_experiment_dir(tmp_path) -> Path:
    """
    Creates a directory with two sub-experiments, each having config & metrics files.
    Also creates multiple config files to test merging strategies.
    """
    # exp1 subdir
    exp1 = tmp_path / "exp1"
    exp1.mkdir()
    config_data1 = {"id": "exp1", "param1": 100}
    (exp1 / "config_1.json").write_text(json.dumps(config_data1))

    metrics_data1 = [{"name": "accuracy", "value": 0.90}]
    (exp1 / "metrics_1.json").write_text(json.dumps(metrics_data1))

    # A second config for exp1, to test merging or first/latest
    time.sleep(0.1)  # ensure the mtime is different
    config_data1b = {"id": "exp1", "param2": 999}
    (exp1 / "config_2.json").write_text(json.dumps(config_data1b))

    # exp2 subdir
    exp2 = tmp_path / "exp2"
    exp2.mkdir()
    config_data2 = {"id": "exp2", "alpha": 0.5}
    (exp2 / "config.json").write_text(json.dumps(config_data2))

    metrics_data2 = [{"name": "loss", "value": 0.12}]
    (exp2 / "metrics.json").write_text(json.dumps(metrics_data2))

    return tmp_path


def test_non_existing_directory():
    """
    from_directory should raise ValueError if the directory doesn't exist.
    """
    with pytest.raises(ValueError):
        ExperimentLoader.from_directory(
            directory_path="some/nonexistent/path",
            config_file_pattern="config*.json",
            metrics_file_pattern="metrics*.json",
        )


def test_no_config_match(single_experiment_dir):
    """
    If no config file matches, that subdir should be skipped.
    """
    # Use a pattern that doesn't match "config.json"
    experiments = ExperimentLoader.from_directory(
        directory_path=single_experiment_dir,
        config_file_pattern="does_not_match_*.json",
        metrics_file_pattern="metrics*.json",
    )
    # We expect 0 experiments loaded because the config pattern won't match
    assert len(experiments) == 0


def test_no_metrics_match(single_experiment_dir):
    """
    If no metrics file matches, that subdir should be skipped.
    """
    # Use a pattern that doesn't match "metrics.json"
    experiments = ExperimentLoader.from_directory(
        directory_path=single_experiment_dir,
        config_file_pattern="config*.json",
        metrics_file_pattern="does_not_match_*.json",
    )
    # We expect 0 experiments loaded because the metrics pattern won't match
    assert len(experiments) == 0


def test_single_experiment_load(single_experiment_dir):
    """
    Basic happy path: one subdir with one config & one metrics file.
    """
    experiments = ExperimentLoader.from_directory(
        directory_path=single_experiment_dir,
        config_file_pattern="config*.json",
        metrics_file_pattern="metrics*.json",
    )

    assert len(experiments) == 1
    exp = experiments[0]
    # Check the config & metrics loaded
    assert exp.config.id == "exp1"
    assert hasattr(exp.config, "learning_rate")
    assert exp.config.learning_rate == 0.01

    assert len(exp.metrics) == 1
    assert exp.metrics[0].name == "accuracy"
    assert exp.metrics[0].value == 0.95


def test_multi_experiment_load_default_merge(multi_experiment_dir):
    """
    Test that loading multiple experiments with default strategies
    merges multiple config files for each experiment (config_merge_strategy='merge')
    and appends metrics files (metrics_merge_strategy='append').
    """
    experiments = ExperimentLoader.from_directory(
        directory_path=multi_experiment_dir,
        config_file_pattern="config*.json",
        metrics_file_pattern="metrics*.json",
        # Default strategies: config_merge_strategy="merge", metrics_merge_strategy="append"
    )

    # We expect 2 experiments: exp1, exp2
    assert len(experiments) == 2

    # Sort by ID for consistency in checks
    experiments.sort(key=lambda e: e.config.id)

    # Check exp1
    exp1 = experiments[0]
    assert exp1.config.id == "exp1"
    # param1=100 (from config_1.json) and param2=999 (from config_2.json) should be merged
    assert exp1.config.param1 == 100
    assert exp1.config.param2 == 999

    # For metrics, we only had one metrics file in exp1 (metrics_1.json)
    assert len(exp1.metrics) == 1
    assert exp1.metrics[0].name == "accuracy"
    assert exp1.metrics[0].value == 0.90

    # Check exp2
    exp2 = experiments[1]
    assert exp2.config.id == "exp2"
    assert exp2.config.alpha == 0.5
    assert len(exp2.metrics) == 1
    assert exp2.metrics[0].name == "loss"
    assert exp2.metrics[0].value == 0.12


def test_multi_experiment_load_first_strategy(multi_experiment_dir):
    """
    Test that using config_merge_strategy='first' picks the oldest config file
    and ignores the others.
    Similarly, we can also test metrics_merge_strategy='first' to pick
    just the first metrics file if multiple existed.
    """
    experiments = ExperimentLoader.from_directory(
        directory_path=multi_experiment_dir,
        config_file_pattern="config*.json",
        metrics_file_pattern="metrics*.json",
        config_merge_strategy="first",
        metrics_merge_strategy="first",
    )
    # We still expect 2 experiments
    assert len(experiments) == 2

    # Sort by ID for consistency
    experiments.sort(key=lambda e: e.config.id)

    # For exp1, the 'first' config file was config_1.json => param1=100
    # param2 should *not* be present because config_2.json was created later.
    exp1 = experiments[0]
    assert exp1.config.id == "exp1"
    assert getattr(exp1.config, "param1", None) == 100
    # param2 should be missing if we used 'first'
    assert not hasattr(exp1.config, "param2")

    # For metrics, we only have one metrics file in each experiment,
    # so 'first' or 'append' doesn't really change much here.
    assert len(exp1.metrics) == 1
    assert exp1.metrics[0].name == "accuracy"

    # For exp2, check it loaded the single config
    exp2 = experiments[1]
    assert exp2.config.id == "exp2"
    assert exp2.config.alpha == 0.5
    assert len(exp2.metrics) == 1
    assert exp2.metrics[0].name == "loss"


def test_multi_experiment_load_latest_strategy(multi_experiment_dir):
    """
    Test that using config_merge_strategy='latest' picks the newest config file
    for each experiment and discards older ones.
    """
    experiments = ExperimentLoader.from_directory(
        directory_path=multi_experiment_dir,
        config_file_pattern="config*.json",
        metrics_file_pattern="metrics*.json",
        config_merge_strategy="latest",
        metrics_merge_strategy="append",
    )

    # We still expect 2 experiments
    assert len(experiments) == 2

    # Sort by ID for consistency
    experiments.sort(key=lambda e: e.config.id)

    # For exp1, 'latest' config file is config_2.json => param2=999
    exp1 = experiments[0]
    assert exp1.config.id == "exp1"
    assert getattr(exp1.config, "param1", None) is None
    assert exp1.config.param2 == 999

    # For exp2, there's only one config
    exp2 = experiments[1]
    assert exp2.config.id == "exp2"
    assert exp2.config.alpha == 0.5


def test_custom_config_merger(multi_experiment_dir):
    """
    Test a custom config merger function that concatenates param values into a list, for example.
    """

    def custom_merger(config_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        # All config dicts have "id" = "exp1", so let's keep that consistent
        merged = {}
        for conf in config_dicts:
            # The "id" is the same, so just set it once
            if "id" in conf:
                merged["id"] = conf["id"]
            # For other keys, collect them in a list
            for k, v in conf.items():
                if k == "id":
                    continue
                if k not in merged:
                    merged[k] = []
                merged[k].append(v)
        return merged

    experiments = ExperimentLoader.from_directory(
        directory_path=multi_experiment_dir,
        config_file_pattern="config*.json",
        metrics_file_pattern="metrics*.json",
        config_merge_strategy="custom",
        custom_config_merger=custom_merger,
        # Keep metrics default
    )

    # We expect 2 experiments
    # For exp1 => param1=[100], param2=[999]
    # For exp2 => alpha=[0.5]
    # (One experiment has multiple config files, the other has only one.)
    # We'll test only exp1 to ensure it used the custom merger
    exp1 = [e for e in experiments if e.config.id == "exp1"][0]
    # Because we appended, we should have:
    # param1=[100] and param2=[999]
    assert isinstance(exp1.config.param1, list)
    assert exp1.config.param1 == [100]
    assert exp1.config.param2 == [999]


def test_custom_parser(multi_experiment_dir):
    """
    Test providing a custom parser that interprets loaded data in a special way.
    The parser receives a dict with {"config": config_data, "metrics": metrics_data}
    and must return (config_dict, metrics_list).
    """

    def custom_parser(
        combined_data: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Suppose the config is stored under "config" => we rename a key from param1 => new_param.
        config = combined_data["config"]
        if isinstance(config, dict) and "param1" in config:
            config["renamed_param"] = config.pop("param1")

        # Suppose the metrics are stored under "metrics" => we'll add a new metric artificially.
        metrics_list = combined_data["metrics"]
        if isinstance(metrics_list, list):
            for m in metrics_list:
                # Just add a new field
                m["custom_flag"] = True
        return config, metrics_list

    experiments = ExperimentLoader.from_directory(
        directory_path=multi_experiment_dir,
        config_file_pattern="config_*.json",  # only pick config_*.json to ensure we see param1/param2
        metrics_file_pattern="metrics*.json",
        parser=custom_parser,
        # We'll keep merge strategies default
    )

    # We should get at least 1 experiment for exp1
    exp1 = [e for e in experiments if e.config.id == "exp1"][0]
    # The custom parser should rename param1 => renamed_param
    assert hasattr(exp1.config, "renamed_param")

    # Check that the metric got the custom_flag
    for m in exp1.metrics:
        assert getattr(m, "custom_flag", False) is True




######################


from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import json
import yaml
from pathlib import Path

def test_load_from_wandb_basic(monkeypatch):
    """
    Test basic loading from wandb with 2 mock runs
    """
    # Mock wandb API and runs
    mock_run1 = MagicMock()
    mock_run1.id = "run1"
    mock_run1.config = {"learning_rate": 0.01, "id": "run1"}
    mock_run1.history.return_value = pd.DataFrame([
        {"_step": 1, "accuracy": 0.8, "_runtime": 10},
        {"_step": 2, "accuracy": 0.85, "_runtime": 20}
    ])

    mock_run2 = MagicMock()
    mock_run2.id = "run2"
    mock_run2.config = {"dropout": 0.5, "id": "run2"}
    mock_run2.history.return_value = pd.DataFrame([
        {"_step": 1, "loss": 1.2, "_timestamp": 1620000000}
    ])

    # Mock API
    mock_api = MagicMock()
    mock_api.runs.return_value = [mock_run1, mock_run2]
    monkeypatch.setattr(wandb, "Api", lambda: mock_api)

    # Test loading
    experiments = ExperimentLoader.from_wandb(project="test_project")

    # Verify results
    assert len(experiments) == 2
    
    # Verify first experiment
    exp1 = next(e for e in experiments if e.config.id == "run1")
    assert exp1.config.learning_rate == 0.01
    assert len(exp1.metrics) == 2  # 2 steps, 1 metric per step (excluding system metrics)
    assert all(m.name == "accuracy" for m in exp1.metrics)
    assert {m.value for m in exp1.metrics} == {0.8, 0.85}

    # Verify second experiment
    exp2 = next(e for e in experiments if e.config.id == "run2")
    assert exp2.config.dropout == 0.5
    assert len(exp2.metrics) == 1
    assert exp2.metrics[0].name == "loss"

def test_wandb_local_saving(tmp_path):
    """
    Test loading from wandb and saving to local directory
    """
    # Setup mock run
    mock_run = MagicMock()
    mock_run.id = "saved_run"
    mock_run.config = {"epochs": 10, "id": "saved_run"}
    mock_run.history.return_value = pd.DataFrame([
        {"_step": 1, "f1_score": 0.75, "_system": 1}
    ])

    # Mock API
    with patch.object(wandb, "Api") as mock_api:
        mock_api.return_value.runs.return_value = [mock_run]

        # Test with local saving
        local_dir = tmp_path / "experiments"
        experiments = ExperimentLoader.from_wandb(
            project="save_test",
            local_dir=local_dir,
            config_file_name="config.yaml",
            metrics_file_name="metrics.json"
        )

    # Verify directory structure
    exp_dir = local_dir / "saved_run"
    assert exp_dir.exists()
    
    # Verify config file
    config_path = exp_dir / "config.yaml"
    assert config_path.exists()
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    assert config_data == {"epochs": 10, "id": "saved_run"}

    # Verify metrics file
    metrics_path = exp_dir / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path) as f:
        metrics_data = json.load(f)
    
    assert len(metrics_data) == 1
    assert metrics_data[0]["name"] == "f1_score"
    assert metrics_data[0]["value"] == 0.75
    assert "_system" not in metrics_data[0]  # Verify system metric exclusion

def test_wandb_system_metric_exclusion():
    """
    Test that system metrics (starting with _) are excluded by default
    """
    mock_run = MagicMock()
    mock_run.id = "sys_metrics_test"
    mock_run.config = {}
    mock_run.history.return_value = pd.DataFrame([
        {"_step": 1, "_runtime": 10, "loss": 1.0},
        {"_step": 2, "_runtime": 20, "loss": 0.5}
    ])

    with patch.object(wandb, "Api") as mock_api:
        mock_api.return_value.runs.return_value = [mock_run]
        experiments = ExperimentLoader.from_wandb(project="sys_metrics_test")

    assert len(experiments) == 1
    exp = experiments[0]
    assert len(exp.metrics) == 2  # Two loss metrics
    assert all(m.name == "loss" for m in exp.metrics)
    assert all(m.step in {1, 2} for m in exp.metrics)

def test_wandb_empty_metrics():
    """
    Test handling of runs with empty metrics
    """
    mock_run = MagicMock()
    mock_run.id = "empty_metrics"
    mock_run.config = {"model": "cnn"}
    mock_run.history.return_value = pd.DataFrame()  # Empty metrics

    with patch.object(wandb, "Api") as mock_api:
        mock_api.return_value.runs.return_value = [mock_run]
        experiments = ExperimentLoader.from_wandb(project="empty_metrics_test")

    assert len(experiments) == 1
    exp = experiments[0]
    assert len(exp.metrics) == 0

def test_wandb_custom_file_formats(tmp_path):
    """
    Test saving with different file formats (YAML/JSON)
    """
    mock_run = MagicMock()
    mock_run.id = "format_test"
    mock_run.config = {"batch_size": 32}
    mock_run.history.return_value = pd.DataFrame([{"accuracy": 0.99}])

    with patch.object(wandb, "Api") as mock_api:
        mock_api.return_value.runs.return_value = [mock_run]
        ExperimentLoader.from_wandb(
            project="format_test",
            local_dir=tmp_path,
            config_file_name="config.yml",
            metrics_file_name="metrics.yaml"
        )

    # Verify YAML config
    config_path = tmp_path / "format_test" / "config.yml"
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    assert config_data["batch_size"] == 32

    # Verify YAML metrics
    metrics_path = tmp_path / "format_test" / "metrics.yaml"
    with open(metrics_path) as f:
        metrics_data = yaml.safe_load(f)
    assert metrics_data[0]["name"] == "accuracy"