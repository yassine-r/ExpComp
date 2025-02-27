import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

from expcomp import logger
from expcomp.evaluation import Metric

from .experiment_config import ExperimentConfig
from .experiment import Experiment

import itertools


class ExperimentLoader:
    """
    A utility class to load experiments from various sources.
    """

    @staticmethod
    def from_directory(
        directory_path: Union[str, Path],
        config_file_pattern: str,
        metrics_file_pattern: str,
        parser: Optional[Callable[[Dict], Tuple[Dict, List[Dict]]]] = None,
        config_merge_strategy: str = "merge",
        metrics_merge_strategy: str = "append",
        custom_config_merger: Optional[Callable[[List[Dict]], Dict]] = None,
        custom_metrics_merger: Optional[Callable[[List[Dict]], List[Dict]]] = None,
        custom_file_loader: Optional[Callable[[Path], Union[Dict, List]]] = None,
    ) -> List[Experiment]:
        """
        Load experiments from a directory.

        Each experiment folder needs to have at least one config file and one metrics file.
        By default:
          - Multiple config files are merged (config_merge_strategy='merge').
          - Multiple metrics files are appended (metrics_merge_strategy='append').

        Args:
            directory_path: Path to the directory containing experiment subfolders.
            config_file_pattern: Glob pattern for config files (e.g., "config*.json").
            metrics_file_pattern: Glob pattern for metrics files (e.g., "metrics*.json").
            parser: Optional function to parse the combined file data into (config_dict, metrics_list).
                    If not provided, code assumes the config data is a dict and metrics data is
                    either a list or a dict with 'metrics' key.
            config_merge_strategy: Strategy for multiple config files
                                   ('merge', 'first', 'latest', 'custom').
            metrics_merge_strategy: Strategy for multiple metrics files
                                    ('merge', 'first', 'latest', 'custom').
            custom_config_merger: Custom function to merge multiple config dicts if
                                  config_merge_strategy='custom'.
            custom_metrics_merger: Custom function to merge multiple metrics lists/dicts if
                                   metrics_merge_strategy='custom'.
            custom_file_loader: Optional function to load files with custom parsing.

        Returns:
            A list of Experiment objects.
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            logger.error(
                f"Directory does not exist or is not a directory: {directory_path}"
            )
            raise ValueError(
                f"Directory does not exist or is not a directory: {directory_path}"
            )

        logger.info(f"Loading experiments from {directory}")
        experiments: List[Experiment] = []

        # Identify subdirectories for experiments
        subdirs = [d for d in directory.iterdir() if d.is_dir()]
        logger.debug(f"Found {len(subdirs)} potential experiment directories")

        for exp_dir in subdirs:
            try:
                logger.debug(f"Processing experiment directory: {exp_dir}")

                # Find config and metrics files
                config_files = list(exp_dir.glob(config_file_pattern))
                if not config_files:
                    logger.warning(
                        f"No config file matching '{config_file_pattern}' found in {exp_dir}, skipping."
                    )
                    continue

                metrics_files = list(exp_dir.glob(metrics_file_pattern))
                if not metrics_files:
                    logger.warning(
                        f"No metrics file matching '{metrics_file_pattern}' found in {exp_dir}, skipping."
                    )
                    continue

                # 1) Handle configs
                config_data = ExperimentLoader._load_and_merge_files(
                    config_files,
                    config_merge_strategy,
                    custom_config_merger,
                    custom_file_loader,
                )

                # 2) Handle metrics
                metrics_data = ExperimentLoader._load_and_merge_files(
                    metrics_files,
                    metrics_merge_strategy,
                    custom_metrics_merger,
                    custom_file_loader,
                )

                # 3) Apply custom parser if provided
                if parser:
                    logger.debug("Applying custom parser...")
                    config_dict, metrics_list = parser(
                        {"config": config_data, "metrics": metrics_data}
                    )
                else:
                    # By default, we assume config_data is a dict
                    config_dict = config_data if isinstance(config_data, dict) else {}

                    # Ensure we have a list of metrics
                    if isinstance(metrics_data, list):
                        metrics_list = metrics_data
                    elif isinstance(metrics_data, dict) and "metrics" in metrics_data:
                        metrics_list = metrics_data["metrics"]
                    else:
                        # If single metric or dict without 'metrics' key
                        metrics_list = [metrics_data] if metrics_data else []

                # 4) Create ExperimentConfig
                experiment_config = ExperimentConfig(**config_dict)
                logger.debug(f"Created config with ID: {experiment_config.id}")

                # 5) Create metrics (each a Metric object)
                metrics_objs = []
                for metric_dict in metrics_list:
                    if "experiment_id" not in metric_dict:
                        metric_dict["experiment_id"] = experiment_config.id

                    metrics_objs.append(Metric.from_dict(metric_dict))

                # 6) Build Experiment object
                experiment = Experiment(config=experiment_config, metrics=metrics_objs)
                experiments.append(experiment)
                logger.debug(f"Added experiment with {len(metrics_objs)} metrics")

            except Exception as e:
                logger.exception(
                    f"Error loading experiment from {exp_dir}, exception: {e}"
                )
                # Continue with next directory

        logger.info(f"Successfully loaded {len(experiments)} experiments")
        return experiments

    @staticmethod
    def flatten(item):
        if not isinstance(item, list):
            return item
        else:
            fl_list = []
            for i in item:
                if not isinstance(i, list):
                    fl_list.append(i)
                else:
                    fl_list.extend(ExperimentLoader.flatten(i))
        return fl_list

    @staticmethod
    def _load_and_merge_files(
        files: List[Path],
        strategy: str,
        custom_merger: Optional[Callable[[List[Any]], Any]],
        custom_file_loader: Optional[Callable[[Path], Union[Dict, List]]],
    ) -> Union[Dict, List, None]:
        """
        Helper to load file(s) depending on the merge strategy.

        Strategies:
          - 'first':  Use the first (oldest by mtime) file.
          - 'latest': Use the latest file.
          - 'custom': Use a custom_merger function to combine data from all files.
          - 'append': Return a list of all loaded data.
          - 'merge':  Merge dictionary data (keys updated). return dict

        Returns a dict, list, or None if no files found (shouldn't happen in practice if caller checks).
        """
        if not files:
            return None

        # 'first' or 'latest' => pick one file
        if strategy in ("first", "latest"):
            # Sort by modification time
            sorted_files = sorted(files, key=lambda f: f.stat().st_mtime)
            file_to_load = sorted_files[0] if strategy == "first" else sorted_files[-1]

            return ExperimentLoader.flatten(
                ExperimentLoader._load_file(file_to_load, custom_file_loader)
            )

        if strategy == "custom":
            if not custom_merger:
                logger.error("Custom merger function is required for 'custom' strategy")
                raise ValueError(
                    "Custom merger function is required for 'custom' strategy"
                )
            loaded_data = [
                ExperimentLoader._load_file(f, custom_file_loader) for f in files
            ]
            return custom_merger(loaded_data)

        if strategy == "append":
            return ExperimentLoader.flatten(
                [ExperimentLoader._load_file(f, custom_file_loader) for f in files]
            )

        if strategy == "merge":
            merged_dict: Dict[Any, Any] = {}
            merged_list: List[Any] = []

            for file_path in files:
                data = ExperimentLoader._load_file(file_path, custom_file_loader)

                if isinstance(data, dict):
                    # Merge dictionary into merged_dict (shallow update)
                    merged_dict.update(data)
                elif isinstance(data, list[dict]):
                    # Extend the merged_list with any lists we encounter
                    merged_list.extend(data)
                    for item in merged_list:
                        if isinstance(item, dict):
                            merged_dict.update(item)
                else:
                    logger.exception(f"Unsupported data type in file: {file_path}")
                    raise ValueError(f"Unsupported data type in file: {file_path}")

            return merged_dict

        # Fallback: unknown strategy
        raise ValueError(f"Unknown merge strategy: {strategy}")

    @staticmethod
    def _load_file(
        file_path: Path,
        custom_loader: Optional[Callable[[Path], Union[Dict, List]]] = None,
    ) -> Union[Dict, List]:
        """
        Load and parse a file. Supports .json, .yaml, .yml by default, or uses
        a custom loader if provided.

        Args:
            file_path: Path to the file.
            custom_loader: Optional callable for custom parsing.

        Returns:
            The parsed data (dict or list).

        Raises:
            ValueError: If the file extension is not supported or parsing fails.
            FileNotFoundError: If the file does not exist.
        """
        if custom_loader:
            return custom_loader(file_path)

        supported_extensions = {".json", ".yaml", ".yml"}
        extension = file_path.suffix.lower()

        if extension not in supported_extensions:
            raise ValueError(
                f"Unsupported file extension: {extension}. "
                f"Supported: {', '.join(supported_extensions)}. "
                f"Use custom_loader to handle this extension."
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if extension == ".json":
                    return json.load(f)
                else:  # .yaml or .yml
                    return yaml.safe_load(f)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Failed to parse {extension} file: {e}") from e
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
