import pandas as pd
from typing import List
from dataclasses import dataclass
from expcomp import Condition
from .experiment import Experiment


class ExperimentComparison:
    def __init__(self, experiments: List[Experiment]):
        self.experiments = experiments
        self.df = self.create_comparison_dataframe()

    def create_comparison_dataframe(self) -> pd.DataFrame:
        """
        Create a pandas DataFrame that aggregates experiment configurations
        and their associated metrics in a tabular format.
        
        Returns:
            A pandas DataFrame with one row per Experiment. Config parameters
            are prefixed with 'config_' and metrics with 'metric_'.
        """
        rows = []

        for experiment in self.experiments:
            row_data = {}

            # Include the experiment ID (from config)
            row_data["experiment_id"] = experiment.config.id

            # Flatten config parameters into row_data, prefixing keys with 'config_'
            config_dict = experiment.config.to_dict()
            for key, value in config_dict.items():
                # Skip the 'id' key to avoid duplication with experiment_id
                if key == "id":
                    continue
                row_data[f"config_{key}"] = value

            # Flatten metric values, prefixing with 'metric_'
            for metric in experiment.evaluations:
                # The metric's name becomes the column name
                col_name = f"metric_{metric.name}"
                row_data[col_name] = metric.value

            rows.append(row_data)

        return pd.DataFrame(rows)

    def filter_experiments(self, conditions: List[Condition]) -> pd.DataFrame:
        """
        Filter the comparison DataFrame by applying a list of Condition objects.
        
        Args:
            conditions: A list of Condition objects to apply sequentially.
        
        Returns:
            A filtered pandas DataFrame that satisfies all the conditions.
        """
        df_filtered = self.df.copy()
        for condition in conditions:
            df_filtered = condition.apply(df_filtered)
        return df_filtered

    def sort_by_metric(self, metric_name: str, ascending: bool = True) -> pd.DataFrame:
        """
        Sort the comparison DataFrame by the specified metric column.
        
        Args:
            metric_name: The metric name to sort by (e.g., 'accuracy').
            ascending: If True, sort in ascending order; otherwise, descending.
        
        Returns:
            A sorted pandas DataFrame.
        """
        col_name = f"metric_{metric_name}"
        if col_name not in self.df.columns:
            raise ValueError(f"Metric '{metric_name}' not found in DataFrame columns.")
        return self.df.sort_values(by=col_name, ascending=ascending)
