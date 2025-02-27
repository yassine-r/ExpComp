from typing import Any, Dict, Optional, Union, List
import json


class Metric:
    """
    A class to manage machine learning experiment metrics.

    Each metric is associated with exactly one experiment and can track
    various performance indicators with their values and metadata.
    """

    def __init__(
        self,
        experiment_id: Any,
        name: str,
        value: Optional[Union[float, int, List]] = None,
        **kwargs,
    ):
        """
        Initialize a metric for an experiment.

        Args:
            experiment_id: The ID of the associated experiment
            name: Name of the metric (e.g., 'accuracy', 'loss', 'f1_score')
            value: The value of the metric
            **kwargs: Additional metric attributes (e.g., units, threshold)
        """
        self.experiment_id = experiment_id
        self.name = name
        self.value = value

        # Store any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update_value(self, value: Union[float, int, List]) -> None:
        """
        Update the metric value.

        Args:
            value: New value for the metric
        """
        self.value = value

    def update(self, **kwargs) -> None:
        """
        Update multiple metric attributes at once.

        Args:
            **kwargs: Attributes to update
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the metric to a dictionary.

        Returns:
            A dictionary containing all metric attributes
        """
        return self.__dict__.copy()

    def to_json(self, indent: int = 2) -> str:
        """
        Convert the metric to a JSON string.

        Args:
            indent: Number of spaces for indentation in the JSON output

        Returns:
            A JSON string representation of the metric
        """
        metric_dict = self.to_dict()

        # Convert non-serializable objects to strings
        for key, value in metric_dict.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                metric_dict[key] = str(value)

        return json.dumps(metric_dict, indent=indent)

    @classmethod
    def from_dict(cls, metric_dict: Dict[str, Any]) -> "Metric":
        """
        Create a Metric instance from a dictionary.

        Args:
            metric_dict: Dictionary containing metric attributes

        Returns:
            A new Metric instance
        """
        return cls(**metric_dict)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"Metric({params_str})"

    def __str__(self) -> str:
        return f"Metric(experiment_id={self.experiment_id}, name={self.name}, value={self.value})"
