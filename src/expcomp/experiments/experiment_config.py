import json
from typing import Any, Dict, Optional

class ExperimentConfig:
    """
    A class to manage machine learning experiment configurations.
   
    This class allows users to define custom experiment parameters while
    maintaining a flexible structure for ML experiment tracking.
    """
   
    def __init__(self, id: Any, **kwargs):
        self.id = id
        for key, value in kwargs.items():
            setattr(self, key, value)
   
    def get(self, param_name: str, default: Optional[Any] = None) -> Any:
        """
        Get a parameter value with an optional default.
       
        Args:
            param_name: The parameter name to retrieve
            default: Value to return if the parameter doesn't exist
           
        Returns:
            The parameter value or the default
        """
        return getattr(self, param_name, default)
   
    def update(self, **kwargs) -> None:
        """
        Update multiple parameters at once.
       
        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
   
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the experiment configuration to a dictionary.
       
        Returns:
            A dictionary containing all experiment parameters
        """
        return self.__dict__.copy()
   
    def to_json(self, indent: int = 2) -> str:
        """
        Convert the experiment configuration to a JSON string.
       
        Args:
            indent: Number of spaces for indentation in the JSON output
           
        Returns:
            A JSON string representation of the experiment
        """
        config_dict = self.to_dict()
       
        for key, value in config_dict.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                config_dict[key] = str(value)
       
        return json.dumps(config_dict, indent=indent)
   
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """
        Create an ExperimentConfig instance from a dictionary.
       
        Args:
            config_dict: Dictionary containing experiment parameters
           
        Returns:
            A new ExperimentConfig instance
        """
        return cls(**config_dict)
   
    def __repr__(self) -> str:
        params_str = ', '.join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"ExperimentConfig({params_str})"
   
    def __str__(self) -> str:
        id = self.id
        return f"ExperimentConfig_{id}"

