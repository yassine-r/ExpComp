import json
from typing import Any, Dict, Optional
import json
from typing import Dict, Any, Optional

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



# Example usage
if __name__ == "__main__":
    # Create an experiment config with custom parameters
    config = ExperimentConfig(
        id = "56788",
        model_name="ResNet50",
        learning_rate=0.001,
        batch_size=32,
        optimizer="Adam",
        dataset="CIFAR-10",
        epochs=100,
        use_augmentation=True
    )
    
    # Access parameters using attribute syntax
    print(f"Model: {config.model_name}")
    print(f"Learning rate: {config.learning_rate}")
    
    # Or using get() with a default value
    print(f"Weight decay: {config.get('weight_decay', 0.0)}")
    
    # Update parameters
    config.update(learning_rate=0.0005, weight_decay=0.001)
    
    # Export to JSON
    config_json = config.to_json()
    print(f"\nExperiment config JSON:\n{config_json}")
    
    # Create a new config from a dictionary
    new_config = ExperimentConfig.from_dict({
        "model_name": "VGG16",
        "learning_rate": 0.01,
        "batch_size": 64,
        "id": "1234"
    })
    
    print(f"\nNew experiment config:\n{new_config.__repr__()}")
    
    print(new_config)