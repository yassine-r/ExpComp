# ExpComp - Machine Learning Experiment Tracking & Comparison

[![PyPI version](https://img.shields.io/pypi/v/expcomp.svg)](https://pypi.org/project/expcomp/)

A lightweight Python library for managing, tracking, and comparing machine learning experiments.

## Features

- 📁 **Automatic Experiment Organization**: Creates standardized folder structures for experiments
- ⚙️ **Config Management**: Store hyperparameters and training configurations in JSON format
- 📊 **Metrics Tracking**: Log performance metrics (accuracy, loss, F1-score, etc.) across experiments
- 🔍 **Comparison Tools**: Filter and compare experiments using pandas DataFrames
- 🎯 **Condition Filtering**: Easily find experiments that match specific criteria

## Installation
```bash
pip install expcomp
```

## Quick Start

### Creating Dummy Experiments

```python
import os
import random
import json

from expcomp import Condition
from expcomp.evaluation import Metric
from expcomp.experiments import Experiment, ExperimentComparison, ExperimentConfig, ExperimentLoader



def generate_experiment_config(exp_id):
    """Generate a dummy experiment configuration with random parameters."""
    model_architectures = ["ResNet50", "VGG16", "MobileNetV2", "EfficientNetB0", "DenseNet121"]
    optimizers = ["Adam", "SGD", "RMSprop", "Adagrad", "Adadelta"]
    datasets = ["CIFAR-10", "MNIST", "ImageNet", "Fashion-MNIST", "COCO"]
    
    # Generate random parameters
    learning_rate = round(random.uniform(0.0001, 0.1), 4)
    batch_size = random.choice([16, 32, 64, 128, 256])
    epochs = random.randint(10, 100)
    model = random.choice(model_architectures)
    optimizer = random.choice(optimizers)
    dataset = random.choice(datasets)
    
    # Create configuration
    config = ExperimentConfig(
        id=exp_id,
        model=model,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        dataset=dataset,
        random_seed=random.randint(1, 1000),
        date_created="2025-02-27",
        description=f"Experiment with {model} on {dataset}"
    )
    
    return config

def generate_metrics(exp_id):
    """Generate dummy metrics for an experiment."""
    # Create base metrics with random values
    accuracy = round(random.uniform(0.7, 0.99), 4)
    loss = round(random.uniform(0.01, 0.5), 4)
    f1 = round(random.uniform(0.65, 0.98), 4)
    precision = round(random.uniform(0.7, 0.99), 4)
    recall = round(random.uniform(0.7, 0.99), 4)
    training_time = round(random.uniform(100, 1000), 2)
    
    # Create metric objects
    metrics = [
        Metric(exp_id, "accuracy", accuracy, units="percentage", threshold=0.8),
        Metric(exp_id, "loss", loss, units="cross_entropy"),
        Metric(exp_id, "f1_score", f1),
        Metric(exp_id, "precision", precision),
        Metric(exp_id, "recall", recall),
        Metric(exp_id, "training_time", training_time, units="seconds")
    ]
    
    return metrics

def generate_dummy_experiments():
    """generate 10 experiments with configs and metrics."""
    # Create base directory
    base_dir = "experiments"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    experiments = []
    
    # Create 10 experiments
    for i in range(1, 11):
        exp_id = f"exp_{i}"
        
        # Create experiment config
        config = generate_experiment_config(exp_id)
        
        # Create experiment metrics
        metrics = generate_metrics(exp_id)
        
        # Create experiment object
        experiment = Experiment(config, metrics)
        experiments.append(experiment)
        
        # Create experiment folder
        exp_dir = os.path.join(base_dir, f"experiment_{i}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save config directly in experiment folder
        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, 'w') as f:
            f.write(config.to_json())
        
        # Save metrics directly in experiment folder
        metrics_path = os.path.join(exp_dir, "metrics.json")
        metrics_list = [metric.to_dict() for metric in metrics]
        with open(metrics_path, 'w') as f:
            json.dump(metrics_list, f, indent=2)
        
        print(f"Created experiment {i} with config and metrics")
    
    return experiments

generate_dummy_experiments()

```

### Loading & Analyzing Experiments
```python

# Load experiments from directory
all_experiments = ExperimentLoader.from_directory(
    directory_path="./experiments",
    config_file_pattern="config*.json",
    metrics_file_pattern="metrics*.json"
)

# Create comparison DataFrame
comparison = ExperimentComparison(all_experiments)

# View experiment data
print(comparison.df.head())

# Filter experiments with specific conditions
filtered_exp = comparison.filter_experiments(
    conditions=[
        Condition("config_epochs", ">=", 0.85),
        Condition("metric_loss	", "<", "0.2")
    ]
)

# show the filtered experiments
print(filtered_exp.head())
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
