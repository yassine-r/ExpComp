from dataclasses import dataclass
from typing import List
from expcomp.evaluation import Metric
from .experiment_config import ExperimentConfig

@dataclass
class Experiment:
    config: ExperimentConfig
    evaluations: List[Metric]