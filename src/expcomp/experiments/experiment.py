from dataclasses import dataclass
from typing import List
from .experiment_config import ExperimentConfig
from expcomp.evaluation import Metric

@dataclass
class Experiment:
    config: ExperimentConfig
    evaluations: List[Metric]

    