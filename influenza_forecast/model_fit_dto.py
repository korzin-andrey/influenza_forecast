import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import List
import os.path as osp

from optimizers.base_optimizer import BaseOptimizer


@dataclass
class ModelFitDto:
    optimizer: BaseOptimizer
    opt_parameters: List

    @staticmethod
    def from_optimizer(optimizer: BaseOptimizer, opt_parameters: List):
        dto = ModelFitDto(optimizer=deepcopy(optimizer),
                          opt_parameters=deepcopy(opt_parameters))
        return dto

    def export(self, city: str, year: str, incidence: str, predict: bool, output_dir: str = "./"):
        with open(osp.join(output_dir, f'{city}_{year}_{incidence}_p_{predict}.pickle'), 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def search(city: str, year: str, incidence: str, predict: bool, output_dir: str = "./"):
        path = osp.join(output_dir, f'{city}_{year}_{incidence}_p_{predict}.pickle')
        if osp.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None
