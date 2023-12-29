from typing import List
from dataclasses import dataclass
from pandas import DataFrame

from models.BR_model_new import BRModel

from optimizers.optimizer_strain import StrainModelOptimizer
from optimizers.optimizer_age import AgeModelOptimizer
from optimizers.optimizer_strain_age import StrainAgeModelOptimizer
from optimizers.optimizer_total import TotalModelOptimizer


@dataclass
class ExperimentalSetup:
    incidence_type: str
    age_groups: List[str]
    strains: List[str]
    contact_matrix: object
    pop_size: float
    mu: float
    sigma: float

    def get_model_and_optimizer(self):
        model, optimizer = BRModel, ExperimentalSetup.get_optimizer_by_incidence(self.incidence_type)

        return model, optimizer

    @staticmethod
    def get_optimizer_by_incidence(incidence_type):
        optimizer = None
        if incidence_type == 'age-group':
            optimizer = AgeModelOptimizer

        elif incidence_type == 'strain':
            optimizer = StrainModelOptimizer

        elif incidence_type == 'strain_age-group':
            optimizer = StrainAgeModelOptimizer

        elif incidence_type == 'total':
            optimizer = TotalModelOptimizer
        return optimizer

    def setup_model(self, model):
        return model(self.contact_matrix, self.pop_size, self.mu,
                     self.incidence_type, self.age_groups, self.strains)

    def setup_optimizer(self, optimizer, model, data, model_detailed):
        return optimizer(model, data, model_detailed, self.sigma)

    def setup_experiment(self, data: DataFrame, model_detailed: bool):
        model, optimizer = self.get_model_and_optimizer()
        model_obj = self.setup_model(model)
        optimizer_obj = self.setup_optimizer(optimizer, model_obj, data, model_detailed)
        return optimizer_obj
