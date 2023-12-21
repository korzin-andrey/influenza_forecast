from copy import deepcopy
import numpy as np
from pathos.multiprocessing import ProcessingPool as P_Pool
import os


def _calibration_fit_func(arg_tuple):
    optimizer, kwargs = arg_tuple
    opt_parameters = optimizer.fit_one_outbreak(**kwargs)
    return optimizer, opt_parameters


class MultipleModelFit:
    optimizers = []  # multiple model optimizers
    criterions = []  # criterions to choose best-fit (compared lexicographically)

    class MetricCriterions:
        @staticmethod
        def SSR_R2(optimizer, opt_parameters):
            ssr = 0
            r_squared = optimizer.R_square_list
            for r2 in r_squared:
                ssr += abs(1 - r2) ** 2
            return ssr

        @staticmethod
        def count_correct_r2_diaps(optimizer, opt_parameters):
            amount = 0
            r_squared = optimizer.R_square_list
            for r2 in r_squared:
                if 0 <= r2 <= 1:
                    amount += 1  # fully-correct
                elif -1 <= r2 <= 1:
                    amount += 0.8  # semi-correct
                elif -2 <= r2 <= 2:
                    amount += 0.5  # partially-correct
            return amount

        @staticmethod
        def std_of_r2(optimizer, opt_parameters):
            return np.std(optimizer.R_square_list)

        @staticmethod
        def descending(function):
            def _negative(*args, **kwargs):
                return -1 * function(*args, **kwargs)

            return _negative

    def __init__(self):
        """
        metrics are listed in descending order of priority
        """

        # 1.
        self.criterions.append(self.MetricCriterions.descending
                               (self.MetricCriterions.count_correct_r2_diaps))  # descending metric
        # 2.
        self.criterions.append(self.MetricCriterions.std_of_r2)  # ascending metric
        # 3.
        self.criterions.append(self.MetricCriterions.SSR_R2)  # ascending metric

    @staticmethod
    def from_optimizer(optimizer, num_iter_fit):
        obj = MultipleModelFit()
        obj.optimizers = [deepcopy(optimizer) for _ in range(num_iter_fit)]
        return obj

    def fit_n_outbreaks(self, **kwargs):

        with P_Pool(min(len(self.optimizers), os.cpu_count()-1)) as pool:
            results = pool.amap(_calibration_fit_func,
                                [(optimizer, kwargs)
                                 for optimizer in self.optimizers])
            pool.close()
            pool.join()

        return [model_fit for model_fit in results.get()]

    def check_critetions(self, optimizer, opt_parameters):
        metrics = []
        for criterion in self.criterions:
            metrics.append(criterion(optimizer, opt_parameters))
        return tuple(metrics)

    def get_models_best_fit(self, **kwargs):
        results = self.fit_n_outbreaks(**kwargs)
        results.sort(key=lambda optimum: self.check_critetions(optimum[0], optimum[1]))
        # descending order by optimality
        return results
