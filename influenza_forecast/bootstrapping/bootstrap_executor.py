import os
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Callable
from pathos.multiprocessing import ProcessingPool as P_Pool


class BootstrapExecutor:
    def __init__(self, optimizer, calibration_function: Callable, synth_data_generator: Callable):
        self.original_data = deepcopy(optimizer.df_data_weekly)
        self.simulated_curve = deepcopy(optimizer.df_simul_weekly)
        self.synth_data_generator = synth_data_generator
        self.calibration_func = calibration_function
        self.synth_data = []
        self.optimizer = optimizer

    def generate_synth_data(self, iter_num, **params):
        self.synth_data = self.synth_data_generator(self.original_data, self.simulated_curve,
                                                    datasets_amount=iter_num, **params)
        return self.synth_data

    def perform_bootstrapping(self):
        if not self.synth_data:
            raise ValueError('synth data was not generated: use "generate_synth_data(self, iter_num, **params)" method')

        with P_Pool(processes=max(os.cpu_count() - 2, 1)) as pool:
            def calib(synth_data):
                return self.calibration_func((deepcopy(self.optimizer), synth_data))

            pool.clear()
            pool.restart(force=True)  # fixing the error "ValueError: Pool not running"
            results = pool.amap(calib, self.synth_data)
            pool.close()
            pool.join()

        results = list(results.get())
        parameters_list = [result[0] for result in results]
        bootstrapping_curves = [result[1] for result in results]
        optimizers = [result[2] for result in results]

        return parameters_list, pd.DataFrame(parameters_list), bootstrapping_curves, optimizers


def calibration_bs_func(args):
    optimizer, data = args
    optimizer.df_data_weekly = data
    opt_parameters = optimizer.fit_one_outbreak(bootstrap_mode=True)
    bootstrap_curve = optimizer.df_simul_weekly.copy()

    return opt_parameters, bootstrap_curve, deepcopy(optimizer)
