import os
import numpy as np
import pandas as pd
from multiprocessing import Pool

from .error_structures import generate_synth_data_neg_bin_trunc


def calibration_bs_func(args):
    optimizer, data = args
    print('process ', os.getpid(), 'started')
    optimizer.df_data_weekly = data
    opt_parameters = optimizer.fit_one_outbreak(bootstrap_mode=True)
    bootstrapping_curve = optimizer.df_simul_weekly.copy()

    return opt_parameters, bootstrapping_curve


def perform_bootstrapping(iter_num, optimizer):
    original_data = optimizer.df_data_weekly
    simulated_curve = optimizer.df_simul_weekly

    synth_data = generate_synth_data_neg_bin_trunc(original_data, simulated_curve, iter_num)

    with Pool(processes=max(os.cpu_count() - 2, 1)) as pool:
        results = pool.map_async(calibration_bs_func, synth_data)
        pool.close()
        pool.join()

    parameters_df = pd.DataFrame([result[0] for result in results.get()])
    bootstrapping_curves = [result[1] for result in results.get()]

    return parameters_df, bootstrapping_curves
