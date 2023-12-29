import os
from copy import deepcopy
from joblib import delayed, Parallel

import numpy as np
import pandas as pd
from scipy import stats
import math


class TruncatedNegativeBinomial:
    ZERO_TRUNCATED = False

    @staticmethod
    def get_zero_truncated():
        obj = TruncatedNegativeBinomial()
        obj.ZERO_TRUNCATED = True
        return obj

    def get_random_value_rp(self, r, p, max_value):
        restraint = np.arange(1 if self.ZERO_TRUNCATED else 0, math.ceil(max_value))
        probabilities = stats.nbinom.pmf(restraint, r, p)
        probabilities /= probabilities.sum()
        return np.random.choice(restraint, p=probabilities)

    def get_random_value_mv(self, mean, variance):
        p = mean / variance
        r = mean * p / (1.0 - p)

        restraint = np.arange(1 if self.ZERO_TRUNCATED else 0, math.ceil(mean + math.sqrt(variance) * 3))
        probabilities = stats.nbinom.pmf(restraint, r, p)
        probabilities /= probabilities.sum()
        return np.random.choice(restraint, p=probabilities)


class NegativeBinomialErrorStructure:
    def __init__(self, original_data: pd.DataFrame):
        self.original_data = original_data.copy(deep=True)
        self.outbreak_begin, self.outbreak_end = original_data.index.min(), original_data.index.max() + 1
        self.distribution = TruncatedNegativeBinomial.get_zero_truncated()

    def generate_synth_data(self, simulated_data: pd.DataFrame, sample_size=None, inflation_parameter=1.0):
        # sample_size, inflation_parameter is the arguments only for predict mode
        if sample_size is None:
            sample_size = self.outbreak_end - self.outbreak_begin

        simulated_incidence = simulated_data.copy(deep=True)

        residuals_variance = (simulated_incidence.iloc[self.outbreak_begin:self.outbreak_begin + sample_size, :] -
                              self.original_data[simulated_incidence.columns[:]]).std() ** 2

        for column in simulated_incidence.columns:
            for i_week in range(self.outbreak_begin, self.outbreak_end):
                current_mean = simulated_incidence.loc[i_week, column]

                if i_week - self.outbreak_begin < sample_size:
                    current_variance = residuals_variance[column]
                else:
                    current_variance = ((simulated_incidence.iloc[self.outbreak_begin:self.outbreak_begin + i_week, :] -
                                         self.original_data[simulated_incidence.columns[:]]).std() ** 2)[column]
                    current_variance *= inflation_parameter

                simulated_incidence.loc[i_week, column] = self.distribution.get_random_value_mv(current_mean,
                                                                                                current_variance)

        return simulated_incidence

    def calculate_inflation_parameter(self, sample_size):
        assert self.outbreak_begin + sample_size <= self.outbreak_end

        sample_std = self.original_data[self.outbreak_begin, self.outbreak_begin + sample_size].std()
        outbreak_std = self.original_data[self.outbreak_begin, self.outbreak_end].std()
        return sample_std / outbreak_std


def generate_synth_data_neg_bin_trunc(original_data, simulated_data, datasets_amount=200, begin=None, end=None,
                                      parallel=True):
    if begin is not None:
        if end is not None:
            NB_err = NegativeBinomialErrorStructure(original_data[begin:end])
        else:
            NB_err = NegativeBinomialErrorStructure(original_data[begin:])
    else:
        if end is not None:
            NB_err = NegativeBinomialErrorStructure(original_data[:end])
        else:
            NB_err = NegativeBinomialErrorStructure(original_data)

    def generation_function(nb_err):
        return nb_err.generate_synth_data(simulated_data)

    if not parallel:
        return [generation_function(NB_err) for _ in range(datasets_amount)]
    else:
        return Parallel(n_jobs=max(os.cpu_count() - 2, 1))(delayed(generation_function)(NB_err) for _ in range(datasets_amount))


def generate_synth_data_neg_bin_trunc_predict(original_data, simulated_data, datasets_amount=200,
                                              sample_size=8, inflation_parameter=1.0,
                                              begin=None, end=None, parallel=True):
    if begin is not None:
        if end is not None:
            NB_err = NegativeBinomialErrorStructure(original_data[begin:end])
        else:
            NB_err = NegativeBinomialErrorStructure(original_data[begin:])
    else:
        if end is not None:
            NB_err = NegativeBinomialErrorStructure(original_data[:end])
        else:
            NB_err = NegativeBinomialErrorStructure(original_data)

    def generation_function(nb_err):
        return nb_err.generate_synth_data(simulated_data, sample_size, inflation_parameter)

    if not parallel:
        return [generation_function(NB_err) for _ in range(datasets_amount)]
    else:
        return Parallel(n_jobs=max(os.cpu_count() - 2, 1))(delayed(generation_function)(NB_err) for _ in range(datasets_amount))


# Deprecated
'''
def generate_synth_data_poisson(simulated_curve: DataFrame):
    return pd.DataFrame(np.random.poisson(simulated_curve),
                        columns=simulated_curve.columns)
'''