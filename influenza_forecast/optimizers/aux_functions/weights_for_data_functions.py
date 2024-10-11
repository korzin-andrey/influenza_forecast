import pandas as pd
# import sys
# sys.path.insert(1, 'aux_functions')
from . import data_functions_old as dtf


def getWeights4Data(df, model_group, sigma):
    weights = {}

    for subgroup in model_group:
        y = list(df[subgroup])
        w_general_list = [1]
        for i in range(len(y)):
            w_general_list.append(w_general_list[-1] / sigma)

        peak_index = dtf.max_elem_index(y)

        w = []
        for i in range(len(y)):  # assigning values of w based on the distance from the peak
            w.append(w_general_list[abs(peak_index - i)])

        weights[subgroup] = w

    return weights
