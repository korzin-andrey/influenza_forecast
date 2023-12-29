import pickle
import numpy as np
from scipy import stats as st

from BaroyanAgeMultistrain_v2.utils import get_config
from data.data_preprocessing import EpiData
from visualization import plot_bootstrapped_fitting


def calculate_rmse(calibration_data, original_data, bs_curves):
    m = calibration_data.index[-1] + 1
    n = original_data.index[-1]

    rmse_list = [np.sqrt(np.square(bs_curve.iloc[m:n] - original_data.iloc[m:]).sum()
                         / len(bs_curve[m:n]))
                 for bs_curve in bs_curves]
    ci = st.norm.interval(confidence=0.95, loc=np.mean(rmse_list), scale=st.sem(rmse_list))
    rmse_mean = np.mean(rmse_list)
    return rmse_mean, *ci


if __name__ == '__main__':

    config = get_config('../config.yaml')

    path = config['data_path']
    city_eng = config['city']
    exposure_year = config['year']
    contact_matrix_path = config['contact_matrix_path']
    percent_protected = config['percent_protected']

    strains = config['strains']
    age_groups = config['age_group']
    output_folder = config['output_folder']
    city = "_".join(city_eng.lower().split())

    incidence = config['INCIDENCE_TYPE']

    epidemic_data = EpiData(path, city_eng, incidence, age_groups, strains)
    M = pickle.load(open(contact_matrix_path, "rb"))

    pop_size_raw = epidemic_data.pop_size(exposure_year)
    suspected_pop_size = pop_size_raw * (1 - percent_protected)

    calib_data, orig_data, simul_curve, bootstrapped_curves = \
        restore_bootstrapped_curves_from_params(M, suspected_pop_size, incidence, age_groups, strains)
    rmse, cil, cih = calculate_rmse(calib_data, orig_data, bootstrapped_curves)
    print('rmse: ', rmse, 'ci_low: ', cil[0], 'ci_high: ', cih[0])

    plot_bootstrapped_fitting(calib_data, orig_data, simul_curve, bootstrapped_curves)
