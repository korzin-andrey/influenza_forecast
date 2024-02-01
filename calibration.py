import os.path as osp
import os
from datetime import datetime

from bootstrapping.predict_gates import PredictGatesGenerator
from model_fit_dto import ModelFitDto
from utils.experiment_setup import ExperimentalSetup
from optimizers.multiple_model_fit import MultipleModelFit
from data.data_preprocessing import get_contact_matrix, prepare_calibration_data

from visualization.bootstrap_results import plot_gates
from visualization.visualization import plot_fitting, plot_immune_population, plot_r0, plot_rt
from utils.utils import get_config, save_results, save_epid_result, save_preset


def calibration(year, mu, incidence, sample_size):
    config = get_config('config.yaml')

    path = config['data_path']
    exposure_year = int(year)
    contact_matrix_path = config['contact_matrix_path']
    sigma = config['sigma']

    age_groups = config['age_groups']
    strains = config['strains']

    data_detail = config['DATA_DETAIL']
    model_detail = config['MODEL_DETAIL']
    predict = config['PREDICT']

    # Data will be grouped if model is grouped (GMDD and DMGD will be run as GMGD)
    if not data_detail or not model_detail:
        incidence = 'total'

    contact_matrix = get_contact_matrix(contact_matrix_path, incidence)
    epidemic_data, pop_size = prepare_calibration_data(path, incidence, age_groups,
                                                       strains, exposure_year)

    # Calibration
    experiment_setter = ExperimentalSetup(
        incidence, age_groups, strains, contact_matrix, pop_size, mu, sigma)

    num_iter_fit = 1  # os.cpu_count() - 1  # amount of re-calibrations to obtain best-fit model

    optimizer = experiment_setter.setup_experiment(epidemic_data, model_detail)

    multiple_model_fit = MultipleModelFit.from_optimizer(
        optimizer, num_iter_fit)

    fitting_params = {"predict": predict,
                      "sample_size": sample_size} if predict else {}

    optimizer, opt_parameters = multiple_model_fit.get_models_best_fit(
        **fitting_params)[0]

    print("Calibration finished")
    return opt_parameters


if __name__ == '__main__':
    calibration()
