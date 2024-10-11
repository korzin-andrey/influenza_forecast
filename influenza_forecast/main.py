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


def main():
    config = get_config('config.yaml')

    path = config['data_path']
    exposure_year = config['year']
    contact_matrix_path = config['contact_matrix_path']
    mu = config['percent_protected']
    sigma = config['sigma']

    age_groups = config['age_groups']
    strains = config['strains']

    output_folder = config['output_folder']
    city_eng = config['city']
    city = "_".join(city_eng.lower().split())

    incidence = config['INCIDENCE_TYPE']
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
    experiment_setter = ExperimentalSetup(incidence, age_groups, strains, contact_matrix, pop_size, mu, sigma)

    num_iter_fit = 1  # os.cpu_count() - 1  # amount of re-calibrations to obtain best-fit model

    optimizer = experiment_setter.setup_experiment(epidemic_data, model_detail)

    multiple_model_fit = MultipleModelFit.from_optimizer(optimizer, num_iter_fit)

    sample_size = 8
    fitting_params = {"predict": predict, "sample_size": sample_size} if predict else {}

    optimizer, opt_parameters = multiple_model_fit.get_models_best_fit(**fitting_params)[0]

    output_dir = osp.join(output_folder, 'data', incidence)
    model_fit = optimizer.df_simul_weekly.dropna(axis=1)
    incidence_data = optimizer.df_data_weekly.loc[:, model_fit.columns]
    calibration_data = optimizer.calib_data_weekly.loc[:, model_fit.columns]
    r_squared = optimizer.R_square_list

    population_immunity = optimizer.population_immunity
    r0 = optimizer.r0
    rt = optimizer.rt
    groups = optimizer.groups

    results_dir = f'{incidence}_({exposure_year})_{datetime.now().strftime("%Y_%m_%d_%H_%M")}_mu_{mu}_sigma_{sigma}'
    full_path = osp.normpath(osp.join(output_dir, results_dir))
    os.makedirs(full_path)

    file_path_fitting = osp.join(full_path, f'fit_{incidence}_{city}_{exposure_year}_P{predict}.png')
    plot_fitting(incidence_data, calibration_data, model_fit, city_eng,
                 exposure_year, file_path_fitting, r_squared=r_squared, predict=predict)

    file_path_pop_i = osp.join(full_path, f'pop_imm_{incidence}_{city}_{exposure_year}.png')
    plot_immune_population(population_immunity, list(model_fit.columns), city_eng, exposure_year, file_path_pop_i)

    file_path_r0 = osp.join(full_path, f'r0_{incidence}_{city}_{exposure_year}.png')
    plot_r0(r0, city_eng, exposure_year, file_path_r0)

    file_path_rt = osp.join(full_path, f'rt_{incidence}_{city}_{exposure_year}.png')
    plot_rt(rt, list(model_fit.columns), city_eng, exposure_year, file_path_rt)

    save_results(opt_parameters, model_fit, calibration_data, incidence_data, full_path)
    if config["SAVE_PRESET"]:
        save_preset(incidence, city_eng, exposure_year, mu, opt_parameters, full_path)

    save_epid_result(population_immunity, groups, 'immunity', full_path)
    save_epid_result(rt, groups, 'rt', full_path)

    if predict:
        inflation_parameter = 1.0
        datasets_amount = 200
        length_of_predict = 4

        file_path_gates = osp.join(full_path, f'fit_{incidence}_{city}_{exposure_year}_P{predict}_gates.png')

        predict_gates_generator = PredictGatesGenerator(optimizer.df_data_weekly.loc[:, model_fit.columns],
                                                        optimizer.df_simul_weekly.dropna(axis=1),
                                                        datasets_amount, sample_size, inflation_parameter)

        plot_gates(city_eng, exposure_year, predict_gates_generator, file_path_gates,
                   predict_gates_generator.generate_predict_gate(25, 75, length=length_of_predict),
                   predict_gates_generator.generate_predict_gate(5, 95, length=length_of_predict),
                   simulated_datasets_max_week=predict_gates_generator.outbreak_begin+sample_size+length_of_predict-1)

    if 'cache_folder' in config and config['cache_folder']:
        ModelFitDto \
            .from_optimizer(optimizer, opt_parameters) \
            .export(city, exposure_year, incidence, predict, config['cache_folder'])


if __name__ == '__main__':
    main()
