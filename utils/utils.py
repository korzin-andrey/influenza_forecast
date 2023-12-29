import os
import json
import yaml
import os.path as osp

from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from typing import Dict, Any, List, Tuple

import matplotlib.colors as mcolors

from .experiment_setup import ExperimentalSetup


OUTPUT_DIR = r'/BaroyanAgeMultistrain_v2/output/data'
SIMULATED_DATA_FILE = 'model_fit.csv'
INCIDENCE_DATA_FILE = 'original_data.csv'
CALIBRATION_DATA_FILE = 'calibration_data.csv'

PARAMETERS_FILE = 'parameters.json'

COLORS = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.BASE_COLORS.keys())


def get_config(config_path: str) -> Dict:
    """
    Reads config file with yaml extension in the main directory
    """
    with open(config_path, "r", encoding='utf8') as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.FullLoader)


def save_results(parameters: Dict,
                 simulated_data: DataFrame,
                 calibration_data: DataFrame,
                 original_data: DataFrame,
                 full_path: str) -> None:
    """
    Preserves all data derived during the calibration process
    """
    os.makedirs(full_path, exist_ok=True)

    json_object = json.dumps(parameters, indent=4)
    with open(osp.join(full_path, PARAMETERS_FILE), 'w') as outfile:
        outfile.write(json_object)

    simulated_data.to_csv(osp.join(full_path, SIMULATED_DATA_FILE))
    original_data.to_csv(osp.join(full_path, INCIDENCE_DATA_FILE))
    calibration_data.to_csv(osp.join(full_path, CALIBRATION_DATA_FILE))


def save_preset(incidence: str,
                city_eng: str,
                year: int,
                mu: float,
                opt_parameters: Dict,
                full_path) -> None:
    """
    Preserves the parameters for GUI manual calibration
    """
    preset_dict = {
        "city": {"Saint Petersburg":"spb", "Moscow":"msk", "Novosibirsk": "novosib"}.get(city_eng, city_eng),
        "year": year,
        "incidence": incidence,
        "exposed": opt_parameters["exposed"],
        "lambda": opt_parameters["lambda"],
        "a": opt_parameters["a"][0],
        "mu": mu,
        "delta": opt_parameters["delta"]
    }

    json_object = json.dumps(preset_dict)
    with open(osp.join(full_path, f"{preset_dict['city']}{year}_{incidence}_PRESET.json"), 'w') as outfile:
        outfile.write(json_object)


def save_epid_result(result: ndarray, columns: List[str], indicator_name: str, output_dir: str) -> None:
    """
    Changes shape of the input array to transform into pandas DataFrame and save in csv file
    """
    shape = result.shape
    array = result.reshape(shape[0] * shape[1], shape[2])
    result_df = pd.DataFrame(array.T, columns=columns)

    os.makedirs(output_dir, exist_ok=True)
    result_df.to_csv(osp.join(output_dir, indicator_name+'.csv'))


def get_parameters(output_dir: str):
    """
    Unpacks all stored data
    """
    with open(osp.join(output_dir, PARAMETERS_FILE), 'r') as f:
        params = json.load(f)

    exposed_list = params['exposed']
    lam_list = params['lambda']
    a_list = params['a']
    delta = params['delta']
    r_squared = params['R2']
    return exposed_list, lam_list, a_list, delta, r_squared


def get_exposed_ready_for_simulation(exposed_list: List[Any], incidence: str,
                                     age_groups: List[str], strains: List[str]) -> List:
    """
    Extends list of exposed population
    """
    exposed_list_cor = []
    if incidence in ['strain_age-group', 'strain']:

        age_groups_num = len(age_groups) if incidence == 'strain_age-group' else 1
        strains_num = len(strains)

        for i in range(age_groups_num):
            sum_exposed = sum(exposed_list[i * strains_num:i * strains_num + strains_num])

            if sum_exposed < 1:
                temp = [exposed_list[i * strains_num + m] for m in range(strains_num)]
                temp.append(1 - sum_exposed)
            else:
                temp = [exposed_list[i * strains_num + m] / sum_exposed for m in range(strains_num)]
                temp.append(0)
            exposed_list_cor.append(temp)
    elif incidence in ['age-group', 'total']:
        for item in exposed_list:
            exposed_list_cor.append([item, 1-item])

    return exposed_list_cor


def restore_from_saved_data(incidence: str) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Returns data  from the given files
    """
    simul_data_path = f'{OUTPUT_DIR}/{incidence}/{SIMULATED_DATA_FILE}'
    simul_data = pd.read_csv(simul_data_path, index_col=0)

    orig_data_path = f'{OUTPUT_DIR}/{incidence}/{INCIDENCE_DATA_FILE}'
    orig_data = pd.read_csv(orig_data_path, index_col=0)

    calib_data_path = f'{OUTPUT_DIR}/{incidence}/{CALIBRATION_DATA_FILE}'
    calib_data = pd.read_csv(calib_data_path, index_col=0)

    return calib_data, simul_data, orig_data


def restore_fit_from_params(contact_matrix: object, pop_size: float,
                            incidence: str, age_groups: List[str],
                            strains: List[str], mu: float,
                            sigma: float, output_dir: str):

    """
    Restores model fit from the given parameters
    """
    factory = ExperimentalSetup(incidence, age_groups, strains, contact_matrix, pop_size, mu, sigma)
    model, _ = factory.get_model_and_optimizer()
    model_obj = factory.setup_model(model)

    exposed_list, lam_list, a_list, delta, r_squared = get_parameters(output_dir)
    exposed_list_cor = get_exposed_ready_for_simulation(exposed_list, incidence, age_groups, strains)

    calib_data_path = f'{output_dir}/{CALIBRATION_DATA_FILE}'
    calib_data = pd.read_csv(calib_data_path, index_col=0)

    orig_data_path = f'{output_dir}/{INCIDENCE_DATA_FILE}'
    orig_data = pd.read_csv(orig_data_path, index_col=0)

    model_obj.set_attributes()
    model_obj.init_simul_params(exposed_list=exposed_list_cor, lam_list=lam_list, a=a_list)
    simul_data, immune_pop, susceptible, _, _ = model_obj.make_simulation()

    shape = simul_data.shape
    simul_data = simul_data.reshape(shape[0] * shape[1], shape[2])
    simul_data = pd.DataFrame(simul_data.T, columns=calib_data.columns)

    days_num = simul_data.shape[0]
    wks_num = int(days_num / 7.0)
    simul_weekly = simul_data.aggregate(func=lambda x: [x[i * 7: i * 7 + 7].sum() for i in range(wks_num)])

    return calib_data, simul_weekly, orig_data, r_squared

