import pandas as pd
from typing import List

from calibration.experiment_setup import ExperimentalSetup


def read_parameters(params_path):
    return pd.read_csv(params_path, index_col=0)


def restore_bootstrapped_curves_from_params(contact_matrix: object, pop_size: float, incidence: str,
                                            age_groups: List[str], strains: List[str]):

    factory = CaseFactory()
    model, _ = factory.get_experiment_setup(incidence)
    model = model(contact_matrix, pop_size, incidence, age_groups, strains)

    calib_data_path = f'{OUTPUT_DIR}/{incidence}/{CALIBRATION_DATA_FILE}'
    calib_data = pd.read_csv(calib_data_path, index_col=0)

    orig_data_path = f'{OUTPUT_DIR}/{incidence}/{INCIDENCE_DATA_FILE}'
    orig_data = pd.read_csv(orig_data_path, index_col=0)

    simul_data_path = f'{OUTPUT_DIR}/{incidence}/{SIMULATED_DATA_FILE}'
    simulated_curve = pd.read_csv(simul_data_path, index_col=0)

    bootstrapped_curves = []
    params = read_parameters('output/bootstrapped_epid_params_4i_7.csv').drop('delta', axis=1)

    for i in range(len(params)):
        exposed_list, lam_list, a_list, _, r_squared = [[float(param.strip('[]'))] for param in params.iloc[i]]
        exposed_list_cor = get_exposed_ready_for_simulation(exposed_list, incidence, age_groups, strains)
        model.init_simul_params(exposed_list=exposed_list_cor, lam_list=lam_list, a=a_list)
        simul_data, immune_pop, susceptible = model.make_simulation()

        days_num = simul_data.shape[1]
        wks_num = int(days_num / 7.0)
        simul_weekly = [sum([simul_data.T[j] for j in range(i * 7, (i + 1) * 7)])
                        for i in range(wks_num)]
        bootstrapped_curves.append(pd.DataFrame(simul_weekly, columns=calib_data.columns))

    return calib_data, orig_data, simulated_curve, bootstrapped_curves
