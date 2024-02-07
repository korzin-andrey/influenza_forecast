import numpy as np
import pandas as pd

import math
from data.data_preprocessing import get_contact_matrix
from data.data_preprocessing import prepare_calibration_data
from models.BR_model_new import BRModel
from utils.utils import get_config


def get_data_and_model(mu, incidence, exposure_year):
    config = get_config('config.yaml')

    path = config['data_path']
    contact_matrix_path = config['contact_matrix_path']

    age_groups = config['age_groups']
    strains = config['strains']

    groups = []
    if incidence == 'age-group':
        groups = age_groups

    elif incidence == 'strain':
        groups = strains

    elif incidence == 'strain_age-group':
        groups = [a + "_" + b for a in strains for b in age_groups]
    elif incidence == 'total':
        groups = ['Все']

    contact_matrix = get_contact_matrix(contact_matrix_path, incidence)
    epid_data, pop_size = prepare_calibration_data(path, incidence,
                                                   age_groups, strains, exposure_year)

    model = BRModel(contact_matrix, pop_size, mu,
                    incidence, age_groups, strains)

    return epid_data, model, groups


def prepare_exposed_list(incidence, exposed_list):
    if incidence == 'age-group':
        exposed_cor = []
        for item in exposed_list:
            exposed_cor.append([item, 1 - item])
        exposed_list = exposed_cor

    elif incidence == 'strain':
        sum_exposed = sum(exposed_list)
        if sum_exposed < 1:
            exposed_list.append(1 - sum_exposed)
        else:
            exposed_list = [item / sum_exposed for item in exposed_list]
            exposed_list.append(0)

    elif incidence == 'strain_age-group':
        strains_num = 3
        exposed_cor = []
        for i in range(2):
            sum_exposed = sum(
                exposed_list[i * strains_num:i * strains_num + strains_num])

            if sum_exposed < 1:
                temp = [exposed_list[i * strains_num + m]
                        for m in range(strains_num)]
                temp.append(1 - sum_exposed)
            else:
                temp = [exposed_list[i * strains_num + m] /
                        sum_exposed for m in range(strains_num)]
                temp.append(0)
            exposed_cor.append(temp)
        exposed_list = exposed_cor

    elif incidence == 'total':
        exposed_list = [[exposed_list[0], 1 - exposed_list[0]]]

    return exposed_list


def transform_days_to_weeks(simul_data, groups):
    shape = simul_data.shape
    simul_data = simul_data.reshape(shape[0] * shape[1], shape[2])
    simul_data = pd.DataFrame(simul_data.T, columns=groups)

    days_num = simul_data.shape[0]
    wks_num = int(days_num / 7.0)
    simul_weekly = simul_data.aggregate(
        func=lambda x: [x[i * 7: i * 7 + 7].sum() for i in range(wks_num)])
    return simul_weekly


def generate_xticks(epid_data, year, last_simul_ind):
    m, n = epid_data.index[0], epid_data.index[-1]
    xticks_vals = pd.DataFrame(0, index=np.arange(
        last_simul_ind), columns=['week', 'year'])
    xticks_vals.loc[epid_data.index.values,
                    :] = epid_data.loc[:, ['Неделя', 'Год']].to_numpy()
    first_week_num = epid_data.loc[m, 'Неделя']
    last_week_num = epid_data.loc[n, 'Неделя']

    for i in range(n + 1, last_simul_ind):
        xticks_vals.loc[i, ['week']] = last_week_num + 1
        xticks_vals.loc[i, ['year']] = year + 1
        if last_week_num != 52:
            last_week_num += 1
        else:
            last_week_num = 0

    for i in range(m-1, -1, -1):
        xticks_vals.loc[i, ['week']] = first_week_num - 1
        xticks_vals.loc[i, ['year']] = year - 1
        if first_week_num != 0:
            first_week_num -= 1
        else:
            first_week_num = 52

    xticks_text = [val if (val % 5 == 0) or (
        val == 1) else '' for val in xticks_vals['week'].tolist()]
    return xticks_vals, xticks_text


def exposed_dict_to_inputs(exposed_dict, incidence):
    if type(exposed_dict) not in (dict, list, tuple):
        return [exposed_dict]

    if incidence == 'total':
        return list(exposed_dict)
    elif incidence == 'strain':
        if type(exposed_dict) in (list, tuple):
            return type(exposed_dict)
        return [exposed_dict['A(H1N1)'], exposed_dict['A(H3N2)'], exposed_dict['B']]
    elif incidence == 'age-group':
        if type(exposed_dict) in (list, tuple):
            return type(exposed_dict)
        return [exposed_dict['0-14'], exposed_dict['15+']]
    elif incidence == 'strain_age-group':
        if type(exposed_dict) in (list, tuple):
            return type(exposed_dict)
        return [exposed_dict['0-14']['A(H1N1)'], exposed_dict['0-14']['A(H3N2)'], exposed_dict['0-14']['B'],
                exposed_dict['15+']['A(H1N1)'], exposed_dict['15+']['A(H3N2)'], exposed_dict['15+']['B']]
    else:
        raise ValueError(f"can't parse incidence: {incidence}")


def lambda_dict_to_inputs(lambda_dict, incidence):
    if type(lambda_dict) not in (dict, list, tuple):
        return [lambda_dict]

    if incidence == 'total':
        return list(lambda_dict)
    elif incidence == 'strain':
        if type(lambda_dict) in (list, tuple):
            return type(lambda_dict)
        return [lambda_dict['A(H1N1)'], lambda_dict['A(H3N2)'], lambda_dict['B']]
    elif incidence == 'age-group':
        return list(lambda_dict)
    elif incidence == 'strain_age-group':
        if type(lambda_dict) in (list, tuple):
            return type(lambda_dict)
        return [lambda_dict['A(H1N1)'], lambda_dict['A(H3N2)'], lambda_dict['B']]
    else:
        raise ValueError(f"can't parse incidence: {incidence}")


cities = {'rus': 'Российская Федерация',
          'spb': 'Санкт-Петербург',
          'msc': 'Москва',
          'novosib': 'Новосибирск'}
