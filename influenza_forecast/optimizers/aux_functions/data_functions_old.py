import csv
import numpy as np

# Functions for handling data

# !!!!!!!!!!!!!!!!!!!!!!!
# DEPRECATED FOR BAROYANAGE!!!!
# But some functions are still in use
# !!!!!!!!!!!!!!!!!!!!!!!


def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)  # get hours and remainder
    m, s = divmod(s, 60)  # split remainder into minutes and seconds
    return '%4i:%02i:%02i' % (h, m, s)


def readFromCsvToList(filename):
    # return list with all data from csv file, skipping the first row (headers)
    reader = csv.reader(open(filename), delimiter=';')
    next(reader)
    res_list = list(reader)
    return res_list


def generateNumSequenceFromZero(final):
    # generates the sequence of the form 0, +-1, +-2, ..., +-final
    num_list = [0]

    for i in range(1, final):
        num_list.extend([i, -i])

    return np.array(num_list)


def doesContainEpidPeak(inc_data):
    prev_inc = 0
    for cur_inc in inc_data:
        if cur_inc < prev_inc:
            return True  # We have a peak in data (considering there's no minor peaks before the big one)
        prev_inc = cur_inc

    return False  # lacking data on incidence peaks


def change_data_detail(df_simul, age_groups):
    bce_acum = 0
    for idx, age_group in enumerate(age_groups):
        bce_acum += df_simul[age_group]
        df_simul.drop(age_group, axis=1, inplace=True)
        df_simul.drop(age_group + "_rel", axis=1, inplace=True)
    df_simul["Все"] = bce_acum

    # Population
    total_population = 0
    for age_group in age_groups:
        total_population += df_simul["Население " + age_group]
        df_simul.drop("Население " + age_group, axis=1, inplace=True)
    df_simul["Население"] = total_population

    # Rel incidence
    df_simul["Все_rel"] = df_simul["Все"] * 1000 / df_simul["Население"]

    return df_simul


def find_grouped_delta(df_data, df_simul, incidence_type, detail, strains, age_groups, predict, tpeak_bias_aux, strains_main_num, EPID_THRESHOLD):
    peak_indices_real, _ = max_elem_indices(df_data, incidence_type,
                                                detail and True,  # Is real data?
                                                strains,
                                                age_groups)
    peak_indices_model, peak_values_model = max_elem_indices(df_simul, incidence_type,
                                                                  detail and False,  # Is real data?
                                                                  strains,
                                                                  age_groups)

    # delta_list_prelim = [peak_index_model - peak_index_real for peak_index_model, peak_index_real in
    #                      zip(peak_indices_model, peak_indices_real)]
    if predict and tpeak_bias_aux != 0:
        delta_list_prelim = [peak_index_model - tpeak_bias_aux for peak_index_model, peak_index_real in
                             zip(peak_indices_model, peak_indices_real)]
    else:
        delta_list_prelim = [peak_index_model - peak_index_real for peak_index_model, peak_index_real in
                             zip(peak_indices_model, peak_indices_real)]

    # Finding the ultimate timeline shift for the model according to the dominant strain (from data)
    ### Legacy code ###
    # if peak_values_model[strains_main_num] > EPID_THRESHOLD:  # There is a peak in the simulation data
    #     delta = delta_list_prelim[strains_main_num]
    # else:  # Aligning by the biggest peak of the remained ones
    delta = delta_list_prelim[max_elem_index(peak_values_model)]
    ### Legacy code ###

    return delta


def remove_background_incidence(y):
    # Considering that in the lowest incidence day the disease incidence equals background
    y_min = min(y)
    return [y[i]-y_min for i in range(0,len(y))], y_min


def max_elem_index(my_list):
    # returns the index of a highest incidence
    max_value = max(my_list)
    max_index = my_list.index(max_value)
    return max_index


def max_elem_indices(df, incidence_type, detail, keys_list_1, key_list_2):
    # returns the index of a highest incidence
    max_values_list = []
    max_indices_list = []

    if incidence_type == "strain":
        for key_1 in keys_list_1:
            my_list = list(df[key_1])
            max_value = max(my_list)
            max_values_list.append(max_value)
            max_indices_list.append(my_list.index(max_value))

    elif incidence_type == "age-group":
        if detail:
            my_list = list(df["Все"])
            max_value = max(my_list)
            max_values_list.append(max_value)
            max_indices_list.append(my_list.index(max_value))
        else:
            for key_2 in key_list_2:
                my_list = list(df[key_2])
                max_value = max(my_list)
                max_values_list.append(max_value)
                max_indices_list.append(my_list.index(max_value))

    elif incidence_type == "strain_age-group":
        for key_1 in keys_list_1:
            for key_2 in key_list_2:
                my_list = list(df[key_1 + '_' + key_2])
                max_value = max(my_list)
                max_values_list.append(max_value)
                max_indices_list.append(my_list.index(max_value))

    elif incidence_type == "total":
        my_list = list(df["Все"])
        max_value = max(my_list)
        max_values_list.append(max_value)
        max_indices_list.append(my_list.index(max_value))

    return max_indices_list, max_values_list


def max_peak_index(df, keys_list_1, key_list_2, incidence_type, detail=False):
    # returns the index of a highest incidence
    max_values_list = []
    if incidence_type == "strain":
        for key_1 in keys_list_1:
            my_list = list(df[key_1])
            max_values_list.append(max(my_list))

    elif incidence_type == "age-group":
        if detail:
            my_list = list(df["Все"])
            max_values_list.append(max(my_list))
        else:
            for key_2 in key_list_2:
                my_list = list(df[key_2])
                # print(my_list)
                max_values_list.append(max(my_list))

    elif incidence_type == "strain_age-group":
        for key_1 in keys_list_1:
            for key_2 in key_list_2:
                my_list = list(df[key_1 + "_" + key_2])
                # print(my_list)
                max_values_list.append(max(my_list))

    elif incidence_type == "total":
        my_list = list(df["Все"])
        max_values_list.append(max(my_list))

    peak_index = max_values_list.index(max(max_values_list))  # Finding the highest incidence among all the strains

    return peak_index


def calculate_dist_squared(x, y, delta):
    # calculating the fitting coefficient r
    # x is real data, y is modeled curve
    # delta is the difference between the epidemic starts in real data and modeled curve
    sum = 0
    for i in range(delta, delta + len(x)):
        # if x[i-delta]>0 and y[i]>0: #do not consider absent data which is marked by -1
        sum = sum + pow(x[i - delta] - y[i], 2)

    return sum


def calculate_dist_squared_list(df_data, df_simul, strains, delta):
    # x is real data, y is modeled curve
    # delta is the difference between the epidemic starts in real data and modeled curve

    sum_list = []
    for strain in strains:
        x = list(df_data[strain])
        y = list(df_simul[strain])

        sum = 0
        for i in range(delta, delta + len(x)):
            sum = sum + pow(x[i - delta] - y[i], 2)

        sum_list.append(sum)

    return sum_list


def calculate_dist_squared_weighted(x, y, delta, w):
    # calculating the fitting coefficient r
    # x is real data, y is modeled curve
    # delta is the difference between the epidemic starts in real data and modeled curve
    # w are the weights marking the importance of particular data points fitting

    sum = 0
    for i in range(delta, delta + len(x)):
        # if x[i-delta]>0 and y[i]>0: #do not consider absent data which is marked by -1
        sum = sum + w[i - delta] * pow(x[i - delta] - y[i], 2)

    return sum


def calculate_dist_squared_weighted_list(df_data, df_data_detail, df_simul, strains, age_groups, delta, w, incidence_type, detail):
    # x is real data, y is modeled curve
    # delta is the difference between the epidemic starts in real data and modeled curve

    sum_list = []
    sum_ww_list = []
    if incidence_type == "strain":
        for strain in strains:
            x = list(df_data[strain])
            y = list(df_simul[strain])

            sum = 0
            sum_ww = 0
            for i in range(delta, delta + len(x)):
                sum = sum + w[strain][i - delta] * pow(x[i - delta] - y[i], 2)
                sum_ww = sum_ww + pow(x[i - delta] - y[i], 2)

            sum_list.append(sum)
            sum_ww_list.append(sum_ww)

    elif incidence_type == "age-group":
        if detail:
            x = list(df_data_detail["Все"])
            y = list(df_simul["Все"])

            sum = 0
            sum_ww = 0
            for i in range(delta, delta + len(x)):
                # sum = sum + pow(x[i - delta] - y[i], 2)
                sum = sum + w["Все"][i - delta] * pow(x[i - delta] - y[i], 2)
                sum_ww = sum_ww + pow(x[i - delta] - y[i], 2)

            sum_list.append(sum)
            sum_ww_list.append(sum_ww)

        else:
            for age_group in age_groups:
                x = list(df_data[age_group])
                y = list(df_simul[age_group])

                sum = 0
                sum_ww = 0
                for i in range(delta, delta + len(x)):
                    sum = sum + w[age_group][i - delta] * pow(x[i - delta] - y[i], 2)
                    sum_ww = sum_ww + pow(x[i - delta] - y[i], 2)

                sum_list.append(sum)
                sum_ww_list.append(sum_ww)

    elif incidence_type == "strain_age-group":
        for strain in strains:
            for age_group in age_groups:
                x = list(df_data[strain + "_" + age_group])
                y = list(df_simul[strain + "_" + age_group])

                sum = 0
                sum_ww = 0
                for i in range(delta, delta + len(x)):
                    sum = sum + w[strain + "_" + age_group][i - delta] * pow(x[i - delta] - y[i], 2)
                    sum_ww = sum_ww + pow(x[i - delta] - y[i], 2)

                sum_list.append(sum)
                sum_ww_list.append(sum_ww)

    elif incidence_type == "total":
        x = list(df_data["Все"])
        y = list(df_simul["Все"])

        sum = 0
        sum_ww = 0
        for i in range(delta, delta + len(x)):
            sum = sum + w["Все"][i - delta] * pow(x[i - delta] - y[i], 2)
            sum_ww = sum_ww + pow(x[i - delta] - y[i], 2)

        sum_list.append(sum)
        sum_ww_list.append(sum_ww)

    return sum_list, sum_ww_list


def calculate_r_square(df_data_weekly, df_data_detail_weekly, df_simul_weekly, strains, age_groups,
                       delta, weights, detail, incidence_type):
    res2_list = find_residuals_weighted_list(
        df_data_detail_weekly if detail else df_data_weekly,
        strains,
        age_groups,
        weights,
        "total" if detail else incidence_type)

    dist2_list, dist2_ww_list = calculate_dist_squared_weighted_list(df_data_weekly,
                                                                     df_data_detail_weekly,
                                                                     df_simul_weekly,
                                                                     strains,
                                                                     age_groups,
                                                                     delta,
                                                                     weights,
                                                                     incidence_type,
                                                                     detail)

    R_square_list = [1 - fun_val / res2 for fun_val, res2 in zip(dist2_list, res2_list)]

    return R_square_list


def calculate_peak_bias(x, y):
    x_peak = max(x)
    # print("Real peak height:", x_peak)
    y_peak = max(y)
    # print("Model peak height:", y_peak)
    # return abs(x_peak-y_peak)
    return y_peak / x_peak


def find_residuals(data):
    res = 0
    mean = np.mean(data)
    for i in range(0, len(data)):
        res += pow(data[i] - mean, 2)
    return res


def find_residuals_list(df, strains):
    res_list = []
    for strain in strains:
        res = 0
        data = list(df[strain])
        mean = np.mean(data)
        for i in range(0, len(data)):
            res += pow(data[i] - mean, 2)
        res_list.append(res)
    return res_list


def find_residuals_weighted(data, w):
    res = 0
    mean = np.mean(data)
    for i in range(0, len(data)):
        res += w[i] * pow(data[i] - mean, 2)
    return res


def find_residuals_weighted_list(df, strains, age_groups, w, incidence_type):
    res_list = []
    if incidence_type == "strain":
        for strain in strains:
            res = 0
            data = list(df[strain])
            mean = np.mean(data)
            for i in range(0, len(data)):
                res += w[strain][i] * pow(data[i] - mean, 2)
            res_list.append(res)

    elif incidence_type == "age-group":
        for age_group in age_groups:
            res = 0
            data = list(df[age_group])
            mean = np.mean(data)
            for i in range(0, len(data)):
                res += w[age_group][i] * pow(data[i] - mean, 2)
            res_list.append(res)

    elif incidence_type == "strain_age-group":
        for strain in strains:
            for age_group in age_groups:
                res = 0
                data = list(df[strain + '_' + age_group])
                mean = np.mean(data)
                for i in range(0, len(data)):
                    res += w[strain + '_' + age_group][i] * pow(data[i] - mean, 2)
                res_list.append(res)

    elif incidence_type == "total":
        res = 0
        data = list(df["Все"])
        mean = np.mean(data)
        for i in range(0, len(data)):
            res += w["Все"][i] * pow(data[i] - mean, 2)
        res_list.append(res)

    return res_list



def nonZeroMoments(y_model, EPS, ground_level = 0.0):
    # Finds the time moment to start model data plotting
    i=1 #omitting the I0
    while y_model[i]<ground_level + EPS and i<len(y_model)-1:
        i+=1
    j = i
    while y_model[j] >= ground_level + EPS and j < len(y_model) - 1:
        j += 1
    return i, j


def CutZeroData(y_model, ground_level = 0.0):
    i, j = nonZeroMoments(y_model, ground_level)
    return y_model[i:j]