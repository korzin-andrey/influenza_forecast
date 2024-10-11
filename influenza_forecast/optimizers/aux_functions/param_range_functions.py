from typing import List

# Msk 2017
# exposed_range = [(0.21, 0.27), (0.32, 0.37), (0.37, 0.42)]
# lam_range = [(0.07, 0.08), (0.07, 0.09), (0.07, 0.09)]
# a_range = (0.05, 0.3)


def set_parameters_range(incidence, a_detail=False):
    exposed_range = lam_range = a_range = None

    if incidence == "strain":
        exposed_range = {
            "A(H1N1)pdm09": (0.1, 0.4),
            "A(H3N2)": (0.1, 0.4),
            "B": (0.1, 0.4)
        }
        lam_range = {
            "A(H1N1)pdm09": (0.06, 0.12),
            "A(H3N2)": (0.07, 0.13),
            "B": (0.06, 0.13)
        }
        a_range = (0.05, 0.5)
    elif incidence == "age-group":
        exposed_range = {
            "0-14": (0.005, 0.05),
            "15 и ст.": (0.5, 0.7)
        }
        lam_range = (0.2, 0.5)
        a_range = (0.0, 0.5)
        """
        # 2010
        exposed_range = {
            "0-14": (0.005, 0.007), # 0.0065
            "15 и ст.": (0.7, 0.76) # 0.73
        }
        lam_range = (0.28, 0.32)  # 0.29
        a_range = (0.1, 0.12)  # 0.11
        
        # 2015 # mu = 0.18
        exposed_range = {
            "0-14": (0.0045, 0.0055), # 0.005
            "15 и ст.": (0.72, 0.8) # 0.783
        }
        lam_range = (0.28, 0.32)  # 0.29
        a_range = (0.12, 0.19)  # 0.177
        """
    elif incidence == "strain_age-group":
        exposed_range = {
            "0-14": {
                "A(H1N1)pdm09": (0.005, 0.9),
                "A(H3N2)": (0.005, 0.9),
                "B": (0.005, 0.9)
            },
            "15 и ст.": {
                "A(H1N1)pdm09": (0.005, 0.9),
                "A(H3N2)": (0.005, 0.9),
                "B": (0.005, 0.9)
            }
        }
        lam_range = {
            "A(H1N1)pdm09": (0.01, 0.3),
            "A(H3N2)": (0.01, 0.3),
            "B": (0.01, 0.3)
        }
        a_range = {
            "0-14": (0.0, 1.0),
            "15 и ст.": (0.0, 1.0)
        }
    elif incidence == "total":
        exposed_range = (0.1, 0.4)
        lam_range = (0.06, 0.12)
        a_range = (0.05, 0.5)

    params_range = []
    for item in [exposed_range, lam_range, a_range]:
        for param in unpack_parameter_values(item):
            params_range.append(param)
    return params_range


def unpack_parameter_values(parameter):
    if isinstance(parameter, tuple):
        yield parameter

    elif isinstance(parameter, dict):
        temp_list = []
        for item in list(parameter.values()):
            if isinstance(item, tuple):
                yield item
            elif isinstance(item, dict):  # nested dictionary
                for value in list(item.values()):
                    yield value
        return temp_list


def get_opt_params(K, incidence, age_groups, strains, a_detail=False):
    """
    Unpacks found parameters according to the given incidence type
    param K:

    """
    m, n = len(strains), len(age_groups)
    params = []

    if incidence == 'strain':
        params = [K[:m], K[m:2*m], [K[-1]]]
    elif incidence == 'age-group':
        params = [K[:n], [K[n]], K[n+1:] if a_detail else [K[-1]]]
    elif incidence == 'strain_age-group':
        params = [K[:m*n], K[m*n:m*n+m], K[-n:]]
    elif incidence == 'total':
        params = [[K[0]], [K[1]], [K[2]]]

    for i, item in enumerate(params):
        if not isinstance(item, List):
            params[i] = list(item)
    return params
