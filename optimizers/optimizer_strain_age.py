from .base_optimizer import BaseOptimizer

from .aux_functions import data_functions as dtf
from .aux_functions import param_range_functions as param_rangef


class StrainAgeModelOptimizer(BaseOptimizer):
    def __init__(self, model, data_obj, model_detail, sigma):
        super().__init__(model, data_obj, model_detail, sigma)
        self.age_groups = model.age_groups
        self.age_group_ind = 0
        self.groups = [a + "_" + b for a in self.strains for b in self.age_groups] if model_detail else ['Все']

    def fit_function(self, k):
        """
        Calculates sum of distances between the original data points and simulated ones
        """
        age_groups_num = len(self.age_groups)
        strains_num = len(self.strains)
        n = strains_num * age_groups_num

        exposed_list = []

        for i in range(age_groups_num):
            sum_exposed = sum(k[i * strains_num:i * strains_num + strains_num])

            if sum_exposed < 1:
                temp = [k[i * strains_num + m] for m in range(strains_num)]
                temp.append(1 - sum_exposed)
            else:
                temp = [k[i * strains_num + m] / sum_exposed for m in range(strains_num)]
                temp.append(0)
            exposed_list.append(temp)

        lam_list = list(k[n:n + strains_num])
        a = list(k[-age_groups_num:])

        dist2_list = self.find_model_fit(exposed_list, lam_list, a)
        dist2 = sum(dist2_list)

        return dist2

    '''
    def update_delta(self):
        age_group = self.age_groups[self.age_group_ind]
        groups = [group for group in self.groups if age_group in group]
        peak_indices_real, _ = dtf.max_elem_indices(self.df_data_weekly, groups)
        peak_indices_model, peak_values_model = dtf.max_elem_indices(self.df_simul_weekly, groups)

        peak_indices_model = [(peak_index + self.tpeak_bias_aux) for peak_index in peak_indices_model]
        delta_list_prelim = [peak_index_model - peak_index_real
                             for peak_index_model, peak_index_real in
                             zip(peak_indices_model, peak_indices_real)]
        self.delta = delta_list_prelim[dtf.max_elem_index(peak_values_model)]

    def fit_one_outbreak(self, predict=False, sample_size=None, bootstrap_mode=False):
        """
        An outbreak fitting function
        """
        self.calib_data_weekly = self.df_data_weekly[:sample_size]
        self.bootstrap_mode = bootstrap_mode

        self.data_weights = self._get_data_weights(self.df_data_weekly, self.sigma)
        self.res2_list = dtf.find_residuals_weighted_list(self.df_data_weekly, self.groups, self.data_weights)
        opt_params = []
        opt_peak = 0
        min_distance = 10e11

        if predict:
            for i in range(len(self.age_groups)):
                self.age_group_ind = i
            opt_result_cur, opt_params_cur = self.run_prediction()
            opt_result_fun = opt_result_cur[1].fun
            if opt_result_fun < min_distance:
                min_distance = opt_result_fun
                opt_params = opt_params_cur
                opt_peak = self.tpeak_bias_aux
        else:
            opt_result, opt_params = self.run_calibration()
            min_distance = opt_result.fun

        print("Values of optimized fit function: ", min_distance)
        print("Optimal predicted peak index: ", opt_peak)

        exposed_opt_list, lambda_opt_list, a_opt = param_rangef.get_opt_params(self.incidence_type,
                                                                               self.a_detail,
                                                                               opt_params)
        epid_params = {'exposed': exposed_opt_list,
                       'lambda': lambda_opt_list,
                       'a': a_opt,
                       'delta': self.delta,
                       'total_recovered': self.model.total_recovered,
                       'R2': self.R_square_list}

        print("Final optimal parameters: ")
        print("exposed: ", exposed_opt_list)
        print("lambda: ", lambda_opt_list)
        print("a: ", a_opt)
        print("R2: ", self.R_square_list)
        print("Recovered: ", self.model.total_recovered)

        return epid_params'''

    def calculate_population_immunity(self, exposed_list, a):
        """
        Calculates population immunity
        """
        for i in range(0, len(exposed_list)):
            self.population_immunity[i] = [
                (imm + (exposed_list[i] * self.active_population * a[i % 4])) / self.active_population
                for imm in self.population_immunity[i]]

        return self.population_immunity
