import numpy as np
from time import time

from .utils import real_to_abs


def f(h, m, a):
    """
    Function to imitate that the individual was exposed to the same virus strain as in the previous season
    :param h: exposure history state
    :param m: virus strain m
    :param a: susceptible fraction of individuals in (0;1] range
    """
    if h == m:
        return a
    else:
        return 1


class BRModel:
    """
    Age-sctructured Baroyan-Rvachev model with comparison for the averaged model without age groups
    """
    def __init__(self, M, pop_size, mu, incidence_type, age_groups, strains):
        """

        :param M: contact matrix
        :param rho (float): susceptible population size
        :param incidence_type (str): experiment setup
        :param age_group (list[str]): list of age groups
        :param strains (list[str]): list of strain names
        """
        self.q = [0.0, 0.0, 1, 0.9, 0.55, 0.3, 0.15, 0.05]  # infectivity
        self.cont_num = 6.528
        self.N = 1800  # in days

        self.M = M
        self.rho = pop_size  # A scalar in absence of separate age groups
        self.mu = mu

        self.incidence_type = incidence_type
        self.a_detail = False  # flag whether to set individual a_value per age group (a - susceptible fraction)
        self.age_groups = age_groups
        self.strains = strains

        self.history_states = []
        self.strains_num = len(strains)
        self.history_states_num = -1  # To be defined according incidence_type

        self.exposed_fraction_h = []  # fractions with different exposure history
        self.lam_m = []
        self.I0 = []
        self.a = []  # Waning immunity level

        self.total_recovered = []  # Recovered from different strains

    def init_simul_params(self, exposed_list, lam_list, a):
        if not isinstance(exposed_list, list):
            exposed_list = [exposed_list]
        if not isinstance(lam_list, list):
            lam_list = [lam_list]
        if not isinstance(a, list):
            a = [a]

        self.exposed_fraction_h = exposed_list.copy()
        self.lam_m = lam_list.copy()
        self.a = a.copy()

    def sum_ill(self, y, t):
        """
        Summing the cumulative infectivity of the infected by the strain m at the moment t
        """
        sum = 0
        T = len(self.q)  # disease duration for a single person

        for epid_day in range(0, T):
            if t - epid_day < 0:
                y_cur = 0
            else:
                y_cur = y[t - epid_day]

            sum = sum + y_cur * self.q[epid_day]

        return sum

    def get_recovery_time(self):
        return len(self.q) + 1  # After active infection period

    @staticmethod
    def calc_relative_prevalence(y_model, rho):
        """
        Converts the absolute prevalence into relative (per 10000 persons)
        :param y_model:  2D array
        :param rho:
        :return:
        """
        y_rel = []
        for i in range(0, len(y_model)):
            y_rel.append(real_to_abs(y_model[i], rho))

        return y_rel

    def make_simulation(self):
        raise NotImplementedError


class StrainModel(BRModel):
    def __init__(self, M, pop_size, mu, incidence_type, age_groups, strains):
        super().__init__(M, pop_size, mu, incidence_type, age_groups, strains)
        self.history_states = self.strains.copy() + ["No exposure"]
        self.history_states_num = len(self.history_states)
        # I0 - initially infected persons
        self.I0 = np.ones(self.strains_num)
        # add number of initially infected persons to susceptible population size
        self.rho += np.sum(self.I0)

    def make_simulation(self):
        y = np.zeros((self.strains_num, self.N + 1))  # 3 x 1801 | strains x N
        x = np.zeros((self.history_states_num, self.N + 1))  # 4 x 1801 | history_strains x age_group x N
        r0 = [0] * self.strains_num
        population_immunity = np.zeros((self.strains_num, self.N + 1))  # 3 x 1801 | strains x N
        self.total_recovered = [0, 0, 0]  # list reset to zero

        for h in range(0, self.history_states_num):  # initial data
            x[h][0] = self.exposed_fraction_h[h] * self.rho

        for m in range(0, self.strains_num):  # initial data
            y[m][0] = self.I0[m]
            population_immunity[m][0] = 0
            for i in range(1, self.get_recovery_time() - 1):
                population_immunity[m][i] = population_immunity[m][0]

        for t in range(0, self.N):  # Each t
            for m in range(0, self.strains_num):  # calculating y_m
                y[m][t + 1] = 0

            for h in range(0, self.history_states_num):
                x[h][t + 1] = x[h][t]

                infect_force_total = 0  # Infection from all strains per h group
                infect_force = []
                for m in range(0, self.strains_num):  # Calculating y_m
                    if t == 0:
                        r0[m] = (self.lam_m[m] * self.cont_num / (1/6)) # Contact rate / rate of removal
                    infect_force.append(self.lam_m[m] * self.cont_num *
                                        self.sum_ill(y[m], t) * f(h, m, self.a[0]) / self.rho)
                    infect_force_total += infect_force[m]  # Considering the overall strength of the infection

                real_infected = min(infect_force_total, 1.0) * x[h][t]
                x[h][t + 1] -= real_infected

                if infect_force_total > 0:
                    for m in range(0, self.strains_num):  # Calculating y_m
                        real_infected_m = real_infected * (infect_force[m] / infect_force_total)  # Причитающаяся доля
                        y[m][t + 1] += real_infected_m
                        self.total_recovered[
                            m] += real_infected_m  # They will get cured (no mortality)
                        if t > self.get_recovery_time() - 1:
                            population_immunity[m][t + 1] = population_immunity[m][t] + real_infected_m
        return y, population_immunity, self.rho, r0


class AgeModel(BRModel):
    def __init__(self, M, pop_size, mu, incidence_type, age_groups, strains):
        super().__init__(M, pop_size, mu, incidence_type, age_groups, strains)
        self.history_states = self.age_groups.copy() + ["No exposure"]
        self.history_states_num = len(self.history_states)
        self.I0 = np.ones(len(self.age_groups))
        self.rho += np.sum(self.I0)
        self.total_recovered =[0 for i in range(len(self.age_groups))]
        self.a_detail = False  # [legacy] change manually if needed

    def make_simulation(self):
        age_groups_num = len(self.age_groups)

        y = np.zeros((age_groups_num, self.N + 1))  # 4 x 1801 | strains x N
        y[:, 0] = self.I0
        x = np.zeros((self.history_states_num, self.N + 1))  # 5 x 1801 | history_states x N
        x[:, 0] = [(exp_fraction * self.rho).item() for exp_fraction in self.exposed_fraction_h]

        population_immunity = np.zeros((age_groups_num, self.N + 1))  # 4 x 1801 | strains x N
        recovery_days_num = self.get_recovery_time()

        for t in range(0, self.N):
            for h in range(0, self.history_states_num):
                x[h][t + 1] = x[h][t]

                infect_force_total = 0  # Infection from all strains per group
                infect_force_list = []

                for m, age_group in enumerate(self.age_groups):  # Calculating y_m
                    if self.a_detail:
                        a = self.a[m]
                    else:
                        a = self.a[0]

                    inf_force = self.lam_m[0] * self.cont_num * self.sum_ill(y[m], t) * f(h, m, a) / self.rho
                    infect_force_list.append(inf_force)
                    infect_force_total += inf_force  # Considering the overall strength of the infection

                real_infected = min(infect_force_total, 1.0) * x[h][t]
                x[h][t + 1] -= real_infected

                if infect_force_total > 0:
                    for m, age_group in enumerate(self.age_groups):  # Calculating y_m
                        real_infected_m = real_infected * (infect_force_list[m] / infect_force_total)

                        y[m][t + 1] += real_infected_m
                        self.total_recovered[m] += real_infected_m  # They will get cured (no mortality)

                        if t > recovery_days_num - 1:
                            if isinstance(real_infected_m, np.ndarray):
                                population_immunity[m][t + 1] = population_immunity[m][t] + real_infected_m[0]
                            else:
                                population_immunity[m][t + 1] = population_immunity[m][t] + real_infected_m

        return y, population_immunity, self.rho, []


class StrainAgeModel(BRModel):
    def __init__(self, M, pop_size, mu, incidence_type, age_groups, strains):
        super().__init__(M, pop_size, mu, incidence_type, age_groups, strains)
        self.history_states = self.strains.copy() + ["No exposure"]
        self.history_states_num = len(self.history_states)
        self.I0 = np.ones(self.strains_num * len(self.age_groups))
        self.rho += np.sum(self.I0)

    def make_simulation(self):
        y = np.zeros((self.strains_num * len(self.age_groups), self.N + 1))  # 12 x 1801 | strains x age_group x N
        x = np.zeros((self.history_states_num * len(self.age_groups), self.N + 1))  # 16 x 1801 | history_strains x age_group x N
        population_immunity = np.zeros((self.strains_num * len(self.age_groups), self.N + 1))  # 12 x 1801 | strains x N
        self.total_recovered = [[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]]  # Обнуляется список | list reseted to zero

        for h in range(0, self.history_states_num):  # initial data
            for age_idx, age_group in enumerate(self.age_groups):
                x[(h * len(self.age_groups)) + age_idx][0] = self.exposed_fraction_h[(h * len(self.age_groups)) + age_idx] * self.rho

        for m in range(0, self.strains_num):  # initial data
            for age_idx in range(len(self.age_groups)):
                y[(m * len(self.age_groups)) + age_idx][0] = self.I0[(m * len(self.age_groups)) + age_idx]
                population_immunity[(m * len(self.age_groups)) + age_idx][0] = 0
                for i in range(1, self.get_recovery_time() - 1):
                    population_immunity[(m * len(self.age_groups)) + age_idx][i] = population_immunity[m][0]

        for t in range(0, self.N):  # Each t
            for m in range(0, self.strains_num):  # calculating y_m
                for age_idx in range(len(self.age_groups)):
                    y[(m * len(self.age_groups)) + age_idx][t + 1] = 0

            for h in range(0, self.history_states_num):
                for age_idx_i, age_group in enumerate(self.age_groups):
                    x[(h * len(self.age_groups)) + age_idx_i][t + 1] = x[(h * len(self.age_groups)) + age_idx_i][t]

                    infect_force_total = 0  # Инфекция от всех штаммов на группу h | Infection from all strains per group
                    infect_force = [[] for strain in self.strains]
                    for m in range(0, self.strains_num):  # Calculating y_m
                        for age_idx_j in range(0, len(self.age_groups)):
                            infect_force[m].append(self.lam_m[m] * self.M[age_idx_i][age_idx_j]
                                                   * self.sum_ill(y[(m * len(self.age_groups)) + age_idx_i], t)
                                                   * f(h, m, self.a[age_idx_i]) / self.rho)
                            # Считаем общую силу инфекции | Considering the overall strength of the infection
                            infect_force_total += infect_force[m][age_idx_j]

                    real_infected = min(infect_force_total, 1.0) * x[(h * len(self.age_groups)) + age_idx_i][t]
                    x[(h * len(self.age_groups)) + age_idx_i][t + 1] -= real_infected

                    if infect_force_total > 0:
                        for m in range(0, self.strains_num):  # Calculating y_m
                            real_infected_m = real_infected * (infect_force[m][age_idx_i] / infect_force_total)  #  Причитающаяся доля
                            y[(m * len(self.age_groups)) + age_idx_i][t + 1] += real_infected_m
                            self.total_recovered[age_idx_i][m] += real_infected_m  #  Они переболеют (нет смертности) | They will get cured (no mortality)
                            if t > self.get_recovery_time() - 1:
                                if type(real_infected_m) == "numpy.ndarray":
                                    population_immunity[(m * len(self.age_groups)) + age_idx_i][t + 1] = \
                                    population_immunity[(m * len(self.age_groups)) + age_idx_i][t] + \
                                    real_infected_m[0]
                                else:
                                    population_immunity[(m * len(self.age_groups)) + age_idx_i][t + 1] = \
                                    population_immunity[(m * len(self.age_groups)) + age_idx_i][t] + real_infected_m
        return y, population_immunity, self.rho, []


class TotalModel(BRModel):
    def __init__(self, M, pop_size, mu, incidence_type, age_groups, strains):
        super().__init__(M, pop_size, mu, incidence_type, age_groups, strains)
        self.history_states = ["Exposure", "No exposure"]
        self.history_states_num = len(self.history_states)
        self.I0 = np.ones(1)
        self.rho -= np.sum(self.I0)

    def make_simulation(self):
        y = np.zeros((1, self.N + 1))  # 1 x 1801 | strains x N
        x = np.zeros((self.history_states_num, self.N + 1))  # 2 x 1801 | history_strains x age_group x N
        population_immunity = np.zeros((1, self.N + 1))  # 1 x 1800 | strains x N
        self.total_recovered = [0]

        for h in range(0, self.history_states_num):  # initial data
            x[h][0] = self.exposed_fraction_h[h] * (1-self.mu) * self.rho

        for m in range(0, 1):  # initial data
            y[m][0] = self.I0[m]
            population_immunity[m][0] = 0
            for i in range(1, self.get_recovery_time() - 1):
                population_immunity[m][i] = population_immunity[m][0]

        for t in range(0, self.N):  # Each t
            for m in range(0, 1):  # calculating y_m
                y[m][t + 1] = 0

            for h in range(0, self.history_states_num):
                x[h][t + 1] = x[h][t]

                infect_force_total = 0  # Инфекция от всех штаммов на группу h | Infection from all strains per group
                infect_force = []
                for m in range(0, 1):  # Calculating y_m
                    infect_force.append(
                        self.lam_m[m] * self.cont_num * self.sum_ill(y[m], t) * f(h, m, self.a[0]) / self.rho)
                    infect_force_total += infect_force[m]  # Considering the overall strength of the infection

                real_infected = min(infect_force_total, 1.0) * x[h][t]
                x[h][t + 1] -= real_infected

                if infect_force_total > 0:
                    for m in range(0, 1):  # Calculating y_m
                        real_infected_m = real_infected * (
                                    infect_force[m] / infect_force_total)  # Причитающаяся доля
                        y[m][t + 1] += real_infected_m
                        self.total_recovered[
                            m] += real_infected_m  # Они переболеют (нет смертности) | They will get cured (no mortality)
                        if t > self.get_recovery_time() - 1:
                            population_immunity[m][t + 1] = population_immunity[m][t] + real_infected_m

        return y, population_immunity, self.rho, []






