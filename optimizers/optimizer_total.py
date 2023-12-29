from .base_optimizer import BaseOptimizer


class TotalModelOptimizer(BaseOptimizer):
    def __init__(self, model, data_obj, model_detail, sigma):
        super().__init__(model, data_obj, model_detail, sigma)
        self.groups = ['Все']
        self.age_groups = ['total']

    def fit_function(self, k):
        """
        Calculates sum of distances between the original data points and simulated ones
        """
        age_groups_num = len(self.age_groups)
        exposed_list = []

        for item in k[:age_groups_num]:
            exposed_list.append([item, 1 - item])

        lam_list = [k[age_groups_num]]

        if self.a_detail:
            a = [k[age_groups_num + 1 + i] for i in range(age_groups_num)]
        else:
            a = [k[age_groups_num + 1]]

        dist2_list = self.find_model_fit(exposed_list, lam_list, a)
        dist2 = sum(dist2_list)

        return dist2

    def calculate_population_immunity(self, exposed_list, a):
        """
        Calculates population immunity
        """
        for i in range(0, len(exposed_list)):
            self.population_immunity[i] = [
                (imm + (exposed_list[i] * self.active_population * a[0])) / self.active_population
                for imm in self.population_immunity[i]]

        return self.population_immunity


