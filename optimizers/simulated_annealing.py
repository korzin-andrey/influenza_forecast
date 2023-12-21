import numpy as np
import sys, time
from simanneal import Annealer

from .aux_functions import data_functions_old as datf


class InitValueFinder(Annealer):
    """Test annealer
    """
    ranges = []
    history_states = []
    age_groups = []
    incidence_type = ""
    a_detail = False
    energy_func = None

    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, ranges, history_states, age_groups, incidence_type, a_detail,
                 optimum_func):
        self.ranges = ranges
        self.history_states = history_states
        self.age_groups = age_groups
        self.incidence_type = incidence_type
        self.energy_func = optimum_func
        self.a_detail = a_detail
        self.steps = 1000  # 10000
        self.updates = 100  # 1000
        self.copy_strategy = "slice"
        super(InitValueFinder, self).__init__(state)  # important!

    def move(self):
        """Moves to different argument"""

        state_list = []
        for i in range(0, len(self.state)):
            state_list.append(np.random.uniform(self.ranges[i][0], self.ranges[i][1]))
        state_array = np.array(state_list)

        self.state = state_array

    def energy(self):
        """Calculates the optimization function value"""

        exposed_list, lam_list, a_list = [], [], []
        age_groups_num = len(self.age_groups)
        strains_num = len(self.history_states) - 1

        if self.incidence_type in ['age-group', 'total']:
            for item in self.state[:age_groups_num]:
                exposed_list.append([item, 1 - item])

            lam_idx = age_groups_num
            lam_list = [self.state[age_groups_num]]

            if self.a_detail:
                # todo: [legacy] hasn't been changed and tested
                a_list = [0] * len(self.age_groups)
                a_idx = lam_idx + 1
                for idx_j, age_group in enumerate(self.age_groups):
                    a_list[idx_j] = self.state[a_idx + idx_j]
            else:
                a_idx = lam_idx + 1
                a_list = [self.state[a_idx]]

        elif self.incidence_type == 'strain_age-group':

            for i in range(age_groups_num):
                sum_exposed = sum(self.state[i * strains_num:i * strains_num + strains_num])

                if sum_exposed < 1:
                    temp = [self.state[i * strains_num + m] for m in range(strains_num)]
                    temp.append(1 - sum_exposed)
                else:
                    temp = [self.state[i * strains_num + m] / sum_exposed for m in range(strains_num)]
                    temp.append(0)
                exposed_list.append(temp)

            n = age_groups_num * strains_num
            lam_list = list(self.state[n:n+strains_num])
            a_list = list(self.state[-age_groups_num:])

        elif self.incidence_type == 'strain':
            self.state = list(self.state)
            sum_exposed = sum(self.state[:strains_num])
            if sum_exposed < 1:
                exposed_list = self.state[:strains_num]
                exposed_list.append(1-sum_exposed)
            else:
                exposed_list = [item / sum_exposed for item in self.state[:strains_num]]
                exposed_list.append(0)

            lam_list = self.state[strains_num:2*strains_num]
            a_list = [self.state[-1]]

        dist2_list = self.energy_func(exposed_list, lam_list, a_list)
        dist2 = sum(dist2_list)

        return dist2

    def update(self, step, T, E, acceptance, improvement):
        elapsed = time.time() - self.start
        if step == 0:
            print(' Temperature        Energy    Accept   Improve     Elapsed   Remaining',
                  file=sys.stderr)
            print('\r%12.5f  %12.2f                      %s            ' %
                  (T, E, datf.time_string(elapsed)), file=sys.stderr)  #
            sys.stderr.flush()
        else:
            remain = (self.steps - step) * (elapsed / step)
            print('\r%12.5f  %12.2f  %7.2f%%  %7.2f%%  %s  %s\r' %
                  (T, E, 100.0 * acceptance, 100.0 * improvement,
                   datf.time_string(elapsed), datf.time_string(remain)), file=sys.stderr)
            sys.stderr.flush()
