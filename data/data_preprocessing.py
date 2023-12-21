import pickle
import pandas as pd
from typing import List

from .input_data_functions import prevalenceType, extractARIForSeason, extractPopSize


class EpiData:
    """
    A class designed for loading and initial processing of morbidity data
    """
    def __init__(self, file_path, incidence, age_groups, strains):
        self.incidence = incidence
        self.age_groups = age_groups
        self.strains = strains
        self.df_raw = self._get_epid_data(file_path)

    def _get_epid_data(self, file_path) -> pd.DataFrame:
        """
        Initial data preprocessing
        returns (pandas.Dataframe) : aggregated dataframe without missing values
        """
        data = pd.read_csv(file_path, index_col=0)
        if self.incidence != 'strain_age-group':
            agg_data = prevalenceType(self.incidence, data, self.age_groups, self.strains)
            return agg_data.dropna()
        return data.dropna()

    def incidence_for_season(self, first_year: int) -> pd.DataFrame:
        """
        Extracts ARI period in weeks for a given season
        param: first_year (int): ARI start year
        returns (pandas.Dataframe) : extracted ARI period
        """
        df = extractARIForSeason(self.df_raw, first_year)
        return df

    def pop_size(self, year: int) -> int:
        """
        Extracts population size for a given year
        param: year (int): provided year
        returns (int) : population size
        """
        return extractPopSize(self.df_raw, year, self.incidence, self.age_groups)


def prepare_calibration_data(path: str, incidence: str, age_groups: List,
                             strains: List, year: int):
    epidemic_data = EpiData(path, incidence, age_groups, strains)
    weekly_data = epidemic_data.incidence_for_season(year)
    population_size = epidemic_data.pop_size(year)

    return weekly_data, population_size


def get_contact_matrix(contact_matrix_path, incidence):
    cm = pickle.load(open(contact_matrix_path, "rb")) \
        if incidence not in ['strain', 'total'] else [[6.528]]
    return cm


