from .Models.TotalBRModel import TotalBRModel
from .Models.StrainBRModel import StrainBRModel
from .Models.StrainAgeGroupBRModel import StrainAgeGroupBRModel
from .Models.AgeGroupBRModel import AgeGroupBRModel

class FactoryBRModel:
    def __new__(self, incidence: str):
        if incidence == 'total':
            return TotalBRModel

        if incidence == 'strain':
            return StrainBRModel

        if incidence == 'age-group':
            return AgeGroupBRModel

        if incidence == 'strain_age-group':
            return StrainAgeGroupBRModel