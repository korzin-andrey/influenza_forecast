import json
import os
import os.path as osp
import pandas as pd
import io
from io import StringIO
import jsonpickle
import matplotlib.pyplot as plt
import sys
import inspect
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb
from pathlib import Path
from sklearn.metrics import r2_score
from pandas import DataFrame
import numpy as np
from numpy import ndarray
from matplotlib import rc
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from matplotlib.colors import BASE_COLORS, TABLEAU_COLORS
from typing import List

from gui import aux_functions
# import aux_functions
from bootstrapping import predict_gates
from models import BR_model_new

from utils.experiment_setup import ExperimentalSetup
from visualization.visualization import plot_fitting, plot_bootstrap_curves_w_ci
from models import BR_model_new


def make_plot_for_bulletin(_, incidence, exposed_values,
                         lambda_values, a, mu, delta, sample_size, city, year,
                         forecast_term, inflation_parameter, plot_error_structures):
    return