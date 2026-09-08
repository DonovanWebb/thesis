"""
Helper functions for figure scripts.
"""

import numpy as np
from scipy.optimize import curve_fit
from oitg.results import load_result
import pandas as pd
from statsmodels.stats.proportion import proportion_confint


def load_data(rid, day):
    f = load_result(rid=rid, day=day, experiment="fastgates")
    return f
