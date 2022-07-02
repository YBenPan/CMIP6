import numpy as np


def pct_change(a, b):
    """Returns the percent change from a to b"""
    return np.round((float(b) - a) / a * 100, 1)
