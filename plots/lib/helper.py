import numpy as np


pop_ssp_dict = {
    "ssp119": "ssp1",
    "ssp126": "ssp1",
    "ssp245": "ssp2",
    "ssp370": "ssp3",
    "ssp434": "ssp2",
    "ssp460": "ssp2",
    "ssp585": "ssp1",
}


def pct_change(a, b):
    """Returns the percent change from a to b"""
    return np.round((float(b) - a) / a * 100, 1)
