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


def init_by_factor(factor_name, factor):
    """Initialize ages and diseases array based on the variable in question"""
    if factor_name == "Age":
        if factor == "25-60":
            ages = [
                "age_25_29_Mean",
                "age_30_34_Mean",
                "age_35_39_Mean",
                "age_40_44_Mean",
                "age_45_49_Mean",
                "age_50_54_Mean",
                "age_55_59_Mean",
            ]
        elif factor == "60-80":
            ages = [
                "age_60_64_Mean",
                "age_65_69_Mean",
                "age_70_74_Mean",
                "age_75_79_Mean",
            ]
        elif factor == "80+":
            ages = [
                "age_80_84_Mean",
                "age_85_89_Mean",
                "age_90_94_Mean",
                "post95_Mean",
            ]
        elif factor == "25+":
            ages = ["all_age_Mean"]
        else:
            raise NameError(f"{factor} ages group not found!")
        disease = None
    elif factor_name == "Disease":
        ages = None
        disease = factor
    else:
        ages = None
        disease = None
    return ages, disease
