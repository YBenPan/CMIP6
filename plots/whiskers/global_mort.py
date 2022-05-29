import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

####################################################################################################
#### CREATE WHISKER PLOT OF GLOBAL MORTALITY IN 2015, 2030, AND 2040
####################################################################################################

#### parent_dir = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/Health/Baseline_Ben_2015_National/5_years"
parent_dir = "D:/CMIP6_data/Outputs/Baseline_Ben_2015_National/5_years"
#### output_dir = "/home/ybenp/CMIP6_Images/Mortality/whiskers/global_mort"
output_dir = "D:/CMIP6_Images/Mortality/whiskers/global_mort"
ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]
years = [2015, 2030, 2040]
disease = "Allcause"


def whisker(pop_baseline, var_name="mean"):
    input()


def main():
    whisker(pop_baseline=[("2010", "2015"), ("var", "2015"), ("var", "2040")])


if __name__ == "__main__":
    main()