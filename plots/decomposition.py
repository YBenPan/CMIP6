import os
from glob import glob
import numpy as np
import pandas as pd

####################################################################################################
#### CREATE DECOMPOSITION GRAPHS OF MORTALITY
####################################################################################################

#### parent_dir = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/Health"
parent_dir = "D:/CMIP6_data/Outputs"
# ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585"]
ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]

pop_ssp_dict = {
    "ssp119": "ssp1",
    "ssp126": "ssp1",
    "ssp245": "ssp2",
    "ssp370": "ssp3",
    "ssp434": "ssp2",
    "ssp460": "ssp2",
    "ssp585": "ssp1"
}


def get_models(ssp):
    files_2015 = sorted(glob(
        f"{parent_dir}/Baseline_Ben_2015_National/5_years/{ssp}/*/CountryMortalityAbsolute/Allcause_mean/*_2015_GEMM.csv"))
    #### models = sorted(set([file.split("/")[-1].split("_")[2] for file in files_2015]))
    models = sorted(set([file.split("\\")[-1].split("_")[2] for file in files_2015]))
    # Add or remove models here
    models = [model for model in models if "EC-Earth3-AerChem" not in model]
    # models = [model for model in models if model in ["CESM2-WACCM6", "GFDL-ESM4", "GISS-E2-1-G", "MIROC-ES2L", "MIROC6", "MRI-ESM2-0", "NorESM2-LM"]]
    return models


def mort(pop, baseline, year, ssp, disease="Allcause", country=-1):
    models = get_models(ssp)
    disease_values = []
    pop_ssp = pop_ssp_dict[ssp]

    for i, model in enumerate(models):
        search_str = f"{parent_dir}/Baseline_Ben_{baseline}_National/5_years/{ssp}/Pop_{pop_ssp}_{pop}/CountryMortalityAbsolute/{disease}_mean/all_ages_{model}_*_{year}_GEMM.csv"
        files = sorted(glob(search_str))
        model_values = []

        for file in files:
            wk = pd.read_csv(file, usecols=np.arange(1, 49, 3))
            model_values.append(wk.iloc[country].values[-1])
        if len(model_values) == 0:
            print(f"No values found in {model}", year, ssp, pop, baseline, disease)
        model_mean = np.mean(model_values)
        disease_values.append(model_mean)
    if len(disease_values) == 0:
        print("No models found", year, ssp, pop, baseline, disease)
    disease_mean = np.mean(disease_values)
    return disease_mean


def main():
    A = "2010"
    B = "2015"
    # C = "2015"
    a = "var"
    b = "2040"
    # c = "2040"

    print("Method 1: (2010, 2015) -> (2010, 2040) -> (var, 2040)")
    print("Method 2: (2010, 2015) -> (var, 2015) -> (var, 2040)")
    print("Method 3: Using decomposition formula")

    for ssp in ssps:
        delta = mort("var", "2040", "2040", ssp) - mort("2010", "2015", "2040", ssp)

        # JF Method 1: (2010, 2015) -> (2010, 2040) -> (var, 2040)
        baseline_contribution = (mort("2010", "2040", "2040", ssp) - mort("2010", "2015", "2040", ssp))
        pop_contribution = (mort("var", "2040", "2040", ssp) - mort("2010", "2040", "2040", ssp))
        print(f"{ssp}: Population Contribution 1: {np.round(pop_contribution / delta * 100)}, Baseline Contribution 1: {np.round(baseline_contribution / delta * 100)}")

        # JF Method 2: (2010, 2015) -> (var, 2015) -> (var, 2040)
        baseline_contribution = (mort("var", "2015", "2040", ssp) - mort("2010", "2015", "2040", ssp))
        pop_contribution = (mort("var", "2040", "2040", ssp) - mort("var", "2015", "2040", ssp))
        print(
            f"{ssp}: Population Contribution 2: {np.round(pop_contribution / delta * 100)}, Baseline Contribution 2: {np.round(baseline_contribution / delta * 100)}")

        # My Method
        alpha_effect = (mort(a, b, "2040", ssp) - mort(A, b, "2040", ssp) + mort(a, B, "2040", ssp) - mort(A, B, "2040", ssp)) / 2
        beta_effect = (mort(a, b, "2040", ssp) - mort(a, B, "2040", ssp) + mort(A, b, "2040", ssp) - mort(A, B, "2040", ssp)) / 2
        print(f"{ssp}: Alpha Effect: {np.round(alpha_effect / delta * 100)}, Beta Effect: {np.round(beta_effect / delta * 100)}")

        # print("Sum verification:")
        # print(f"Method 1: {np.round(pop_contribution + baseline_contribution)}; Method 2: {np.round(alpha_effect + beta_effect)}")

        # Obsolete Method
        # alpha_effect = \
        #     (mort(a, b, c, ssp) + mort(a, B, C, ssp)) / 3 + \
        #     (mort(a, b, C, ssp) + mort(a, B, c, ssp)) / 6 - \
        #     (mort(A, b, c, ssp) + mort(A, B, C, ssp)) / 3 - \
        #     (mort(A, b, C, ssp) + mort(A, B, c, ssp)) / 6
        # beta_effect = \
        #     (mort(a, b, c, ssp) + mort(A, b, C, ssp)) / 3 + \
        #     (mort(a, b, C, ssp) + mort(A, b, c, ssp)) / 6 - \
        #     (mort(a, B, c, ssp) + mort(A, B, C, ssp)) / 3 - \
        #     (mort(a, B, C, ssp) + mort(A, B, c, ssp)) / 6
        # gamma_effect = \
        #     (mort(a, b, c, ssp) + mort(A, B, c, ssp)) / 3 + \
        #     (mort(a, B, c, ssp) + mort(A, b, c, ssp)) / 6 - \
        #     (mort(a, b, C, ssp) + mort(A, B, C, ssp)) / 3 - \
        #     (mort(a, B, C, ssp) + mort(A, b, C, ssp)) / 6



if __name__ == "__main__":
    main()
