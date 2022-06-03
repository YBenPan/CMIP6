import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

####################################################################################################
#### CREATE WHISKER PLOT OF GLOBAL MORTALITY IN 2015, 2030, AND 2040
####################################################################################################

#### parent_dir = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/Health"
parent_dir = "D:/CMIP6_data/Outputs"
#### output_dir = "/home/ybenp/CMIP6_Images/Mortality/whiskers/global_mort"
output_dir = "D:/CMIP6_Images/Mortality/whiskers/global_mort"
# ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585"]
ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]
years = [2015, 2030, 2040]
diseases = ["Allcause", "IHD"]

pop_ssp_dict = {
    "ssp119": "ssp1",
    "ssp126": "ssp1",
    "ssp245": "ssp2",
    "ssp370": "ssp3",
    "ssp434": "ssp2",
    "ssp460": "ssp2",
    "ssp585": "ssp1"
}


def get_models(ssps):
    all_models = []
    for i, ssp in enumerate(ssps):
        files_2015 = sorted(
            glob(
                f"{parent_dir}/Baseline_Ben_2015_National/5_years/{ssp}/*/CountryMortalityAbsolute/Allcause_mean/*_2015_GEMM.csv"))
        #### models = sorted(set([file.split("/")[-1].split("_")[2] for file in files_2015]))
        models = sorted(set([file.split("\\")[-1].split("_")[2] for file in files_2015]))
        # Add or remove models here
        models = [model for model in models if "EC-Earth3-AerChem" not in model]
        # models = [model for model in models if model in ["CESM2-WACCM6", "GFDL-ESM4", "GISS-E2-1-G", "MIROC-ES2L", "MIROC6", "MRI-ESM2-0", "NorESM2-LM"]]
        all_models.append(models)
    return all_models


def diseases_stacked(pop_baselines, var_name="mean"):
    data = np.zeros((len(years), len(ssps), len(pop_baselines), len(diseases)))
    df = pd.DataFrame(columns=["Year", "SSP", "Pop_Baseline", "Disease", "Mean"])
    models = get_models(ssps)
    for i, year in enumerate(years):

        for j, ssp in enumerate(ssps):
            pop_ssp = pop_ssp_dict[ssp]

            for k, pop_baseline in enumerate(pop_baselines):
                pop, baseline = pop_baseline
                all_values = []

                for p, disease in enumerate(diseases):

                    disease_values = []

                    for q, model in enumerate(models[j]):
                        search_str = f"{parent_dir}/Baseline_Ben_{baseline}_National/5_years/{ssp}/Pop_{pop_ssp}_{pop}/CountryMortalityAbsolute/{disease}_mean/all_ages_{model}_*_{year}_GEMM.csv"
                        files = sorted(glob(search_str))
                        model_values = []

                        for file in files:
                            wk = pd.read_csv(file, usecols=np.arange(1, 49, 3))
                            model_values.append(wk.iloc[-1].values[-1])

                        model_mean = np.mean(model_values)
                        disease_values.append(model_mean)
                    disease_mean = np.mean(disease_values)
                    all_values.append(disease_mean)
                    # data[i, j, k, p] = all_mean
                    df = df.append({"Year": year, "SSP": ssp, "Pop_Baseline": pop_baseline, "Disease": disease, "Mean": disease_mean},
                                   ignore_index=True)
                    # print(f"{year}, {ssp}, {pop_baseline}, {disease} mean: {all_mean}")

    df = df.groupby(["Year", "SSP", "Pop_Baseline", "Disease"]).sum().unstack("Disease")
    fig, ax = plt.subplots(figsize=(10, 10))
    df.plot.bar(ax=ax, stacked=True)
    plt.show()


def main():
    # diseases_stacked(pop_baseline=[("2010", "2015"), ("var", "2015"), ("var", "2040")])
    diseases_stacked(pop_baselines=[("var", "2015")])


if __name__ == "__main__":
    main()
