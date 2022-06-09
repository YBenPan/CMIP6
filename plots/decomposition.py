import os
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

####################################################################################################
#### CREATE DECOMPOSITION GRAPHS OF MORTALITY
####################################################################################################

#### parent_dir = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/Health"
parent_dir = "D:/CMIP6_data/Outputs"
# ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585"]
ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]
diseases = ["Allcause", "COPD", "IHD", "LC", "LRI", "Stroke", "T2D"]
# diseases = ["LRI"]
countries = [-1, 35, 77, 183, 85, 123]
country_long_names = ["World", "China", "India", "The United States", "Japan", "Nigeria"]
output_dir = "/home/ybenp/CMIP6_Images/Mortality/decomposition"

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

    sns.set()

    for ssp in ssps:

        # Initialize plotting
        fig, axes = plt.subplots(2, 3, figsize=(15, 12))
        fig.suptitle("Decomposition of changes in mortality attributable to PM2.5 from 2015 to 2040")
        # Plotting Settings
        ymin = -100
        ymax = 500

        for i, (country, country_long_name) in enumerate(zip(countries, country_long_names)):
            pms = []
            baselines = []
            pops = []
            deltas = []

            # Select current ax:
            j = i // 3
            k = i % 3
            ax = axes[j, k]
            ax.set_ylim([ymin, ymax])

            for disease in diseases:
                ref = mort("2010", "2015", "2015", disease, ssp, country)
                delta = mort("var", "2040", "2040", disease, ssp, country) - mort("2010", "2015", "2015", disease, ssp,
                                                                                  country)

                # JF Method: (Pop 2010, Base 2015, Year 2015) ->
                #            (Pop 2010, Base 2015, Year 2040) ->
                #            (Pop 2010, Base 2040, Year 2040) ->
                #            (Pop var, Base 2040, Year 2040)
                pm_contribution = (
                            mort("2010", "2015", "2040", disease, ssp, country) - mort("2010", "2015", "2015", disease,
                                                                                       ssp, country))
                baseline_contribution = (
                            mort("2010", "2040", "2040", disease, ssp, country) - mort("2010", "2015", "2040", disease,
                                                                                       ssp, country))
                pop_contribution = (
                            mort("var", "2040", "2040", disease, ssp, country) - mort("2010", "2040", "2040", disease,
                                                                                      ssp, country))

                pm_percent = np.round(pm_contribution / ref * 100, 1)
                baseline_percent = np.round(baseline_contribution / ref * 100, 1)
                pop_percent = np.round(pop_contribution / ref * 100, 1)
                delta_percent = np.round(delta / ref * 100, 1)

                pms.append(pm_percent)
                baselines.append(baseline_percent)
                pops.append(pop_percent)
                deltas.append(delta_percent)

                print(
                    f"{ssp}, {country_long_name}, {disease}: PM Contribution: {pm_percent}%; Population Contribution: {pop_percent}%; Baseline Contribution: {baseline_percent}%")
                print(f"{ssp}, {country_long_name}, {disease}: Overall Change: {delta_percent}%")

            # Visualize with a stacked bar plot

            df = pd.DataFrame(
                {
                    "Disease": diseases,
                    "PM2.5 Concentration": pms,
                    "Baseline Mortality": baselines,
                    "Population": pops,
                    "Overall": deltas
                },
            )
            wk_df = df[["Disease", "PM2.5 Concentration", "Baseline Mortality", "Population"]]
            wk_df = wk_df.sort_index(axis=1)
            wk_df.plot(x="Disease", kind="bar", stacked=True, ax=ax, color=["gold", "cornflowerblue", "lightgreen"])
            sns.scatterplot(x="Disease", y="Overall", data=df, ax=ax)

            ax.set_title(country_long_name)
            if k == 0:  # First column
                ax.set_ylabel("Change in Percent")
            if j == axes.shape[0] - 1:  # Last row
                ax.set_xlabel("Diseases")

        # Plot legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        for ax in axes.flatten():
            ax = ax.get_legend().remove()

        output_file = f"{output_dir}/{ssp}.png"
        plt.savefig(output_file)


if __name__ == "__main__":
    main()
