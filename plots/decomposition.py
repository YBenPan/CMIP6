import os
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

####################################################################################################
#### CREATE DECOMPOSITION GRAPHS OF MORTALITY
####################################################################################################

# General Settings
parent_dir = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/Health"
output_dir = "/home/ybenp/CMIP6_Images/Mortality/decomposition"
pop_ssp_dict = {
    "ssp119": "ssp1",
    "ssp126": "ssp1",
    "ssp245": "ssp2",
    "ssp370": "ssp3",
    "ssp434": "ssp2",
    "ssp460": "ssp2",
    "ssp585": "ssp1",
}

# Run Settings
ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]
diseases = ["Allcause", "COPD", "IHD", "LC", "LRI", "Stroke", "T2D"]
age_groups = ["25-60", "60-80", "80+", "25+"]
countries = [-1, 35, 77, 183, 85, 53]
country_long_names = ["World", "China", "India", "The United States", "Japan", "Egypt"]
factor_name = "Age"
if factor_name == "Disease":
    factors = diseases
elif factor_name == "Age":
    factors = age_groups
else:
    raise NameError(f"{factor_name} not found!")


def get_models(ssp):
    """Get all models from given ssp"""
    files_2015 = sorted(
        glob(
            f"{parent_dir}/Baseline_Ben_2015_National/5_years/{ssp}/*/CountryMortalityAbsolute/Allcause_mean/*_2015_GEMM.csv"
        )
    )
    models = sorted(set([file.split("/")[-1].split("_")[2] for file in files_2015]))
    # Add or remove models here
    models = [model for model in models if "EC-Earth3-AerChem" not in model]
    # models = [model for model in models if model in ["CESM2-WACCM6", "GFDL-ESM4", "GISS-E2-1-G", "MIROC-ES2L",
    # "MIROC6", "MRI-ESM2-0", "NorESM2-LM"]]
    return models


def mort(pop, baseline, year, ssp, ages=None, disease=None, country=-1):
    """Get mortality value from projections given a set of conditions"""
    ages = ["all_age_Mean"] if ages is None else ages
    disease = "Allcause" if disease is None else disease
    models = get_models(ssp)
    factor_values = []
    pop_ssp = pop_ssp_dict[ssp]

    for i, model in enumerate(models):
        search_str = f"{parent_dir}/Baseline_Ben_{baseline}_National/5_years/{ssp}/Pop_{pop_ssp}_{pop}/CountryMortalityAbsolute/{disease}_mean/all_ages_{model}_*_{year}_GEMM.csv"
        files = sorted(glob(search_str))
        model_values = np.zeros(len(files))

        for j, file in enumerate(files):
            wk = pd.read_csv(file, usecols=np.arange(1, 49, 3))
            model_values[j] = np.sum(wk.iloc[country][ages].values)
        if len(model_values) == 0:
            print(f"No values found in {model}", year, ssp, pop, baseline, disease)
        model_mean = np.mean(model_values)
        factor_values.append(model_mean)
    if len(factor_values) == 0:
        print("No models found", year, ssp, pop, baseline, disease)
    factor_mean = np.mean(factor_values)
    return factor_mean


def init_by_factor(factor_name, factor):
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
            raise NameError(f"{factor} age group not found!")
        disease = None
    elif factor_name == "Disease":
        ages = None
        disease = factor
    else:
        raise NameError(f"{factor_name} factor not found!")
    return ages, disease


def compute(factor_name, factors, ssp, country, country_long_name):
    pms = []
    baselines = []
    pops = []
    deltas = []
    for factor in factors:
        ages, disease = init_by_factor(factor_name, factor)
        ref = mort(
            pop="2010",
            baseline="2015",
            year="2015",
            ssp=ssp,
            ages=ages,
            disease=disease,
            country=country,
        )
        delta = mort(
            pop="var",
            baseline="2040",
            year="2040",
            ssp=ssp,
            ages=ages,
            disease=disease,
            country=country,
        ) - mort(
            pop="2010",
            baseline="2015",
            year="2015",
            ssp=ssp,
            ages=ages,
            disease=disease,
            country=country,
        )

        # JF Method: (Pop 2010, Base 2015, Year 2015) ->
        #            (Pop 2010, Base 2015, Year 2040) ->
        #            (Pop 2010, Base 2040, Year 2040) ->
        #            (Pop var, Base 2040, Year 2040)
        pm_contribution = mort(
            pop="2010",
            baseline="2015",
            year="2040",
            ssp=ssp,
            ages=ages,
            disease=disease,
            country=country,
        ) - mort(
            pop="2010",
            baseline="2015",
            year="2015",
            ages=ages,
            ssp=ssp,
            disease=disease,
            country=country,
        )
        baseline_contribution = mort(
            pop="2010",
            baseline="2040",
            year="2040",
            ages=ages,
            ssp=ssp,
            disease=disease,
            country=country,
        ) - mort(
            pop="2010",
            baseline="2015",
            year="2040",
            ages=ages,
            ssp=ssp,
            disease=disease,
            country=country,
        )
        pop_contribution = mort(
            pop="var",
            baseline="2040",
            year="2040",
            ages=ages,
            ssp=ssp,
            disease=disease,
            country=country,
        ) - mort(
            pop="2010",
            baseline="2040",
            year="2040",
            ages=ages,
            ssp=ssp,
            disease=disease,
            country=country,
        )

        pm_percent = np.round(pm_contribution / ref * 100, 1)
        baseline_percent = np.round(baseline_contribution / ref * 100, 1)
        pop_percent = np.round(pop_contribution / ref * 100, 1)
        delta_percent = np.round(delta / ref * 100, 1)

        pms.append(pm_percent)
        baselines.append(baseline_percent)
        pops.append(pop_percent)
        deltas.append(delta_percent)

        print(
            f"{ssp}, {country_long_name}, {factor}: PM Contribution: {pm_percent}%; Population Contribution: "
            f"{pop_percent}%; Baseline Contribution: {baseline_percent}%"
        )
        print(f"{ssp}, {country_long_name}, {factor}: Overall Change: {delta_percent}%")
    return pms, baselines, pops, deltas


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
        fig, axes = plt.subplots(2, 3, figsize=(16, 12))
        fig.suptitle(
            "Decomposition of changes in mortality attributable to PM2.5 from 2015 to 2040"
        )
        # Plotting Settings
        ymin = -100
        ymax = 400

        for i, (country, country_long_name) in enumerate(
            zip(countries, country_long_names)
        ):

            # Select current ax:
            j = i // 3
            k = i % 3
            ax = axes[j, k]
            ax.set_ylim([ymin, ymax])

            # Compute contributions
            pms, baselines, pops, deltas = compute(
                factor_name=factor_name,
                factors=factors,
                ssp=ssp,
                country=country,
                country_long_name=country_long_name,
            )
            # Visualize with a stacked bar plot

            df = pd.DataFrame(
                {
                    factor_name: factors,
                    "PM2.5 Concentration": pms,
                    "Baseline Mortality": baselines,
                    "Population": pops,
                    "Overall": deltas,
                },
            )
            wk_df = df[
                [factor_name, "PM2.5 Concentration", "Baseline Mortality", "Population"]
            ]
            wk_df = wk_df.sort_index(axis=1)
            wk_df.plot(
                x=factor_name,
                kind="bar",
                stacked=True,
                ax=ax,
                color=["gold", "cornflowerblue", "lightgreen"],
            )
            sns.scatterplot(x=factor_name, y="Overall", data=df, ax=ax)

            ax.set_title(country_long_name)
            if k == 0:  # First column
                ax.set_ylabel("Change in Percent")
            else:
                ax.set_ylabel("")
            if j == axes.shape[0] - 1:  # Last row
                ax.set_xlabel(factor_name)
            else:
                ax.set_xlabel("")
        # Plot legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        for ax in axes.flatten():
            ax = ax.get_legend().remove()

        output_file = f"{output_dir}/{factor_name}_{ssp}.png"
        plt.savefig(output_file)


if __name__ == "__main__":
    main()
