import os
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lib.country import get_country_names, get_regions
from lib.regions import *
from lib.helper import pop_ssp_dict, init_by_factor

####################################################################################################
#### CREATE DECOMPOSITION GRAPHS OF MORTALITY
####################################################################################################

# General Settings
parent_dir = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/Health"
home_dir = "/home/ybenp"
output_dir = "/home/ybenp/CMIP6_Images/Mortality/decomposition"

# Run Settings
ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]
diseases = ["COPD", "IHD", "LC", "LRI", "Stroke", "T2D"]
age_groups = ["25-60", "60-80", "80+", "25+"]


# Get countries and regions
country_dict = get_country_names()
regions, region_countries, region_countries_names = get_regions()


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
    models = [
        model
        for model in models
        if model
        in [
            "CESM2-WACCM6",
            "GFDL-ESM4",
            "GISS-E2-1-G",
            "MIROC-ES2L",
            "MIROC6",
            "MRI-ESM2-0",
            "NorESM2-LM",
        ]
    ]
    return models


def mort(
    pop,
    baseline,
    year,
    ssp,
    ages=None,
    diseases=None,
    countries=None,
    return_values=False,
):
    """Get mortality value from projections given a set of conditions"""
    if countries is None:
        countries = [-1]
    ages = ["all_age_Mean"] if ages is None else ages
    diseases = (
        ["COPD", "IHD", "LC", "LRI", "Stroke", "T2D"] if diseases is None else diseases
    )
    year = str(year)
    models = get_models(ssp)
    factor_values = []
    pop_ssp = pop_ssp_dict[ssp]

    for model in models:

        # Search Allcause first to get the number of files/realizations in the model
        search_str = f"{parent_dir}/Baseline_Ben_{baseline}_National/5_years/{ssp}/Pop_{pop_ssp}_{pop}/CountryMortalityAbsolute/Allcause_mean/all_ages_{model}_*_{year}_GEMM.csv"
        files = sorted(glob(search_str))
        model_values = np.zeros(len(files))

        for disease in diseases:

            search_str = f"{parent_dir}/Baseline_Ben_{baseline}_National/5_years/{ssp}/Pop_{pop_ssp}_{pop}/CountryMortalityAbsolute/{disease}_mean/all_ages_{model}_*_{year}_GEMM.csv"
            files = sorted(glob(search_str))

            for j, file in enumerate(files):
                wk = pd.read_csv(file, usecols=np.arange(1, 49, 3))
                model_values[j] += np.sum(wk.iloc[countries][ages].values)

        model_mean = np.mean(model_values)
        # print(f"{model}: {model_mean}")
        factor_values.append(model_mean)
    if len(factor_values) == 0:
        print("No models found", year, ssp, pop, baseline, disease)
    factor_mean = np.mean(factor_values)
    if return_values:
        return factor_values
    return factor_mean


def multi_year_mort(
    pop,
    baseline,
    year,
    ssp,
    ages=None,
    diseases=None,
    countries=None,
    return_values=True,
):
    """Call mort() for multiple years"""
    if countries is None:
        countries = [-1]
    years = np.arange(year - 2, year + 3)  # +/- 2
    if year == 2015 or year == 2040:
        years = [year]
    morts = list()
    for year in years:
        morts.append(
            mort(pop, baseline, year, ssp, ages, diseases, countries, return_values)
        )
    mort_mean = np.mean(morts)
    std = np.std(morts)
    return mort_mean, std


def decompose(factor_name, factors, ssp, region, countries):
    """Compute contributions of each factor using different combinations of pop and baseline scenarios"""
    pms = []
    baselines = []
    pops_25_60 = []
    ages_25_60 = [
        "age_25_29_Mean",
        "age_30_34_Mean",
        "age_35_39_Mean",
        "age_40_44_Mean",
        "age_45_49_Mean",
        "age_50_54_Mean",
        "age_55_59_Mean",
    ]
    pops_60_80 = []
    ages_60_80 = [
        "age_60_64_Mean",
        "age_65_69_Mean",
        "age_70_74_Mean",
        "age_75_79_Mean",
    ]
    pops_80plus = []
    ages_80plus = [
        "age_80_84_Mean",
        "age_85_89_Mean",
        "age_90_94_Mean",
        "post95_Mean",
    ]
    deltas = []
    for factor in factors:
        ages, diseases = init_by_factor(factor_name, factor)
        if factor_name == "SSP":
            ssp = factor
        ref = mort("2010", "2015", 2015, ssp, ages, diseases, countries)
        delta = mort("var", "2040", 2040, ssp, ages, diseases, countries) - mort(
            "2010", "2015", 2015, ssp, ages, diseases, countries
        )

        # Method: (Pop 2010, Base 2015, Year 2015) -> PM2.5 Contribution
        #         (Pop 2010, Base 2015, Year 2040) -> Baseline Mortality Contribution
        #         (Pop 2010, Base 2040, Year 2040) -> Population Contribution by age groups
        #         (Pop var, Base 2040, Year 2040)
        pm_contrib = mort("2010", "2015", 2040, ssp, ages, diseases, countries) - mort(
            "2010", "2015", 2015, ssp, ages, diseases, countries
        )
        baseline_contrib = mort(
            "2010", "2040", 2040, ssp, ages, diseases, countries
        ) - mort("2010", "2015", 2040, ssp, ages, diseases, countries)
        pop_25_60_contrib = mort(
            "var", "2040", 2040, ssp, ages_25_60, diseases, countries
        ) - mort("2010", "2040", 2040, ssp, ages_25_60, diseases, countries)
        pop_60_80_contrib = mort(
            "var", "2040", 2040, ssp, ages_60_80, diseases, countries
        ) - mort("2010", "2040", 2040, ssp, ages_60_80, diseases, countries)
        pop_80plus_contrib = mort(
            "var", "2040", 2040, ssp, ages_80plus, diseases, countries
        ) - mort("2010", "2040", 2040, ssp, ages_80plus, diseases, countries)

        # Calculate contribution in percent
        pm_pct = np.round(pm_contrib / ref * 100, 1)
        baseline_pct = np.round(baseline_contrib / ref * 100, 1)
        pop_25_60_pct = np.round(pop_25_60_contrib / ref * 100, 1)
        pop_60_80_pct = np.round(pop_60_80_contrib / ref * 100, 1)
        pop_80plus_pct = np.round(pop_80plus_contrib / ref * 100, 1)
        delta_pct = np.round(delta / ref * 100, 1)

        pms.append(pm_pct)
        baselines.append(baseline_pct)
        pops_25_60.append(pop_25_60_pct)
        pops_60_80.append(pop_60_80_pct)
        pops_80plus.append(pop_80plus_pct)
        deltas.append(delta_pct)

        print(f"{ssp}, {region}, {factor}:")
        print(f"PM Contribution: {pm_pct}%;")
        print(f"Baseline Contribution: {baseline_pct}%;")
        print(
            f"Pop Contribution: {pop_25_60_pct}%, {pop_60_80_pct}%, {pop_80plus_pct}%;"
        )
        print(f"Overall Change: {delta_pct}%")
        print()

    return pms, baselines, pops_25_60, pops_60_80, pops_80plus, deltas


def visualize(factor_name, factors):
    """Driver program for visualization/output"""

    sns.set()

    overall_df = pd.DataFrame()

    for ssp in ssps:

        rows = 5
        cols = 5

        # Initialize plotting
        fig, axes = plt.subplots(rows, cols, figsize=(20, 25))
        fig.suptitle(
            f"Decomposition of changes in PM2.5-attributable mortality from 2015 to 2040 {'in' + ssp if factor_name != 'SSP' else ''} by factor"
        )
        # Plotting Settings
        ymin = -100
        ymax = 400

        ssp_df = pd.DataFrame()

        for i, (region, countries, countries_names) in enumerate(
            zip(regions, region_countries, region_countries_names)
        ):

            # Select current ax:
            j = i // cols
            k = i % cols

            ax = axes[j, k]

            # Compute contributions
            pms, baselines, pops_25_60, pops_60_80, pops_80plus, deltas = decompose(
                factor_name=factor_name,
                factors=factors,
                ssp=ssp,
                region=region,
                countries=countries,
            )
            # Visualize with a stacked bar plot

            df = pd.DataFrame(
                {
                    "Region": region,
                    factor_name: factors,
                    # "SSP": ssp,
                    "PM2.5 Concentration": pms,
                    "Baseline Mortality": baselines,
                    "Population 25-60": pops_25_60,
                    "Population 60-80": pops_60_80,
                    "Population 80+": pops_80plus,
                    "Overall": deltas,
                },
            )
            ssp_df = ssp_df.append(df)
            wk_df = df[
                [
                    factor_name,
                    "PM2.5 Concentration",
                    "Baseline Mortality",
                    "Population 25-60",
                    "Population 60-80",
                    "Population 80+",
                ]
            ]
            wk_df = wk_df.sort_index(axis=1)
            wk_df.plot(
                x=factor_name,
                kind="bar",
                stacked=True,
                ax=ax,
                color=[
                    "gold",
                    "cornflowerblue",
                    "lightgreen",
                    "green",
                    "darkolivegreen",
                ],
            )

            # Plot dots representing overall change in mortality
            sns.scatterplot(
                x=factor_name, y="Overall", data=df, ax=ax, color="orangered"
            )
            ax.set_ylim([ymin, ymax])
            ax.set_title(region)
            ax.set_ylabel("")
            ax.set_xlabel(factor_name)
            ax.set_yticklabels(["-100%", "0%", "100%", "200%", "300%", "400%"])

            if factor_name == "SSP":
                ax.set_xticklabels(["1", "2", "3", "5"], rotation=0)

        # Plot legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        labels = [
            "PM2.5 Concentration",
            "Baseline Mortality",
            "Population 25-60",
            "Population 60-80",
            "Population 80+",
        ]
        fig.legend(handles, labels, loc="upper right")
        for i, ax in enumerate(axes.flatten()):
            if i >= len(regions):
                continue
            ax = ax.get_legend().remove()
        fig.tight_layout(rect=[0, 0.03, 0.93, 0.95])

        output_file = f"{output_dir}/{factor_name}{('_' + ssp) if factor_name != 'SSP' else ''}.png"
        plt.savefig(output_file)
        overall_df = overall_df.append(ssp_df)
        if factor_name == "SSP":
            break

    output_file = os.path.join(output_dir, f"{factor_name}.csv")
    overall_df = overall_df.reset_index(drop=True)
    overall_df.to_csv(output_file, index=False)


def main():
    # Choose factor
    factor_name = "SSP"
    if factor_name == "Disease":
        factors = diseases
    elif factor_name == "Age":
        raise Exception("Plotting by age still in development")
        # factors = age_groups
    elif factor_name == "SSP":
        factors = ssps
    else:
        raise NameError(f"{factor_name} not found!")
    visualize(factor_name, factors)


if __name__ == "__main__":
    main()
