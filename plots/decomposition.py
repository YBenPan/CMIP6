import os
from re import L
import sys
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool
from lib.country import get_country_names, get_regions
from lib.regions import *
from lib.helper import pop_ssp_dict, init_by_factor

####################################################################################################
#### CREATE DECOMPOSITION GRAPHS OF MORTALITY
####################################################################################################


# Run Settings
ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]
diseases = ["COPD", "DEM", "IHD", "LC", "LRI", "Stroke", "T2D"]
age_groups = ["25-60", "60-80", "80+", "25+"]

# General Settings
parent_dir = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/Health"
home_dir = "/home/ybenp"


def get_models(ssp):
    """Get all models from given ssp"""
    files_2015 = sorted(
        glob(
            f"{parent_dir}/Baseline_Ben_2015_National_GBD/5_years/{ssp}/*/CountryMortalityAbsolute/Allcause_mean/*_2015_GEMM.csv"
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
        ["COPD", "DEM", "IHD", "LC", "LRI", "Stroke", "T2D"]
        if diseases is None
        else diseases
    )
    baseline = str(baseline)
    year = str(year)
    models = get_models(ssp)
    factor_values = []
    pop_ssp = pop_ssp_dict[ssp]

    for model in models:

        # Search Allcause first to get the number of files/realizations in the model
        search_str = f"{parent_dir}/Baseline_Ben_{baseline}_National_GBD/5_years/{ssp}/Pop_{pop_ssp}_{pop}/CountryMortalityAbsolute/Allcause_mean/all_ages_{model}_*_{year}_GEMM.csv"
        files = sorted(glob(search_str))
        model_values = np.zeros(len(files))

        for disease in diseases:

            search_str = f"{parent_dir}/Baseline_Ben_{baseline}_National_GBD/5_years/{ssp}/Pop_{pop_ssp}_{pop}/CountryMortalityAbsolute/{disease}_mean/all_ages_{model}_*_{year}_GEMM.csv"
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


def country_mort(
    pop,
    baseline,
    year,
    ssp,
    ages=None,
    diseases=None,
    countries=None,
):
    """Get country-level mortality from projections"""
    if countries is None:
        countries = [-1]
    ages = ["all_age_Mean"] if ages is None else ages
    diseases = (
        ["COPD", "DEM", "IHD", "LC", "LRI", "Stroke", "T2D"]
        if diseases is None
        else diseases
    )
    baseline = str(baseline)
    year = str(year)
    models = get_models(ssp)
    factor_values = []
    pop_ssp = pop_ssp_dict[ssp]

    all_values = np.zeros((len(models), len(countries)))

    for i, model in enumerate(models):

        # Search Allcause first to get the number of files/realizations in the model
        search_str = f"{parent_dir}/Baseline_Ben_{baseline}_National_GBD/5_years/{ssp}/Pop_{pop_ssp}_{pop}/CountryMortalityAbsolute/Allcause_mean/all_ages_{model}_*_{year}_GEMM.csv"
        files = sorted(glob(search_str))
        model_values = np.zeros((len(files), len(countries)))

        for disease in diseases:

            search_str = f"{parent_dir}/Baseline_Ben_{baseline}_National_GBD/5_years/{ssp}/Pop_{pop_ssp}_{pop}/CountryMortalityAbsolute/{disease}_mean/all_ages_{model}_*_{year}_GEMM.csv"
            files = sorted(glob(search_str))

            for j, file in enumerate(files):
                wk = pd.read_csv(file, usecols=np.arange(1, 49, 3))
                model_values[j] += wk.iloc[countries][ages].values.ravel()

        model_mean = np.mean(model_values, axis=0)
        all_values[i] = model_mean

    all_mean = np.mean(all_values, axis=0)
    return all_mean


def output_country_mort(
    pops,
    baselines,
    years,
    ages=None,
    diseases=None,
    countries=None
):
    """Output country-level mortality to csv files"""
    if countries is None:
        countries = np.arange(0, 193)
    ages = ["all_age_Mean"] if ages is None else ages
    diseases = (
        ["COPD", "DEM", "IHD", "LC", "LRI", "Stroke", "T2D"]
        if diseases is None
        else diseases
    )
    output_dir = f"/home/ybenp/CMIP6_Images/Mortality/decomposition/country"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mort.csv")

    data = np.zeros((len(countries), len(years) * len(ssps)))
    col_labels = []

    for i, year in enumerate(years): 
        for j, ssp in enumerate(ssps): 
            index = i * len(ssps) + j
            col_labels.append(f"{year} {ssp}")
            data[:, index] = country_mort(pops[i], baselines[i], years[i], ssp, ages, diseases, countries)
    
    df = pd.DataFrame(data, columns=col_labels, index=country_names[:-1])
    df.to_csv(output_file)


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


def decompose(factor_name, factors, ssp, region, countries, output_type):
    """Compute contributions of each factor using different combinations of pop and baseline scenarios"""
    morts_2015 = []
    morts_2040 = []
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
        mort_2015 = mort("2010", "2015", 2015, ssp, ages, diseases, countries)
        mort_2040 = mort("var", "2040", 2040, ssp, ages, diseases, countries)

        # Method: (Pop 2010, Base 2015, Year 2015) -> PM2.5 Contribution
        #         (Pop 2010, Base 2015, Year 2040) -> Baseline Mortality Contribution
        #         (Pop 2010, Base 2040, Year 2040) -> Population Contribution by age groups
        #         (Pop var, Base 2040, Year 2040)
        pm = mort("2010", "2015", 2040, ssp, ages, diseases, countries) - mort(
            "2010", "2015", 2015, ssp, ages, diseases, countries
        )
        baseline = mort("2010", "2040", 2040, ssp, ages, diseases, countries) - mort(
            "2010", "2015", 2040, ssp, ages, diseases, countries
        )
        pop_25_60 = mort(
            "var", "2040", 2040, ssp, ages_25_60, diseases, countries
        ) - mort("2010", "2040", 2040, ssp, ages_25_60, diseases, countries)
        pop_60_80 = mort(
            "var", "2040", 2040, ssp, ages_60_80, diseases, countries
        ) - mort("2010", "2040", 2040, ssp, ages_60_80, diseases, countries)
        pop_80plus = mort(
            "var", "2040", 2040, ssp, ages_80plus, diseases, countries
        ) - mort("2010", "2040", 2040, ssp, ages_80plus, diseases, countries)

        # Calculate contribution in percent
        if output_type == "pct":
            pm = np.round(pm / ref * 100, 1)
            baseline = np.round(baseline / ref * 100, 1)
            pop_25_60 = np.round(pop_25_60 / ref * 100, 1)
            pop_60_80 = np.round(pop_60_80 / ref * 100, 1)
            pop_80plus = np.round(pop_80plus / ref * 100, 1)
            delta = np.round(delta / ref * 100, 1)

        morts_2015.append(mort_2015)
        morts_2040.append(mort_2040)
        pms.append(pm)
        baselines.append(baseline)
        pops_25_60.append(pop_25_60)
        pops_60_80.append(pop_60_80)
        pops_80plus.append(pop_80plus)
        deltas.append(delta)

        print(f"{ssp}, {region}, {factor}:")
        print(f"PM Contribution: {pm};")
        print(f"Baseline Contribution: {baseline};")
        print(f"Pop Contribution: {pop_25_60}, {pop_60_80}, {pop_80plus};")
        print(f"Overall Change: {delta}")
        print()

    return (
        morts_2015,
        morts_2040,
        pms,
        baselines,
        pops_25_60,
        pops_60_80,
        pops_80plus,
        deltas,
    )


def visualize(factor_name, region_source, change_type):
    """Driver program for visualization/output"""
    print(f"Job started: {factor_name}, {region_source}, {change_type}")

    # Output directory setting
    output_dir = f"/home/ybenp/CMIP6_Images/Mortality/decomposition/{region_source}"
    os.makedirs(output_dir, exist_ok=True)

    # Choose factor
    if factor_name == "Disease":
        factors = diseases
    elif factor_name == "Age":
        raise Exception("Plotting by age still in development")
        # factors = age_groups
    elif factor_name == "SSP":
        factors = ssps
    else:
        raise NameError(f"{factor_name} not found!")

    # Get countries and regions
    country_dict = get_country_names()
    regions, region_countries, region_countries_names = get_regions(region_source)

    sns.set_style("ticks")

    overall_df = pd.DataFrame()

    for ssp in ssps:

        # Plotting Settings

        if region_source == "GBD":
            rows = 5
            cols = 5
            fig, axes = plt.subplots(rows, cols, figsize=(20, 25))
        elif region_source == "SDI":
            rows = 3
            cols = 2
            fig, axes = plt.subplots(rows, cols, figsize=(12, 15))
        elif region_source == "GBD_super":
            rows = 2
            cols = 4
            fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
        ymin = -100
        ymax = 200

        # Initialize plotting
        # fig.suptitle(
        #     f"Decomposition of changes in PM2.5-attributable mortality from 2015 to 2040 {'in' + ssp if factor_name != 'SSP' else ''}by {factor_name}"
        # )

        ssp_df = pd.DataFrame()

        for i, (region, countries, countries_names) in enumerate(
            zip(regions, region_countries, region_countries_names)
        ):

            # Select current ax:
            j = i // cols
            k = i % cols

            ax = axes[j, k]

            # Compute contributions
            (
                morts_2015,
                morts_2040,
                pms,
                baselines,
                pops_25_60,
                pops_60_80,
                pops_80plus,
                deltas,
            ) = decompose(factor_name, factors, ssp, region, countries, change_type)
            # Visualize with a stacked bar plot

            df_dict = {
                "Region": region,
                factor_name: factors,
                "PM2.5 Concentration": pms,
                "Baseline Mortality": baselines,
                "Population 25-60": pops_25_60,
                "Population 60-80": pops_60_80,
                "Population 80+": pops_80plus,
                "Overall": deltas,
            }
            if factor_name != "SSP":
                df_dict["SSP"] = ssp
            if change_type == "absolute":
                df_dict["2015 Mortality"] = morts_2015
                df_dict["2040 Mortality"] = morts_2040
            df = pd.DataFrame(df_dict)

            ssp_df = ssp_df.append(df)
            wk_df = df[
                [
                    factor_name,
                    "Baseline Mortality",
                    "PM2.5 Concentration",
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
                    "lightpink",
                    "salmon",
                    "firebrick",
                ],
            )

            # Plot dots representing overall change in mortality
            sns.scatterplot(
                x=factor_name, y="Overall", data=df, ax=ax, color="mediumseagreen"
            )

            # Plot dotted line at 0%
            ax.axhline(0, ls="--", color="black")

            # Set labels and tick labels
            ax.set_ylim([ymin, ymax])
            ax.set_title(region)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if k == 0:
                ax.set_yticklabels(
                    ["-100%", "-50%", "0%", "50%", "100%", "150%", "200%", "250%"]
                )
                ax.set_ylabel("Pct Change")
            if j == rows - 1:
                ax.set_xlabel(factor_name)
                if factor_name == "SSP":
                    ax.set_xticklabels(["1", "2", "3", "5"], rotation=0)
                else:
                    ax.set_xticklabels(factors)

        # Plot legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        labels = [
            "Baseline Mortality",
            "PM2.5 Concentration",
            "Population 25-60",
            "Population 60-80",
            "Population 80+",
        ]
        fig.legend(handles, labels, loc="upper right")
        for i, ax in enumerate(axes.flatten()):
            if i >= len(regions):
                continue
            ax = ax.get_legend().remove()
        fig.tight_layout(rect=[0, 0.03, 0.93, 0.90])
        sns.despine()

        output_file = f"{output_dir}/{factor_name}{('_' + ssp) if factor_name != 'SSP' else ''}_{change_type}.png"
        if change_type == "pct":
            plt.savefig(output_file)
        overall_df = overall_df.append(ssp_df)
        if factor_name == "SSP":
            break

    output_file = os.path.join(output_dir, f"{factor_name}_{change_type}.csv")
    overall_df = overall_df.reset_index(drop=True)
    overall_df.to_csv(output_file, index=False)

    return f"Done: {factor_name}, {region_source}, {change_type}"


def main():
    # # Multiprocessing (Doesn't work because of system lock)
    # factor_names = ["Disease", "SSP"]
    # region_sources = ["GBD", "SDI"]
    # change_types = ["absolute", "pct"]
    # args = list(itertools.product(factor_names, region_sources, change_types))
    # with Pool() as pool:
    #     for result in pool.starmap(visualize, args):
    #         print(result, flush=True)

    # Test country mortality function
    # print(
    #     country_mort(
    #         pop="var",
    #         baseline="2040",
    #         year=2040,
    #         ssp="ssp245",
    #         countries=np.arange(0, 193),
    #     )
    # )

    output_country_mort(
        pops=["2010", "var"],
        baselines=[2015, 2040],
        years=[2015, 2040],
        ages=None,
        diseases=None,
        countries=None
    )

    # Get arguments from CLI
    # assert len(sys.argv) == 4
    # factor_name, region_source, change_type = sys.argv[1:]

    # visualize(factor_name, region_source, change_type)


if __name__ == "__main__":
    main()
