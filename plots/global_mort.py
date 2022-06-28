import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from decomposition import multi_year_mort, init_by_factor, get_country_names
from regions import *

####################################################################################################
#### CREATE STACKED BAR PLOT OF GLOBAL MORTALITY IN 2015, 2030, AND 2040
####################################################################################################

# General Settings
parent_dir = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/Health"
output_dir = "/home/ybenp/CMIP6_Images/Mortality/global_mort"
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
years = [2015, 2020, 2030, 2040]
diseases = ["COPD", "IHD", "LC", "LRI", "Stroke", "T2D"]
age_groups = ["25-60", "60-80", "80+"]


# Custom region settings
region_countries_dict = {
    "W. Europe": Western_Europe,
    "Central Europe": Central_Europe,
    "E. Europe": Eastern_Europe,
    "Canada, US": High_income_North_America,
    "Australia, New Zealand": Australasia,

    "Caribbean": Caribbean,
    "Central America": Central_Latin_America,
    "Argentina, Chile, Uruguay": Southern_Latin_America,
    "Brazil, Paraguay": Tropical_Latin_America,
    "Bolivia, Ecuador, Peru": Andean_Latin_America,

    "Central Asia": Central_Asia,
    "South Asia": South_Asia,
    "East Asia": East_Asia,
    "Brunei, Japan, Singapore, S. Korea": High_income_Asia_Pacific,
    "S.E. Asia": Southeast_Asia,

    "N. Africa and Middle East": North_Africa_and_Middle_East,
    "Central Africa": Central_Sub_Saharan_Africa,
    "E. Africa": Eastern_Sub_Saharan_Africa,
    "S. Africa": Southern_Sub_Saharan_Africa,
    "W. Africa": Western_Sub_Saharan_Africa,

    "World": ["World"],
}
country_dict = get_country_names()
region_countries_names = list(region_countries_dict.values())
regions = list(region_countries_dict.keys())
region_countries = [
    [country_dict[country_name] for country_name in countries_names] for countries_names in region_countries_names
]
assert len(regions) == len(region_countries) == len(region_countries_names)


def diseases_stacked(factor_name, factors, pop, baseline, countries=None, region=None):
    if countries == None:
        countries = [-1]
    if region == None:
        region = "World"
    df = pd.DataFrame(columns=["Year", "SSP", factor_name, "Mean"])
    xlabels = []
    for i, year in enumerate(years):

        for j, ssp in enumerate(ssps):
            pop_ssp = pop_ssp_dict[ssp]

            for p, factor in enumerate(factors):
                ages, disease = init_by_factor(factor_name, factor)

                factor_mean, std = multi_year_mort(
                    pop=pop,
                    baseline=baseline,
                    year=year,
                    ssp=ssp,
                    ages=ages,
                    disease=disease,
                    countries=countries,
                    return_values=True,
                )
                df = df.append(
                    {
                        "Year": year,
                        "SSP": ssp,
                        factor_name: factor,
                        "Mean": factor_mean,
                        "STD": std,
                    },
                    ignore_index=True,
                )
                print(f"Region {region}. {year}, {ssp}, {pop}, {baseline}, {factor} mean: {factor_mean}; STD: {std}")
            xlabels.append(f"{year}, {ssp}")

    sns.set()
    g = sns.catplot(kind="bar", data=df, col="Year", x="SSP", y="Mean", hue=factor_name)
    g.set_axis_labels("SSP", "Number of Deaths")
    g.set_xticklabels(["1", "2", "3", "5"])
    g.set_titles("{col_name}")
    fig = g.fig
    fig.suptitle(region)
    fig.tight_layout(rect=[0, 0.03, 0.93, 0.95])

    # Add error bars
    for i, year in enumerate(years):
        ax = g.axes[0][i]
        year_df = df[df["Year"] == year]
        year_stds = year_df["STD"].values
        x_coords = sorted([p.get_x() + 0.5 * p.get_width() for p in ax.patches])
        y_coords = year_df["Mean"].values
        ax.errorbar(x=x_coords, y=y_coords, yerr=year_stds, fmt="none", color="black")

        all_df = year_df.groupby(by=["Year", "SSP"]).sum()
        all_means = all_df["Mean"].values
        x_coords = [0, 1, 2, 3]
        y_coords = all_means
        ax.scatter(x=x_coords, y=y_coords)

    output_file = f"{output_dir}/{region}_{factor_name}_{pop}_{baseline}.png"
    plt.savefig(output_file)
    # plt.show()
    plt.close(fig)


def main():
    for (region, countries, countries_names) in zip(regions, region_countries, region_countries_names):   
        diseases_stacked(factor_name="Disease", factors=diseases, pop="var", baseline="2040", countries=countries,
                         region=region)



if __name__ == "__main__":
    main()
