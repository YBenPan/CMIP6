import os
from glob import glob
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy
from decomposition import multi_year_mort
from lib.country import get_country_names, get_regions
from lib.regions import *
from lib.helper import pop_ssp_dict, init_by_factor
from lib.mean import mean
from lib.map import get_countries_mask

####################################################################################################
#### CREATE STACKED BAR PLOT OF GLOBAL MORTALITY IN 2015, 2030, AND 2040
####################################################################################################

# General Settings
parent_dir = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/Health"
output_dir = "/home/ybenp/CMIP6_Images/Mortality/global_mort"

# Run Settings
ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]
years = [2015, 2020, 2030, 2040]
diseases = ["COPD", "IHD", "LC", "LRI", "Stroke", "T2D"]
age_groups = ["25-60", "60-80", "80+"]
latitude = np.arange(-89.75, 90.25, 0.5)
longitude = np.arange(0.25, 360.25, 0.5)

# Custom region settings
country_dict = get_country_names()
regions, region_countries, region_countries_names = get_regions()

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
                print(
                    f"Region {region}. {year}, {ssp}, {pop}, {baseline}, {factor} mean: {factor_mean}; STD: {std}"
                )
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


def map_year(year, countries=None):
    """Driver program for map plots of mortality in a specific year"""
    # Get country mask
    fractionCountries = get_countries_mask(countries=countries)

    sns.set()
    bounds = [0, 100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 4000]
    bounds = [x / 2 for x in bounds]
    vmin = bounds[0]
    vmax = bounds[-1]
    cmap = matplotlib.cm.get_cmap("jet", lut=len(bounds) + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(18, 8)
    fig.suptitle(f"PM2.5 attributable mortality in {year}")

    all_data = []
    for i, ssp in enumerate(ssps):
        
        mort, awm, pwm = mean(ssp, year, fractionCountries, type="Mortality")
        data = mort
        all_data.append(data)

    all_data = np.mean(all_data, axis=0)

    ax.pcolormesh(longitude, latitude, data, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=0.5)
    ax.coastlines(resolution="10m")

    output_dir = "/home/ybenp/CMIP6_Images/Mortality/"
    os.makedirs(output_dir, exist_ok=True)
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        ticks=bounds,
        spacing="proportional",
        shrink=0.9,
    )
    cbar_label = "Mortality"
    cbar.set_label(cbar_label)
    plt.savefig(os.path.join(output_dir, "map", f"World_{year}.png"))
    plt.close(fig)   


def main():
    # for (region, countries, countries_names) in zip(
    #     regions, region_countries, region_countries_names
    # ):
    #     diseases_stacked(
    #         factor_name="Disease",
    #         factors=diseases,
    #         pop="var",
    #         baseline="2040",
    #         countries=countries,
    #         region=region,
    #     )
    #
    # # Get 2015 global mortality numbers
    # all_means = []
    # for ssp in ssps:
    #     mort_mean, std = multi_year_mort(
    #         pop="2010",
    #         baseline="2015",
    #         year=2015,
    #         ssp=ssp,
    #     )
    #     all_means.append(mort_mean)
    #     print(mort_mean, std)
    # print(np.mean(all_means))

    # # Get 2040 globall mortality numbers
    # all_means = []
    # for ssp in ssps:
    #     mort_mean, std = multi_year_mort(
    #         pop="var",
    #         baseline="2040",
    #         year=2040,
    #         ssp=ssp,
    #     )
    #     all_means.append(mort_mean)
    #     print(mort_mean, std)
    # print(np.mean(all_means))

    map_year(year=2015)


if __name__ == "__main__":
    main()
