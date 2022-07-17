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
from lib.helper import pop_ssp_dict, init_by_factor, pct_change
from lib.mean import mean
from lib.map import get_countries_mask

####################################################################################################
#### CREATE STACKED BAR PLOT OF GLOBAL MORTALITY IN 2015, 2030, AND 2040
####################################################################################################

# General Settings
parent_dir = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/Health"

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


def bar(factor_name, factors, pop, baseline, countries=None, region=None):
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

    output_dir = "/home/ybenp/CMIP6_Images/Mortality/bar"
    output_file = f"{output_dir}/{region}_{factor_name}_{pop}_{baseline}.png"
    plt.savefig(output_file)
    plt.close(fig)


def pie(factor_name, factors, countries=None, region=None):
    if countries == None:
        countries = [-1]
    if region == None:
        region = "World"
    mort_data = np.zeros((2, len(ssps), len(factors) + 1))  # 2015, 2040
    years = [2015, 2040]

    if factor_name == "Disease":
        factors = ["Allcause", "COPD", "IHD", "LC", "LRI", "Stroke", "T2D"]

    fig, axes = plt.subplots(2, 4)
    fig.suptitle(region)
    sns.set()

    for i, year in enumerate(years):

        for j, ssp in enumerate(ssps):

            for k, factor in enumerate(factors):
                ages, disease = init_by_factor(factor_name, factor)

                factor_mean, std = multi_year_mort(
                    pop="var",
                    baseline=year,
                    year=year,
                    ssp=ssp,
                    ages=ages,
                    disease=disease,
                    countries=countries,
                    return_values=True,
                )
                mort_data[i, j, k] = factor_mean

            # Other = Allcause - (COPD + ... + T2D)
            mort_data[i, j, 0] -= np.sum(mort_data[i, j, 1:])

            # Plot pie chart
            ax = axes[i, j]
            data = mort_data[i, j]
            labels = [f"{np.round(mort, -3)}%" for mort in mort_data[i, j]]
            ax.pie(
                data,
                labels=labels,
                autopct="%.1f",
                textprops={"size": 4},
                startangle=90,
            )

            # Transform to donut plot
            circle = plt.Circle((0, 0), 0.7, color="white")
            ax.add_patch(circle)
            total_deaths = f"{int(np.round(np.sum(mort_data[i, j]), -3)):,}"
            ax.set_title(f"{ssp}, {year}", fontsize=6)
            ax.text(0, 0, f"{total_deaths}", ha="center", va="center", fontsize=5)
            ax.text(0, -0.2, f"deaths", ha="center", va="center", fontsize=5)

    output_dir = "/home/ybenp/CMIP6_Images/Mortality/pie"
    output_file = f"{output_dir}/{region}_{factor_name}_other.png"
    factors[0] = "Other"
    plt.legend(labels=factors, bbox_to_anchor=(1.5, 0.5), fontsize=4)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    print(f"Done: {region}")


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


def map_delta():
    """Driver program for delta mortality map plots"""
    # Get country mask
    fractionCountries = get_countries_mask(countries=None)

    fig, axes = plt.subplots(2, 2, subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(18, 8)
    fig.suptitle(f"Mortality in 2040")

    # bounds = [-50, -40, -30, -20, -10, 0, 50, 100, 200, 300, 500]
    bounds = [0, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    vmin = bounds[0]
    vmax = bounds[-1]
    cmap = matplotlib.cm.get_cmap("YlOrRd", lut=len(bounds) + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    pct_change_data = np.zeros((len(regions), len(ssps)))

    for i, ssp in enumerate(ssps):

        print(ssp)

        for j, (region, countries, countries_names) in enumerate(
            zip(regions, region_countries, region_countries_names)
        ):
            fractionRegion = get_countries_mask(countries=countries)
            mort_2015, awm_2015, pwm_2015 = mean(
                ssp, 2015, fractionRegion, type="Mortality"
            )
            mort_2040, awm_2040, pwm_2040 = mean(
                ssp, 2040, fractionRegion, type="Mortality"
            )
            tot_mort_2015 = np.sum(mort_2015)
            tot_mort_2040 = np.sum(mort_2040)
            pct_change_data[j, i] = pct_change(tot_mort_2015, tot_mort_2040)
            print(
                f"{region}: Mortality Change: {pct_change(tot_mort_2015, tot_mort_2040)}%"
            )

        mort_2015, awm_2015, pwm_2015 = mean(
            ssp, 2015, fractionCountries, type="Mortality"
        )
        mort_2040, awm_2040, pwm_2040 = mean(
            ssp, 2040, fractionCountries, type="Mortality"
        )
        tot_mort_2015 = np.sum(mort_2015)
        tot_mort_2040 = np.sum(mort_2040)
        pct_change_data[-1, i] = pct_change(tot_mort_2015, tot_mort_2040)
        print(f"World: Mortality Change: {pct_change(tot_mort_2015, tot_mort_2040)}%")

        mort = mort_2040
        ax_i = i // 2
        ax_j = i % 2
        ax = axes[ax_i, ax_j]

        ax.pcolormesh(
            longitude, latitude, mort, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm
        )
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
        ax.add_feature(cartopy.feature.BORDERS, linewidth=0.5)
        ax.set_title(ssp)

    output_dir = "/home/ybenp/CMIP6_Images/Mortality/map"
    os.makedirs(output_dir, exist_ok=True)

    # Output csv
    # output_file = os.path.join(output_dir, "pct_change.csv")
    # df = pd.DataFrame(pct_change_data, index=regions, columns=ssps)
    # df.to_csv(output_file)

    # Add color bar
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        ticks=bounds,
        spacing="uniform",
        shrink=0.9,
    )
    cbar.set_label(f"Mortality")

    # Output figure
    output_file = os.path.join(output_dir, f"World_2040.png")
    plt.savefig(output_file)
    plt.close(fig)


def main():
    for (region, countries, countries_names) in zip(
        regions, region_countries, region_countries_names
    ):
        pie(
            factor_name="Disease",
            factors=diseases,
            countries=countries,
            region=region,
        )

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

    # map_year(year=2015)
    # map_delta()


if __name__ == "__main__":
    main()
