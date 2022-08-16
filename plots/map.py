from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import os
from decomposition import mort
from lib.helper import init_by_factor
from multiprocessing import Pool

ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]
diseases = ["COPD", "DEM", "IHD", "LC", "LRI", "Stroke", "T2D"]
# diseases = ["Allcause"]

# country_names = ["United States of America", "India", "China"]
country_codes = np.arange(0, 193)
# country_codes = [183, 77, 35]
country_path = "/home/ybenp/CMIP6_data/population/national_pop/countryvalue_blank.csv"
wk = pd.read_csv(country_path, usecols=[1])
country_names = wk["COUNTRY"].tolist()
country_conversion_dict = {
    "Russian Federation": "Russia",
    "Bahamas": "The Bahamas",
    "Republic of the Congo": "Congo",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "eSwatini": "Swaziland",
    "Lao PDR": "Laos",
    "Dem. Rep. Korea": "North Korea",
    "Republic of Korea": "South Korea",
    "Brunei Darussalam": "Brunei",
    "Somaliland": "Somalia",
    "Serbia": "Serbia+Montenegro",
    "Montenegro": "Serbia+Montenegro",
    "Greenland": "Denmark",
    "South Sudan": "Sudan",
    "Taiwan": "China",
    "Western Sahara": "Morocco",
}
pop_ssp_dict = {
    "ssp119": "ssp1",
    "ssp126": "ssp1",
    "ssp245": "ssp2",
    "ssp370": "ssp3",
    "ssp434": "ssp2",
    "ssp460": "ssp2",
    "ssp585": "ssp1",
}

shpfilename = shpreader.natural_earth(
    resolution="110m", category="cultural", name="admin_0_countries"
)
reader = shpreader.Reader(shpfilename)
countries = reader.records()
tmp = sorted([country.attributes["NAME_LONG"] for country in countries])


def findColor(colorbounds, colormap, num):

    for x in np.arange(1, len(colorbounds)):
        if num >= colorbounds[x]:
            continue
        else:
            return colormap(x - 1)
    return colormap(len(colorbounds) - 2)


def output_diff(data, diff_file, ssp, models):
    if ssp == ssps[0] and os.path.exists(diff_file):
        os.remove(diff_file)
    with open(diff_file, "a") as file:
        if ssp == ssps[0] and os.path.exists(diff_file):
            file.write("SSP,Model,Diff,Pct_change\n")
        for i, model in enumerate(models):
            print(
                f"{model}: Diff: {np.round(np.sum(data[1, i, :, :] - data[0, i, :, :]), 0)}, Percent Change: {np.round(np.sum(data[1, i, :, :] - data[0, i, :, :]) / np.sum(data[0, i, :, :]), 2)}"
            )
            file.write(
                f"{ssp},{model},{np.sum(data[1, i, :, :] - data[0, i, :, :])},{np.sum(data[1, i, :, :] - data[0, i, :, :]) / np.sum(data[0, i, :, :])}\n"
            )


def contribution_process(ssp, factor):
    data = np.zeros(len(country_names))

    for country in country_codes:

        if factor == "PM25":
            contrib = mort("2010", "2015", 2040, ssp, None, None, country) - mort(
                "2010", "2015", 2015, ssp, None, None, country
            )
        elif factor == "Population":
            contrib = mort("var", "2040", 2040, ssp, None, None, country) - mort(
                "2010", "2040", 2040, ssp, None, None, country
            )
        elif factor == "Aging":
            ages_60plus = [
                "age_60_64_Mean",
                "age_65_69_Mean",
                "age_70_74_Mean",
                "age_75_79_Mean",
                "age_80_84_Mean",
                "age_85_89_Mean",
                "age_90_94_Mean",
                "post95_Mean",
            ]
            contrib = mort("var", "2040", 2040, ssp, ages_60plus, None, country) - mort(
                "2010", "2040", 2040, ssp, ages_60plus, None, country)
        elif factor == "Population Growth":
            ages_25_60 = [
                "age_25_29_Mean",
                "age_30_34_Mean",
                "age_35_39_Mean",
                "age_40_44_Mean",
                "age_45_49_Mean",
                "age_50_54_Mean",
                "age_55_59_Mean",
            ]
            contrib = mort("var", "2040", 2040, ssp, ages_25_60, None, country) - mort(
                "2010", "2040", 2040, ssp, ages_25_60, None, country)
        elif factor == "Baseline Mortality":
            contrib = mort("2010", "2040", 2040, ssp, None, None, country) - mort(
            "2010", "2015", 2040, ssp, None, None, country
        )
        ref = mort("2010", "2015", 2015, ssp, None, None, country)
        data[country] = contrib / ref * 100
        print(f"{ssp}, {country_names[country]}: {data[country]}%")

    print(f"DONE: {ssp}")

    return data


def contribution(factor):
    """Plot the contribution of PM2.5 by SSPs"""
    # Define settings
    if factor == "PM25":
        bounds = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
        cmap = matplotlib.cm.get_cmap("coolwarm", lut=len(bounds) + 1)
    elif factor == "Population" or factor == "Aging":
        bounds = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300]
        cmap = matplotlib.cm.get_cmap("Reds", lut=len(bounds) + 1)
    elif factor == "Population Growth":
        bounds = [0, 10, 20, 30, 40, 50, 60, 70, 80]
        cmap = matplotlib.cm.get_cmap("Reds", lut=len(bounds) + 1)
    elif factor == "Baseline Mortality":
        bounds = [-60, -50, -40, -30, -20, -10, 0]
        cmap = matplotlib.cm.get_cmap("Blues_r", lut=len(bounds) + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(2, 2, subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(16, 8)

    factor_args = [factor] * len(ssps)
    pool = Pool()
    data = pool.starmap(contribution_process, zip(ssps, factor_args))

    for i, ssp in enumerate(ssps):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        # ax.add_feature(cartopy.feature.LAND)
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.BORDERS)

        countries = reader.records()
        for country in countries:
            cur_name = country.attributes["NAME_LONG"]
            if cur_name in country_conversion_dict:
                cur_name = country_conversion_dict[cur_name]
            if cur_name in country_names:
                cur_index = country_names.index(cur_name)
                if np.isnan(data[i][cur_index]):
                    continue
                color = findColor(bounds, cmap, data[i][cur_index])
                ax.add_geometries(
                    [country.geometry], ccrs.PlateCarree(), facecolor=color
                )

        ax.set_title(ssp)

    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axes.ravel().tolist(),
        shrink=0.6,
        ticks=bounds,
        spacing="uniform",
        format="%d",
    )
    cbar.ax.set_ylabel(f"Percent Change")
    fig.suptitle(f"Contribution of {factor}")

    output_dir = f"/home/ybenp/CMIP6_Images/Mortality/map/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{factor}_contrib.png"
    plt.savefig(output_file)
    plt.close(fig)
    # del fig, ax, cbar


def main():
    contribution(factor="Baseline Mortality")


if __name__ == "__main__":
    main()
