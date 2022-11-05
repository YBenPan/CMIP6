from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import os
import sys
from decomposition import mort, country_mort
from lib.helper import init_by_factor
from multiprocessing import Pool
from string import ascii_uppercase

ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]
output_ssps = ["SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]
diseases = ["COPD", "DEM", "IHD", "LC", "LRI", "Stroke", "T2D"]
# diseases = ["Allcause"]

country_codes = np.arange(0, 193)
country_path = "/home/ybenp/CMIP6_data/population/national_pop/countryvalue_blank.csv"
wk = pd.read_csv(country_path, usecols=[1])
country_names = wk["COUNTRY"].tolist()

# country_codes = [183, 77, 35, 140, 23, 30, 148]
# country_names = ["United States of America", "India", "China", "Russia", "Brazil", "Canada", "Saudi Arabia"]

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

    for x in np.arange(0, len(colorbounds)):
        if num >= colorbounds[x]:
            continue
        else:
            return colormap(x)
    return colormap(len(colorbounds))


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


def contribution_process(ssp, factor, type):
    data = np.zeros(len(country_codes))

    # Variable i is needed for testing purposes
    for i, country in enumerate(country_codes):

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
        elif factor == "Population Size":
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
        else:
            raise Exception(f"{factor} not valid!")
        ref = mort("2010", "2015", 2015, ssp, None, None, country)
        # Raw percent change relative to 2015 mortality
        if type == "pct":
            data[i] = contrib / ref * 100
        # How much of the overall change is due to this factor
        elif type == "frac":
            delta = mort("var", "2040", 2040, ssp, None, None, country) - mort(
                "2010", "2015", 2015, ssp, None, None, country
            )
            data[i] = contrib / delta * 100
            if abs(data[i]) > 200: 
                data[i] = 0
        elif type == "absolute":
            data[i] = contrib
        print(f"{ssp}, {factor}, {country_names[i]}: {data[i]}%")

    print(f"DONE: {ssp}")

    return data


def contribution(factor, type="pct", ssp=None):
    """Plot the contribution of PM2.5 by SSPs"""
    # Define settings
    if type == "absolute":
        bounds = [-100000, -10000, -1000, -100, -10, -1, 1, 10, 100, 1000, 10000, 100000, 500000, 1000000]
        cmap = matplotlib.cm.get_cmap("coolwarm", lut=len(bounds) + 1)
    else:
        if factor == "PM25":
            bounds = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
            cmap = matplotlib.cm.get_cmap("coolwarm", lut=len(bounds) + 1)
        elif factor == "Population" or factor == "Aging":
            bounds = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
            cmap = matplotlib.cm.get_cmap("Reds", lut=len(bounds) + 1)
        elif factor == "Population Size":
            bounds = [0, 10, 20, 30, 40, 50, 60, 70, 80]
            cmap = matplotlib.cm.get_cmap("Reds", lut=len(bounds) + 1)
        elif factor == "Baseline Mortality":
            bounds = [-60, -50, -40, -30, -20, -10, 0]
            cmap = matplotlib.cm.get_cmap("Blues_r", lut=len(bounds) + 1)
        elif factor == "SSP":
            # bounds = [-200, -100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100, 200]
            bounds = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
            cmap = matplotlib.cm.get_cmap("coolwarm", lut=len(bounds) + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Pre-emptively make output directory
    output_dir = f"/home/ybenp/CMIP6_Images/Mortality/map/"
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(15, 8)

    pool = Pool()
    if factor == "SSP":
        assert(ssp != None)
        factors = ["PM25", "Baseline Mortality", "Population Size", "Aging"]
        ssp_args = [ssp] * len(factors)
        type_args = [type] * len(factors)
        data = pool.starmap(contribution_process, zip(ssp_args, factors, type_args))
        data = np.array(data)

        df = pd.DataFrame({
            "Country Index": country_codes,
            "Country Name": country_names,
            "PM2.5": data[0],
            "Baseline Mortality": data[1],
            "Population Size (25-60)": data[2],
            "Aging (60+)": data[3],
            "Overall": np.sum(data, axis=0)
        })
        df.to_csv(f"{output_dir}/{ssp}_contrib_{type}.csv", index=False)

    else:
        factor_args = [factor] * len(ssps)
        type_args = [type] * len(ssps)
        data = pool.starmap(contribution_process, zip(ssps, factor_args, type_args))

    for i in range(4):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

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
        
        # ax.add_feature(cartopy.feature.LAND)
        # ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.3)
        ax.add_feature(cartopy.feature.BORDERS, linewidth=0.3)
    
        ax.set_title(ascii_uppercase[i], loc="left")
        if factor == "SSP":
            ax.set_title(factors[i], loc="right")
        else:
            ax.set_title(output_ssps[i], loc="right")

    handles = []
    for i in range(len(bounds) + 1): 
        color = cmap(i)
        if i == 0: 
            label = f"< {bounds[i]}"
        elif i == len(bounds): 
            label = f"> {bounds[i - 1]}"
        else: 
            label = f"{bounds[i - 1]} to {bounds[i]}"
        patch = mpatches.Patch(color=color, label=label)
        handles.append(patch)
    fig.legend(handles=handles, loc="center right", title="% Change")

    # cbar = fig.colorbar(
    #     matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
    #     ax=axes.ravel().tolist(),
    #     shrink=0.9,
    #     ticks=bounds,
    #     spacing="uniform",
    #     format="%d",
    # )
    # cbar.ax.set_ylabel(f"Percent Change")
    # if factor == "SSP":
    #     fig.suptitle(f"Contribution of different factors")
    # else:
    #     fig.suptitle(f"Contribution of {factor} under different SSPs")

    output_file = f"{output_dir}/{f'{ssp}_' if factor == 'SSP' else f'{factor}_'}contrib_{type}"
    plt.savefig(output_file + ".eps", format="eps", dpi=1200)
    plt.savefig(output_file + ".png", format="png", dpi=1200)
    plt.close(fig)


def snapshot(year):
    """Plot mortality in given year on a map"""
    # Define settings
    bounds = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
    cmap = matplotlib.cm.get_cmap("coolwarm", lut=len(bounds) + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Pre-emptively make output directory
    output_dir = f"/home/ybenp/CMIP6_Images/Mortality/map/snapshots"
    os.makedirs(output_dir, exist_ok=True)

    for i, ssp in enumerate(ssps):

        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        fig.set_size_inches(15, 8)
        # row = i // 2
        # col = i % 2
        # ax = axes[row, col]        

        mort_2040 = country_mort(pop="var", baseline=year, year=2040, ssp=ssp, countries=np.arange(0, 193))
        mort_2015 = country_mort(pop="2010", baseline=2015, year=2015, ssp=ssp, countries=np.arange(0, 193))
        print(mort_2040[35], mort_2015[35])
        data = (mort_2040 - mort_2015) / mort_2015 * 100

        countries = reader.records()
        for country in countries:
            cur_name = country.attributes["NAME_LONG"]
            if cur_name in country_conversion_dict:
                cur_name = country_conversion_dict[cur_name]
            if cur_name in country_names:
                cur_index = country_names.index(cur_name)
                if np.isnan(data[cur_index]):
                    continue
                color = findColor(bounds, cmap, data[cur_index])
                ax.add_geometries(
                    [country.geometry], ccrs.PlateCarree(), facecolor=color
                )
        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
        ax.add_feature(cartopy.feature.BORDERS, linewidth=0.5)
        ax.set_title(output_ssps[i], loc="center")
    
        handles = []
        for i in range(len(bounds) + 1): 
            color = cmap(i)
            if i == 0: 
                label = f"< {bounds[i]}"
            elif i == len(bounds): 
                label = f"> {bounds[i - 1]}"
            else: 
                label = f"{bounds[i - 1]} to {bounds[i]}"
            patch = mpatches.Patch(color=color, label=label)
            handles.append(patch)
        fig.legend(handles=handles, loc="center right", title="% Change", fontsize=8)

        # cbar = fig.colorbar(
        #     matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
        #     ax=axes.ravel().tolist(),
        #     shrink=0.9,
        #     ticks=bounds,
        #     spacing="uniform",
        #     format="%d",
        # )
        # cbar.ax.set_ylabel(f"Change in Number of Deaths")

        output_file = f"{output_dir}/{ssp}_delta.png"
        plt.savefig(output_file, format="png", dpi=1200)
        plt.close(fig)


def main():
    type = sys.argv[1]
    if (type == "snapshot"):
        # snapshot(2015)
        snapshot(2040)
        return
    factor = sys.argv[2]
    # contribution(factor="PM25")
    # contribution(factor="Baseline Mortality")
    # contribution(factor="Population")
    # contribution(factor="Population Size")
    # contribution(factor="Aging")
    if factor in ssps:
        contribution(factor="SSP", ssp=factor)
    else:
        contribution(factor, ssp=None)


if __name__ == "__main__":
    main()
