from glob import glob
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import os

# ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585"]
ssps = ["ssp370"]
# diseases = ["Allcause", "COPD", "IHD", "LC", "LRI", "Stroke", "T2D"]
diseases = ["Allcause"]

# country_codes = [183, 77, 35]
# country_names = ["United States of America", "India", "China"]
country_codes = np.arange(0, 193)
#### country_path = "/home/ybenp/CMIP6_data/population/national_pop/countryvalue_blank.csv"
country_path = "D:/CMIP6_data/population/national_pop/countryvalue_blank.csv"
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
    "ssp585": "ssp1"
}

shpfilename = shpreader.natural_earth(resolution='110m',
                                      category='cultural',
                                      name='admin_0_countries')
reader = shpreader.Reader(shpfilename)
countries = reader.records()
tmp = sorted([country.attributes['NAME_LONG'] for country in countries])


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
                f"{model}: Diff: {np.round(np.sum(data[1, i, :, :] - data[0, i, :, :]), 0)}, Percent Change: {np.round(np.sum(data[1, i, :, :] - data[0, i, :, :]) / np.sum(data[0, i, :, :]), 2)}")
            file.write(
                f"{ssp},{model},{np.sum(data[1, i, :, :] - data[0, i, :, :])},{np.sum(data[1, i, :, :] - data[0, i, :, :]) / np.sum(data[0, i, :, :])}\n")


def ssp_pop_2040_mort():

    bounds = [-10000, -5000, -4000, -3000, -2000, -1000, 0, 5000, 10000, 15000, 20000, 50000, 100000]
    cmap = matplotlib.cm.get_cmap("jet", lut=len(bounds) + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    # Define settings
    year_bins = [
        "2015",
        "2040",
    ]

    var_name = "mean"
    age_bins = ["25-60", "60-80", "80+"]
    #### parent_dir = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/Health/Baseline_Ben_2015_National/5_years"
    parentdir = "D:/CMIP6_data/Outputs/Baseline_Ben_2015_National/5_years"

    for ssp in ssps:
        pop_ssp = pop_ssp_dict[ssp]

        for disease in diseases:

            # var_name = ['post25'] if disease in ["Allcause", "IHD", "Stroke", "NCD"] else [' Mean']

            # first index is 2 for diff, 1 for 2040, 0 for 2015
            # second index represents model
            # third index represents country
            # fourth index represents age cohorts (25-60, 60-80, 80+)
            data = np.zeros((2 + 1, 10, len(country_codes), 3))

            files_2015 = sorted(
                glob(f"{parentdir}/{ssp}/*/CountryMortalityAbsolute/{disease}_{var_name}/*_2015_GEMM.csv"))
            # MODIFY BASED ON INPUT
            #### models = sorted(set([file.split("/")[-1].split("_")[2] for file in files_2015]))
            models = sorted(set([file.split("\\")[-1].split("_")[2] for file in files_2015]))

            # Add or remove models here
            models = [model for model in models if "EC-Earth3-AerChem" not in model]
            # models = [model for model in models if any(["GFDL-ESM4" == model, "MRI-ESM2-0" == model])]

            # Aggregate into age cohorts
            for year_bin_ind, year_bin in enumerate(year_bins):

                for i, model in enumerate(models):
                    files = sorted(glob(
                            f"{parentdir}/{ssp}/*/CountryMortalityAbsolute/{disease}_{var_name}/all_ages_{model}*_{year_bin}_GEMM.csv"))
                    model_means = np.zeros((len(country_codes), 3))

                    for file in files:
                        wk = pd.read_csv(file, usecols=np.arange(1, 46, 3))
                        # wk = wk.iloc[country_codes].values.flatten()
                        # print(file, np.sum(wk.iloc[country_codes].values[:, 0:7], axis=1).shape)
                        # input()
                        model_means[:, 0] += np.sum(wk.iloc[country_codes].values[:, 0:7], axis=1)
                        model_means[:, 1] += np.sum(wk.iloc[country_codes].values[:, 7:11], axis=1)
                        model_means[:, 2] += np.sum(wk.iloc[country_codes].values[:, 11:15], axis=1)
                    model_means /= len(files)
                    data[year_bin_ind, i] = model_means

            # Output inter-model differences if needed
            #### output_diff(data=data, diff_file=f"/home/ybenp/CMIP6/diff.csv", ssp=ssp, models=models)
            output_diff(data=data, diff_file=f"{parentdir}/diff.csv", ssp=ssp, models=models)

            # Take the mean
            data /= len(models)
            data[0, 0] = np.sum(data[0, 1:], axis=0)
            data[1, 0] = np.sum(data[1, 1:], axis=0)

            # Calculate difference
            # data[2, :, :] = np.divide((data[1, :, :] - data[0, :, :]), data[0, :, :], where=data[0, :, :]!=0)
            data[2] = data[1] - data[0]
            diff = data[2, 0]

            for age_bin_ind, age_bin in enumerate(age_bins):
                fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
                fig.set_size_inches(16, 12)
                # ax.add_feature(cartopy.feature.LAND)
                ax.add_feature(cartopy.feature.OCEAN)
                ax.add_feature(cartopy.feature.COASTLINE)
                ax.add_feature(cartopy.feature.BORDERS)

                countries = reader.records()
                for country in countries:
                    cur_name = country.attributes['NAME_LONG']
                    if cur_name in country_conversion_dict:
                        cur_name = country_conversion_dict[cur_name]
                    if cur_name in country_names:
                        cur_index = country_names.index(cur_name)
                        if np.isnan(diff[cur_index, age_bin_ind]):
                            continue
                        color = findColor(bounds, cmap, diff[cur_index, age_bin_ind])
                        ax.add_geometries([country.geometry], ccrs.PlateCarree(), facecolor=color)

                cbar = fig.colorbar(
                    matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
                    ax=ax,
                    shrink=0.6,
                    ticks=bounds,
                    spacing="uniform",
                    format="%d"
                )
                cbar.ax.set_ylabel("Change in Mortality")
                plt.title(f"Change in Mortality caused by {disease}_{var_name} in scenario {ssp}, age cohort {age_bin} from 2015 to 2040")

                #### outputdir = f"/home/ybenp/CMIP6_Images/Mortality/map/Baseline_Ben_2015_National/5_years/{ssp}/Pop_{pop_ssp}_var/{disease}_{var_name}"
                outputdir = f"D:/CMIP6_Images/Mortality/map/Baseline_Ben_2015_National/5_years/{ssp}/Pop_{pop_ssp}_var/{disease}_{var_name}/"
                os.makedirs(outputdir, exist_ok=True)
                output_file = f"{outputdir}/{age_bin}.png"
                plt.show()
                # plt.savefig(output_file)
                plt.close(fig)
                del fig, ax, cbar
                print(f"DONE: {disease}, {ssp}")
                print(np.mean(diff[:, age_bin_ind]))


def main():
    ssp_pop_2040_mort()


if __name__ == "__main__":
    main()






