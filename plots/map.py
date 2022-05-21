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


def const_pop_const_mort():
    cmap = matplotlib.cm.get_cmap('afmhot_r')
    # Define settings
    parentdir = "D:/CMIP6_data/Mortality/"
    outputdir = "D:/CMIP6_Images/Mortality/map/"

    year_bins = [
        np.arange(0, 1),
        np.arange(25, 26),
    ]
    for i in range(len(year_bins)):
        year_bins[i] = [str(x + 2015) for x in year_bins[i]]
    year_bin_names = ["2015", "2040"]

    excludes = [
        r"ssp245/MRI-ESM2-0",
        r"ssp370/EC-Earth3-AerChem",
        r"ssp370/IPSL-CM5A2-INCA",
        r"ssp370/MPI-ESM-1-2-HAM",
        r"ssp370/NorESM2-LM",
    ]

    for disease in diseases:

        var_name = ['post25'] if disease in ["Allcause", "IHD", "Stroke"] else [' Mean']

        for ssp in ssps:

            fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
            fig.set_size_inches(16, 12)
            # ax.add_feature(cartopy.feature.LAND)
            ax.add_feature(cartopy.feature.OCEAN)
            ax.add_feature(cartopy.feature.COASTLINE)
            ax.add_feature(cartopy.feature.BORDERS)

            data = np.zeros((3, len(country_codes)))

            for year_bin_ind in range(len(year_bins)):
                year_bin = year_bins[year_bin_ind]
                year_bin_name = year_bin_names[year_bin_ind]

                files = sorted(glob(parentdir + ssp + "/*" + "_" + disease + "_CountryMortalityAbsolute_GEMM.csv"))

                files = [x for x in files if
                         any(year in x for year in year_bin)                # Extract models based on year
                         and not any(exclude in x for exclude in excludes)  # Exclude models that don't go up to 2100
                         ]

                for file in files:
                    wk = pd.read_csv(file, usecols=var_name)
                    wk = wk.iloc[country_codes].values.flatten()
                    data[year_bin_ind] += wk
                data[year_bin_ind, :] /= len(files)

            # Calculate difference
            for i in range(len(country_names)):
                data[2, i] = (data[1, i] - data[0, i]) / data[0, i] if data[0, i] != 0 else np.nan
            diff = data[2]

            #for country_ind in country_codes:
            #    print(country_names[country_ind], data[0, country_ind], data[1, country_ind], data[2, country_ind])

            # Normalize difference to 0 to 1 for color map
            # norm_data = (diff - np.nanmin(diff)) / (np.nanmax(diff) - np.nanmin(diff))
            norm_data = (diff - (-1)) / (0.5 - (-1))

            #for country_ind in country_codes:
                #print(country_names[country_ind], norm_data[country_ind])

            # norm_data = diff
            countries = reader.records()
            for country in countries:
                cur_name = country.attributes['NAME_LONG']
                if cur_name in country_conversion_dict:
                    cur_name = country_conversion_dict[cur_name]
                if cur_name in country_names:
                    cur_index = country_names.index(cur_name)
                    if np.isnan(norm_data[cur_index]):
                        continue
                    color = cmap(norm_data[cur_index])
                    # print(cur_name, norm_data[cur_index])
                    ax.add_geometries([country.geometry], ccrs.PlateCarree(), facecolor=color,)

            cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-1, vmax=0.5),cmap=cmap), ax=ax, shrink=0.6, ticks=np.arange(-1, 0.75, 0.25))
            cbar.ax.set_yticklabels(['-100%', '-75%', '-50%', '-25%', '0%', '-25%', '50%'])
            plt.title(f"Change in Mortality caused by {disease} in scenario {ssp} from 2015 to 2040")

            output_file = outputdir + disease + "_" + ssp + "_" + str(var_name)[2:-2] + ".png"
            plt.savefig(output_file)
            plt.close(fig)
            del fig, ax, cbar, data
            print(f"DONE: {disease}, {ssp}")
            print(np.nanmax(diff))

            # plt.show()

            # df = pd.DataFrame(data, index=country_names, columns=["2010s", "2090s"])
            # print(df)


def findColor(colorbounds, colormap, num):

    for x in np.arange(1, len(colorbounds)):
        if num >= colorbounds[x]:
            continue
        else:
            return colormap.colors[x - 1]
    return colormap.colors[len(colorbounds) - 2]


def ssp_pop_2040_mort():
    cmap = matplotlib.colors.ListedColormap([
        'darkblue', 'deepskyblue',
        'azure', 'wheat', 'gold', 'orange',
        'red', 'darkred'
    ])
    colorbounds = [-10000, -5000, 0, 5000, 10000, 15000, 20000, 50000, 100000]
    assert len(colorbounds) == cmap.N + 1
    # Define settings
    year_bins = [
        "2015",
        "2040",
    ]

    var_name = "mean"
    age_bins = ["25-60", "60-80", "80+"]
    parentdir = "D:/CMIP6_data/Outputs/Baseline_Ben_2015_National/5_years"

    for ssp in ssps:
        pop_ssp = pop_ssp_dict[ssp]

        for disease in diseases:

            # var_name = ['post25'] if disease in ["Allcause", "IHD", "Stroke", "NCD"] else [' Mean']

            # first index is 2 for diff, 1 for 2040, 0 for 2015
            # third index represents age cohorts (25-60, 60-80, 80+)
            data = np.zeros((2 + 1, len(country_codes), 3))

            # Aggregate into age cohorts
            for year_bin_ind, year_bin in enumerate(year_bins):

                files_2015 = sorted(glob(f"{parentdir}/{ssp}/*/CountryMortalityAbsolute/{disease}_{var_name}/*_2015_GEMM.csv"))
                # MODIFY BASED ON INPUT
                models = sorted(set([file.split("\\")[-1].split("_")[2] for file in files_2015]))

                # Add or remove models here
                models = [model for model in models if "EC-Earth3-AerChem" not in model]
                models = [model for model in models if any(["GFDL-ESM4" == model, "MRI-ESM2-0" == model])]

                for model in models:
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
                    data[year_bin_ind] += model_means
                data[year_bin_ind] /= len(models)

            # Calculate difference
            # data[2, :, :] = np.divide((data[1, :, :] - data[0, :, :]), data[0, :, :], where=data[0, :, :]!=0)
            data[2] = data[1] - data[0]
            diff = data[2]

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
                        color = findColor(colorbounds, cmap, diff[cur_index, age_bin_ind])
                        ax.add_geometries([country.geometry], ccrs.PlateCarree(), facecolor=color)

                norm = matplotlib.colors.BoundaryNorm(colorbounds, cmap.N)
                cbar = fig.colorbar(
                    matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
                    shrink=0.6, ticks=colorbounds, spacing="uniform", format="%d")
                cbar.ax.set_ylabel("Change in Mortality")
                plt.title(f"Change in Mortality caused by {disease}_{var_name} in scenario {ssp}, age cohort {age_bin} from 2015 to 2040")

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






