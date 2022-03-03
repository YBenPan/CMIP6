from glob import glob
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import math

parentdir = "D:/CMIP6_data/Mortality/"
outputdir = "D:/CMIP6_Images/Mortality/map/"

ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]
# ssps = ["ssp370"]
# diseases = ["Allcause", "COPD", "IHD", "LC", "LRI", "Stroke", "T2D"]
diseases = ["Allcause"]
year_bins = [
    np.arange(0, 5),
    np.arange(75, 85),
]
for i in range(len(year_bins)):
    year_bins[i] = [str(x + 2015) for x in year_bins[i]]
year_bin_names = ["2010s", "2090s"]

excludes = [
    r"ssp245/MRI-ESM2-0",
    r"ssp370/EC-Earth3-AerChem",
    r"ssp370/IPSL-CM5A2-INCA",
    r"ssp370/MPI-ESM-1-2-HAM",
    r"ssp370/NorESM2-LM",
]

# country_codes = [183, 77, 35]
# country_names = ["United States of America", "India", "China"]
country_codes = np.arange(0, 193)
country_path = "F:/Computer Programming/Projects/CMIP6/data/population/countryvalue_blank.csv"
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

shpfilename = shpreader.natural_earth(resolution='110m',
                                      category='cultural',
                                      name='admin_0_countries')
reader = shpreader.Reader(shpfilename)
countries = reader.records()
tmp = sorted([country.attributes['NAME_LONG'] for country in countries])

cmap = matplotlib.cm.get_cmap('afmhot_r')

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
            data[2, i] = (data[1, i] - data[0, i])
        diff = data[2]

        #for country_ind in country_codes:
        #    print(country_names[country_ind], data[0, country_ind], data[1, country_ind], data[2, country_ind])

        # Normalize difference to 0 to 1 for color map
        # norm_data = (diff - np.nanmin(diff)) / (np.nanmax(diff) - np.nanmin(diff))
        norm_data = (diff - (-1e6)) / (4e5 - (-1e6))

        for country_ind in country_codes:
            print(country_names[country_ind], diff[country_ind])

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

        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-1e6, vmax=4e5),cmap=cmap), ax=ax, shrink=0.6, ticks=np.arange(-1e6, 6e5, 2e5))
        cbar.ax.set_yticklabels([str(x) for x in np.arange(-1e6, 6e5, 2e5)])
        plt.title(f"Change in Mortality caused by {disease} in scenario {ssp} from 2010s to 2090s")

        output_file = outputdir + disease + "_" + ssp + "_" + str(var_name)[2:-2] + ".png"
        plt.show()
        # plt.savefig(output_file)
        plt.close(fig)
        del fig, ax, cbar, data
        print(f"DONE: {disease}, {ssp}")
        print(np.nanmax(diff))


        # df = pd.DataFrame(data, index=country_names, columns=["2010s", "2090s"])
        # print(df)










