import os
import xarray as xr
from glob import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

####################################################################################################
#### CREATE WHISKER PLOT OF US, INDIA, CHINA, WORLD in 2010s, 2050s, 2090s COMBINED
####################################################################################################

# parentdir = "/glade/scratch/lamar/Data/PM2.5/Baseline_2015_Population_2015/"
parentdir = "D:/CMIP6_data/Mortality/"
output_dir = "D:/CMIP6_Images/Mortality/whisker_all_time/"
ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]  # "ssp434","ssp460","ssp534os",
years = [str(x) + '*' for x in np.arange(201, 211)]

countries = ["USA", "India", "China", "World"]

diseases = ["Allcause", "COPD", "IHD", "LC", "LRI", "Stroke", "T2D"]

years = np.arange(2015, 2101)
year_bins = [
    np.arange(0, 5),    # 2015-2019
    # np.arange(5, 15),   # 2020-2029
    # np.arange(15, 25),  # 2030-2039
    # np.arange(25, 35),  # 2040-2049
    np.arange(35, 45),  # 2050-2059
    # np.arange(45, 55),  # 2060-2069
    # np.arange(55, 65),  # 2070-2079
    # np.arange(65, 75),  # 2080-2089
    np.arange(75, 85),  # 2090-2099
    # [85]                # 2100
]
for i in range(len(year_bins)):
    year_bins[i] = [str(x + 2015) for x in year_bins[i]]
# year_bin_names = [str(x) + '0s' for x in np.arange(201, 211)]
year_bin_names = ["2010s", "2050s", "2090s"]

excludes = [
    r"ssp245/MRI-ESM2-0",
    r"ssp370/EC-Earth3-AerChem",
    r"ssp370/IPSL-CM5A2-INCA",
    r"ssp370/MPI-ESM-1-2-HAM",
    r"ssp370/NorESM2-LM",
]

for disease in diseases:

    #
    # create array
    #
    max_num_files = 1000
    idx_files = np.arange(0, max_num_files)
    num_ssps = len(ssps)
    num_year_bins = len(year_bins)
    idx_ssps = np.arange(0, num_ssps)
    idx_year_bins = np.arange(0, num_year_bins)

    var_name = ['post25'] if disease in ["Allcause", "IHD", "Stroke"] else [' Mean']

    df = [pd.DataFrame(columns=['year_bin', 'ssp', 'val'])] * 4

    for year_bin_ind in range(len(year_bins)):
        year_bin = year_bins[year_bin_ind]
        year_bin_name = year_bin_names[year_bin_ind]
        #
        # loop over SSPs
        #
        i = 0
        for ssp in ssps:
            #
            # create list of files
            #
            files = sorted(glob(parentdir + ssp + "/*" + "_" + disease + "_CountryMortalityAbsolute_GEMM.csv"))
            files = [x for x in files if
                     any(year in x for year in year_bin)                # Extract models based on year
                     and not any(exclude in x for exclude in excludes)  # Exclude models that don't go up to 2100
                     ]
            j = 0
            for file in files:
                #
                # read csv file; only extract specific variable
                #
                wk = pd.read_csv(file, usecols=var_name)
                #
                # extract regional data
                #
                val = [wk.iloc[183].values[0], wk.iloc[77].values[0], wk.iloc[35].values[0], wk.iloc[1:].sum().values[0]]

                for country_ind in range(len(countries)):

                    row = pd.Series([year_bin_name, ssp, val[country_ind]], index=df[country_ind].columns)
                    df[country_ind] = df[country_ind].append(row, ignore_index=True)

                #
                # increment counter for files
                #
                j += 1
            #
            # increment counter for SSPs
            #
            i += 1

    #
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(20, 15)
    i1 = 0
    i2 = 0
    for country_ind in range(len(countries)):

        sns.boxplot(x='year_bin', y='val', data=df[country_ind], showfliers=False, hue='ssp', ax=axes[i1, i2])
        title = disease + " " + str(var_name)[2:-2] + " " + countries[country_ind]
        axes[i1, i2].set_title(title, fontsize=14)
        axes[i1, i2].set_ylabel("Number of deaths")
        axes[i1, i2].set_xlabel("Years")

        i2 = i2 + 1
        if i2 == 2:
            i1 = i1 + 1
            i2 = 0

    # plt.show()
    output_file = output_dir + disease + "_" + str(var_name)[2:-2] + ".png"
    plt.savefig(output_file)
    plt.close(fig)
    print(f'DONE: {disease}')