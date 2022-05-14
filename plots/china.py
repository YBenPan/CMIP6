from glob import glob
import numpy as np
import seaborn
import os
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
from netCDF4 import Dataset

ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585"]
pm25_path = "D:/CMIP6_data/PM2.5_annual"
country_fraction_path = "D:/CMIP6_data/population/national_pop"
country_fraction_file = f"{country_fraction_path}/countryFractions_2010_0.5x0.5.nc"

# Import country fraction
f1 = Dataset(country_fraction_file, "r")
fractionCountry = f1.variables["fractionCountry"][
    :, :, :
]  # countryIndex, latitude, longitude
latitude = f1.variables["latitude"][:]
longitude = f1.variables["longitude"][:]
fractionCountry[fractionCountry < 0.0] = 0.0
fractionCountry[fractionCountry > 1.0] = 1.0
# Change Longitude from -180 to 180 to 0 to 360 for ease of computation
fractionCountry = np.concatenate(
    [
        fractionCountry[:, :, len(longitude) // 2 :],
        fractionCountry[:, :, : len(longitude) // 2],
    ],
    axis=2,
)
longitude = np.arange(0.25, 360, 0.5)

# Import China fraction
china_fraction_file = "D:/CMIP6_data/fraction/china_state_fraction_0.5x0.5.nc"
f1 = Dataset(china_fraction_file, "r")
fractionState = f1.variables["fractionState"][
    :, :, :
]  # state, latitude, longitude
states = f1.variables["state"][:]
f1.close()
fractionState[fractionState < 0.0] = 0.0
fractionState[fractionState > 1.0] = 1.0
# Change Longitude from -180 to 180 to 0 to 360 for ease of computation
fractionState = np.concatenate(
    [
        fractionState[:, :, len(longitude) // 2 :],
        fractionState[:, :, : len(longitude) // 2],
    ],
    axis=2,
)

# Settings for China
country = 35
area = np.sum(fractionCountry[country])


# Set years
years = [2015]

# Set colors
colors = [(228, 245, 253), (204,236,249), (178,225,251), (149,212,243), (127,191,227), (103,174,220),
          (85,151,211), (69,148,185), (72,158,145), (71,168,114), (69,181,83), (114,196,72), (163,208,83), (208,219,91),
          (251,230,89), (248,196,76), (246,162,64), (244,131,53), (241,98,40), (238,76,38), (228,56,40), (220,37,41),
          (200,29,37), (180,27,32), (165,24,30)]
for i, color in enumerate(colors):
    color = tuple(x / 256 for x in color)
    colors[i] = color
cmap = ListedColormap(colors)
norm = matplotlib.colors.BoundaryNorm(np.arange(0, 104, 4), cmap.N)

def mean():
    print("MIROC, GISS excluded")
    for i, ssp in enumerate(ssps):

        for j, year in enumerate(years):

            # Compute mean PM2.5 concentration of all models
            files = sorted(glob(f"{pm25_path}/{ssp}/mmrpm2p5/*/*/annual_avg_{year}.nc"))
            all_conc = []
            for file in files:
                if "EC-Earth3" in file:
                    continue # Outlier: extremely large data
                if "MIROC" in file or "GISS" in file:
                    continue # Test skip
                wk = Dataset(file, "r")
                conc = wk["concpm2p5"][:]
                # country_conc = conc * fractionCountry[country] * (10 ** 9) # Apply mask to concentration array
                state_means = np.zeros(len(states))
                for k, state in enumerate(states):
                    state_conc = conc * fractionState[k] * (10 ** 9)
                    state_area = np.sum(fractionState[k])
                    state_means[k] = np.sum(state_conc) / state_area
                all_conc.append(state_means)

            all_conc = np.mean(all_conc, axis=0)

            norm_conc = all_conc / 100

            # cmap = matplotlib.cm.get_cmap("Spectral_r")
            # color = cmap(norm_conc)

            fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
            fig.set_size_inches(12, 8)
            ax.coastlines(resolution='10m')
            ax.set_extent([65, 140, 10, 55], ccrs.PlateCarree())
            ax.set_title(f"{year} {ssp}")

            shp_file = "D:/CMIP6_data/country_shapefiles/gadm40_CHN_shp/gadm40_CHN_1.shp"
            china_shapes = list(shpreader.Reader(shp_file).geometries())

            for k, shape in enumerate(china_shapes):
                color = cmap(norm_conc[k])
                ax.add_geometries([shape], ccrs.PlateCarree(), facecolor=color)

            # im = ax.pcolormesh(longitude, latitude, all_conc, vmin=0, vmax=100, cmap=cmap)
            # fig.colorbar(im, ax=ax)
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

            plt.show()
            plt.close(fig)
            print(f"{ssp} {year} mean: {np.mean(all_conc)}")










def main():
    mean()


if __name__ == "__main__":
    main()