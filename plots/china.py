from glob import glob
import numpy as np
import os
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
from netCDF4 import Dataset
from map import findColor
import math

# ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585"]
ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]
pm25_path = "D:/CMIP6_data/PM2.5_annual"
country_fraction_path = "D:/CMIP6_data/population/national_pop"
country_fraction_file = f"{country_fraction_path}/countryFractions_2010_0.5x0.5.nc"


def get_country_mask(base_path="D:/CMIP6_data/population/national_pop/", base_file="countryFractions_2010_0.5x0.5.nc", country=-1):
    # If no country is supplied, then return uniform mask
    if country == -1:
        return np.ones((360, 720))

    f1 = Dataset(base_path + base_file, "r")
    fractionCountry = f1.variables["fractionCountry"][
                      :, :, :
                      ]  # countryIndex, latitude, longitude
    latitude = f1.variables["latitude"][:]
    longitude = f1.variables["longitude"][:]
    f1.close()

    fractionCountry[fractionCountry < 0.0] = 0.0
    fractionCountry[fractionCountry > 1.0] = 0.0

    # Change Longitude from -180 to 180 to 0 to 360 for ease of computation
    fractionCountry = np.concatenate(
        [
            fractionCountry[:, :, len(longitude) // 2:],
            fractionCountry[:, :, : len(longitude) // 2],
        ],
        axis=2,
    )

    return fractionCountry[country]


def get_grid_area(fractionCountry=np.ones((360, 720))):
    lon_start = -179.75
    lat_start = -89.75
    earth_radius2 = 6371 ** 2
    deg2rad = math.pi / 180.0
    dx = 0.5
    dy = 0.5

    grid_areas = np.zeros((int(180 / dy), int(360 / dx)))
    for i, lat in enumerate(np.arange(lat_start, 90, dy)):
        grid_areas[i] = earth_radius2 * math.cos(lat * deg2rad) * (dx * deg2rad) * (dy * deg2rad)
    grid_areas = grid_areas * fractionCountry
    tot_area = np.sum(grid_areas)
    return grid_areas, tot_area


def get_pop(fractionCountry=np.ones((360, 720))):
    pop_path = "D:/CMIP6_data/population/gridded_pop/ssp1"
    pop_file = f"{pop_path}/ssp1_tot_2020.nc"
    f1 = Dataset(pop_file, "r")
    pop = f1["population"][:] * fractionCountry
    tot_pop = np.sum(pop)
    f1.close()
    return pop, tot_pop


latitude = np.arange(-89.75, 90.25, 0.5)
longitude = np.arange(0.25, 360, 0.5)

# Get China mask
fractionCountry = get_country_mask(country=35)

# Get grid areas for area weighted mean
grid_area, tot_area = get_grid_area(fractionCountry)

# Get population for population weighted mean
pop, tot_pop = get_pop(fractionCountry)

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
        fractionState[:, :, len(longitude) // 2:],
        fractionState[:, :, : len(longitude) // 2],
    ],
    axis=2,
)

# Set years
# years = [2015, 2030, 2040]
years = [2015]

pop_ssp_dict = {
    "ssp119": "ssp1",
    "ssp126": "ssp1",
    "ssp245": "ssp2",
    "ssp370": "ssp3",
    "ssp434": "ssp2",
    "ssp460": "ssp2",
    "ssp585": "ssp1"
}


def mean(models, ssp, year, fractionCountry, grid_area, tot_area, pop, tot_pop):
    all_conc = []
    all_awm = []
    all_pwm = []

    for model in models:
        # Outlier: extremely large data
        if "EC-Earth3" in model:
            continue
        if "IPSL" in model or "MPI" in model:
            continue
        # Skip models that do not include natural PM2.5 sources (anthropogenic only)
        # if model not in ["GFDL-ESM4", "MRI-ESM2-0"]:
        #     continue

        # Compute mean PM2.5 concentration of all realizations
        files = sorted(glob(f"{pm25_path}/{ssp}/mmrpm2p5/{model}/*/annual_avg_{year}.nc"))
        if len(files) == 0:
            continue
        model_conc = []
        model_awm = []
        model_pwm = []

        for file in files:
            # Import concentration NC file
            wk = Dataset(file, "r")
            conc = wk["concpm2p5"][:]

            # Calculate concentration and means
            country_conc = conc * fractionCountry * (10 ** 9)  # Apply mask to concentration array
            area_weighted_mean = np.sum(grid_area * country_conc) / tot_area
            pop_weighted_mean = np.sum(pop * country_conc) / tot_pop

            # Compute mean concentration of every province
            # state_means = np.zeros(len(states))
            # for k, state in enumerate(states):
            #     state_conc = conc * fractionState[k] * (10 ** 9)
            #     state_area = np.sum(fractionState[k])
            #     state_means[k] = np.sum(state_conc) / state_area
            # all_conc.append(state_means)

            model_conc.append(country_conc)
            model_awm.append(area_weighted_mean)
            model_pwm.append(pop_weighted_mean)

            # real = file.split("mmrpm2p5/")[1].split("\\annual_avg")[0]

        model_conc = np.mean(model_conc, axis=0)
        model_awm = np.mean(model_awm, axis=0)
        model_pwm = np.mean(model_pwm, axis=0)
        all_conc.append(model_conc)
        all_awm.append(model_awm)
        all_pwm.append(model_pwm)
        print(f"{model}: PWM: {np.round(model_pwm, 2)}, AWM: {np.round(model_awm, 2)}")
    all_conc = np.mean(all_conc, axis=0)
    all_awm = np.mean(all_awm, axis=0)
    all_pwm = np.mean(all_pwm, axis=0)
    return all_conc, all_awm, all_pwm


def pm25_mean():
    # Set colors
    colors = [(228, 245, 253), (204, 236, 249), (178, 225, 251), (149, 212, 243), (127, 191, 227), (103, 174, 220),
              (85, 151, 211), (69, 148, 185), (72, 158, 145), (71, 168, 114), (69, 181, 83), (114, 196, 72),
              (163, 208, 83), (208, 219, 91),
              (251, 230, 89), (248, 196, 76), (246, 162, 64), (244, 131, 53), (241, 98, 40), (238, 76, 38),
              (228, 56, 40), (220, 37, 41),
              (200, 29, 37), (180, 27, 32), (165, 24, 30)]
    for i, color in enumerate(colors):
        color = tuple(x / 256 for x in color)
        colors[i] = color
    cmap = ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(np.arange(0, 104, 4), cmap.N)

    for i, ssp in enumerate(ssps):

        for j, year in enumerate(years):

            models = os.listdir(f"{pm25_path}/{ssp}/mmrpm2p5")

            all_conc, all_awm, all_pwm = mean(models, ssp, year)

            fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
            fig.set_size_inches(12, 8)
            ax.coastlines(resolution='10m')
            ax.set_extent([65, 140, 10, 55], ccrs.PlateCarree())
            ax.set_title(f"{year} {ssp}")

            # Import shapefiles for subnational visualization
            # shp_file = "D:/CMIP6_data/country_shapefiles/gadm40_CHN_shp/gadm40_CHN_1.shp"
            # china_shapes = list(shpreader.Reader(shp_file).geometries())
            #
            # for k, shape in enumerate(china_shapes):
            #     color = cmap(norm_conc[k])
            #     ax.add_geometries([shape], ccrs.PlateCarree(), facecolor=color)

            im = ax.pcolormesh(longitude, latitude, all_conc, vmin=0, vmax=100, cmap=cmap)
            fig.colorbar(im, ax=ax)
            # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

            # plt.show()
            plt.close(fig)
            print(f"{ssp} {year} inter-model PWM: {np.round(all_pwm, 2)}, AWM: {np.round(all_awm, 2)}")


def mortality():
    mort_dir = "D:/CMIP6_data/Outputs/Baseline_Ben_2015_National/5_years"
    cmap = matplotlib.colors.ListedColormap([
        "antiquewhite", "moccasin", "pink", "palevioletred", "indianred", "brown", "maroon"
    ])
    colorbounds = [0, 50, 100, 150, 200, 250, 300, 350]

    fig, axes = plt.subplots(len(ssps), len(years), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(16, 12)
    for i, ssp in enumerate(ssps):
        pop_ssp = pop_ssp_dict[ssp]
        for j, year in enumerate(years):
            files = sorted(glob(f"{mort_dir}/{ssp}/*/MortalityAbsolute/Allcause_mean/*{year}_GEMM.nc"))
            all_deaths = []
            for file in files:
                if "EC-Earth3" in file:
                    continue # Outlier: extremely large data
                if "MIROC" in file or "GISS" in file:
                    continue # Test skip
                wk = Dataset(file, "r")
                deaths = wk["deaths__mean"]
                state_deaths = np.zeros(len(states))
                for k, state in enumerate(states):
                    state_deaths[k] = np.sum(deaths * fractionState[k])
                all_deaths.append(state_deaths)
            all_deaths = np.mean(all_deaths, axis=0)
            all_deaths /= 1000

            if len(ssps) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            ax.coastlines(resolution="10m")
            ax.set_extent([65, 140, 10, 55], ccrs.PlateCarree())
            ax.set_title(f"{ssp}_{year}")

            # Import China level 1 shapefile
            shp_file = "D:/CMIP6_data/country_shapefiles/gadm40_CHN_shp/gadm40_CHN_1.shp"
            china_shapes = list(shpreader.Reader(shp_file).geometries())

            for k, shape in enumerate(china_shapes):
                color = findColor(colorbounds, cmap, all_deaths[k])
                ax.add_geometries([shape], ccrs.PlateCarree(), facecolor=color)

    norm = matplotlib.colors.BoundaryNorm(colorbounds, cmap.N)
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes.ravel().tolist(),
        shrink=0.6, ticks=colorbounds, spacing="uniform", format="%d")
    cbar.ax.set_ylabel("PM2.5-related mortality (thousand)")

    output_dir = f"D:/CMIP6_Images/Mortality/China/Baseline_Ben_2015_National/5_years/{ssp}/Pop_{pop_ssp}_var/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/deaths_excluded.png"
    plt.show()
    # plt.savefig(output_file)
    plt.close(fig)
    del fig, ax, cbar
    print(f"Done: {ssp}")


def main():
    # get_pop(fractionCountry[country])
    # get_grid_area(fractionCountry[country])
    pm25_mean()
    # mortality()


if __name__ == "__main__":
    main()