import os
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from netCDF4 import Dataset
import math
import cartopy.crs as ccrs

ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585"]
# ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]
# years = [2015, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
years = np.arange(2015, 2105, 10)
pm25_path = "D:/CMIP6_data/PM2.5_annual"
latitude = np.arange(-89.75, 90.25, 0.5)
longitude = np.arange(0.25, 360.25, 0.5)


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


def get_country_mask(base_path="D:/CMIP6_data/population/national_pop/", base_file="countryFractions_2010_0.5x0.5.nc",
                     country=-1, output=False):
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

    fractionCountry = np.concatenate(
        [
            fractionCountry[:, :, len(longitude) // 2:],
            fractionCountry[:, :, : len(longitude) // 2],
        ],
        axis=2,
    )
    longitude = np.arange(0.25, 360, 0.5)
    us_fraction = fractionCountry[country]
    if not output:
        return us_fraction
    output_path = "D:/CMIP6_data/population/national_pop"
    output_file = f"{output_path}/us_mask.nc"
    ds = Dataset(output_file, "w", format="NETCDF4")
    ds.createDimension("lat", len(latitude))
    ds.createDimension("lon", len(longitude))
    lats = ds.createVariable("lat", "f4", ("lat",))
    lons = ds.createVariable("lon", "f4", ("lon",))
    fractions = ds.createVariable("us_fraction", "f4", ("lat", "lon"))
    lats[:] = latitude
    lons[:] = longitude
    fractions[:, :] = us_fraction
    lats.units = "degrees_north"
    lons.units = "degress_east"
    ds.description = "Country mask for the United States on a grid with resolution 0.5 deg x0.5 deg"
    ds.contact = "Yuhao (Ben) Pan - ybenp8104@gmail.com"
    ds.close()

    return us_fraction


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


# Get country mask
fractionCountry = get_country_mask(country=183)

# Get grid areas for area weighted mean
grid_area, tot_area = get_grid_area(fractionCountry)

# Get population for population weighted mean
pop, tot_pop = get_pop(fractionCountry)


def line():
    pm25_path = "D:/CMIP6_data/PM2.5_annual"
    sns.set_theme()
    fig, axes = plt.subplots(2)
    fig.set_size_inches(6.4, 9.6)

    for i, ssp in enumerate(ssps):
        awm_data = np.zeros(len(years))
        pwm_data = np.zeros(len(years))

        for j, year in enumerate(years):
            models = os.listdir(f"{pm25_path}/{ssp}/mmrpm2p5")
            all_conc, all_awm, all_pwm = mean(models, ssp, year, fractionCountry, grid_area, tot_area, pop, tot_pop)

            # Multi-model mean
            print(f"{ssp} {year} inter-model PWM: {np.round(all_pwm, 2)}, AWM: {np.round(all_awm, 2)}")
            awm_data[j] = all_awm
            pwm_data[j] = all_pwm

        df = pd.DataFrame({"year": years, "concentration": awm_data})
        sns.lineplot(data=df, x="year", y="concentration", label=ssp, marker="o", ax=axes[0])
        df = pd.DataFrame({"year": years, "concentration": pwm_data})
        sns.lineplot(data=df, x="year", y="concentration", label=ssp, marker="o", ax=axes[1], legend=None)
        print(f"Done: {ssp}")

    axes[0].legend(bbox_to_anchor=(1.25, 1), loc="upper right", borderaxespad=0)
    axes[0].set_ylabel("Area weighted")
    axes[1].set_ylabel("Population weighted")
    plt.suptitle("PM2.5 Concentration in the United States (μg / m^3)")
    plt.tight_layout()
    plt.show()
    # plt.savefig("D:/CMIP6_Images/PM2.5/us.png")


def map_plot(year, ssp, longitude, latitude, all_conc, fig, ax, cmap, norm, vmin=0, vmax=100, **kwargs):
    ax.coastlines(resolution='10m')
    if "extent" in kwargs:
        ax.set_extent(kwargs["extent"], ccrs.PlateCarree())
    ax.set_title(f"{year}")

    # Import shapefiles for subnational visualization
    # shp_file = "D:/CMIP6_data/country_shapefiles/gadm40_CHN_shp/gadm40_CHN_1.shp"
    # china_shapes = list(shpreader.Reader(shp_file).geometries())
    #
    # for k, shape in enumerate(china_shapes):
    #     color = cmap(norm_conc[k])
    #     ax.add_geometries([shape], ccrs.PlateCarree(), facecolor=color)

    im = ax.pcolormesh(longitude, latitude, all_conc, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
    # fig.colorbar(im, ax=ax)
    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

    # plt.show()


def map():
    # Set colors
    # colors = [(228, 245, 253), (204, 236, 249), (178, 225, 251), (149, 212, 243), (127, 191, 227), (103, 174, 220),
    #           (85, 151, 211), (69, 148, 185), (72, 158, 145), (71, 168, 114), (69, 181, 83), (114, 196, 72),
    #           (163, 208, 83), (208, 219, 91),
    #           (251, 230, 89), (248, 196, 76), (246, 162, 64), (244, 131, 53), (241, 98, 40), (238, 76, 38),
    #           (228, 56, 40), (220, 37, 41),
    #           (200, 29, 37), (180, 27, 32), (165, 24, 30)]
    # for i, color in enumerate(colors):
    #     color = tuple(x / 256 for x in color)
    #     colors[i] = color
    # cmap = ListedColormap(colors)

    bounds = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8]
    cmap = matplotlib.cm.get_cmap("jet", lut=len(bounds) + 1)

    vmin = 0
    vmax = 8
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    pm25_path = "D:/CMIP6_data/PM2.5_annual"

    for i, ssp in enumerate(ssps):

        fig, axes = plt.subplots(3, 3, subplot_kw={"projection": ccrs.PlateCarree()})
        fig.set_size_inches(12, 8)

        fig.suptitle(f"PM2.5 concentration in {ssp}")

        for j, year in enumerate(years):
            models = os.listdir(f"{pm25_path}/{ssp}/mmrpm2p5")

            all_conc, all_awm, all_pwm = mean(models, ssp, year, fractionCountry, grid_area, tot_area, pop, tot_pop)

            ax_i = j // 3
            ax_j = j % 3

            map_plot(year, ssp, longitude, latitude, all_conc, vmin=vmin, vmax=vmax, fig=fig, ax=axes[ax_i, ax_j],
                     cmap=cmap, norm=norm, extent=[-180, -60, 15, 70])

            print(f"{ssp} {year} inter-model PWM: {np.round(all_pwm, 2)}, AWM: {np.round(all_awm, 2)}")

        output_dir = f"D:/CMIP6_Images/PM2.5/map"
        os.makedirs(output_dir, exist_ok=True)
        fig.tight_layout()
        cbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes.ravel().tolist(),
            ticks=bounds,
            spacing="proportional",
            shrink=0.9
        )
        cbar.set_label("Concentration (μg / m^3)")
        # plt.show()
        plt.savefig(f"{output_dir}/us_{ssp}.png")
        plt.close(fig)


def main():
    get_country_mask(country=183, output=True)
    # line()
    # map()


if __name__ == "__main__":
    main()
