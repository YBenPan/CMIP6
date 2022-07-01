import os
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy
from lib.country import get_country_names, get_regions
from lib.map import get_countries_mask, get_grid_area, get_pop
from lib.mean import mean, get_means, output_means

# ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585"]
ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]
years = [2015, 2020, 2030, 2040]
pm25_path = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/annual_0.5x0.5"
latitude = np.arange(-89.75, 90.25, 0.5)
longitude = np.arange(0.25, 360.25, 0.5)


def line(region, countries, countries_names):
    """Line plot"""
    # Get country mask
    fractionCountries = get_countries_mask(countries=countries)

    # Get grid areas for area weighted mean
    grid_area, tot_area = get_grid_area(fractionCountries)

    # Get population for population weighted mean
    pop, tot_pop = get_pop(fractionCountries)

    sns.set_theme()
    fig, axes = plt.subplots(2)
    fig.set_size_inches(6.4, 9.6)

    for i, ssp in enumerate(ssps):
        awm_data = np.zeros(len(years))
        pwm_data = np.zeros(len(years))

        for j, year in enumerate(years):
            models = os.listdir(os.path.join(pm25_path, ssp, "mmrpm2p5"))
            all_conc, all_awm, all_pwm = mean(
                models, ssp, year, fractionCountries, grid_area, tot_area, pop, tot_pop
            )

            # Multi-model mean
            print(
                f"{ssp} {year} inter-model PWM: {np.round(all_pwm, 2)}, AWM: {np.round(all_awm, 2)}"
            )
            awm_data[j] = all_awm
            pwm_data[j] = all_pwm

        df = pd.DataFrame({"year": years, "concentration": awm_data})
        sns.lineplot(
            data=df,
            x="year",
            y="concentration",
            label=ssp,
            marker="o",
            ax=axes[0],
        )
        df = pd.DataFrame({"year": years, "concentration": pwm_data})
        sns.lineplot(
            data=df,
            x="year",
            y="concentration",
            label=ssp,
            marker="o",
            ax=axes[1],
            legend=None,
        )
        print(f"Done: {ssp}")

    axes[0].legend(bbox_to_anchor=(1.25, 1), loc="upper right", borderaxespad=0)
    axes[0].set_ylabel("Area weighted")
    axes[1].set_ylabel("Population weighted")
    plt.suptitle(f"PM2.5 Concentration in {region} (μg / m^3)")
    plt.tight_layout()
    plt.show()
    # plt.savefig("/home/ybenp/CMIP6_Images/PM2.5/us.png")


def map_plot(
    year,
    ssp,
    longitude,
    latitude,
    all_conc,
    fig,
    ax,
    cmap,
    norm,
    vmin=0,
    vmax=100,
    **kwargs,
):
    """Map Plot"""
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

    im = ax.pcolormesh(
        longitude, latitude, all_conc, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm
    )
    # fig.colorbar(im, ax=ax)
    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

    # plt.show()


def map(region, countries, countries_names):
    """Driver program for map plots"""
    # Get country mask
    fractionCountries = get_countries_mask(countries=countries)

    # Get grid areas for area weighted mean
    grid_area, tot_area = get_grid_area(fractionCountries)

    # Get population for population weighted mean
    pop, tot_pop = get_pop(fractionCountries)

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

    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
    cmap = matplotlib.cm.get_cmap("jet", lut=len(bounds) + 1)

    vmin = 0
    vmax = 60
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # pm25_path = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/annual_0.5x0.5"

    for i, ssp in enumerate(ssps):

        fig, axes = plt.subplots(2, 2, subplot_kw={"projection": ccrs.PlateCarree()})
        fig.set_size_inches(18, 8)

        fig.suptitle(f"PM2.5 concentration in {ssp}")

        for j, year in enumerate(years):
            # models = os.listdir(os.path.join(pm25_path, ssp, "mmrpm2p5"))

            all_conc, all_awm, all_pwm = mean(
                ssp, year, fractionCountries, grid_area, tot_area, pop, tot_pop
            )

            ax_i = j // 2
            ax_j = j % 2
            ax = axes[ax_i, ax_j]

            map_plot(
                year,
                ssp,
                longitude,
                latitude,
                all_conc,
                vmin=vmin,
                vmax=vmax,
                fig=fig,
                ax=ax,
                cmap=cmap,
                norm=norm,
            )
            ax.add_feature(cartopy.feature.OCEAN)
            ax.add_feature(cartopy.feature.COASTLINE)
            ax.add_feature(cartopy.feature.BORDERS)

            # extent=[-180, -60, 15, 70]

            print(
                f"{ssp} {year} inter-model PWM: {np.round(all_pwm, 2)}, AWM: {np.round(all_awm, 2)}"
            )

        output_dir = "/home/ybenp/CMIP6_Images/PM2.5/map"
        os.makedirs(output_dir, exist_ok=True)
        fig.tight_layout()
        cbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes.ravel().tolist(),
            ticks=bounds,
            spacing="proportional",
            shrink=0.9,
        )
        cbar.set_label("Concentration (μg / m^3)")
        # plt.show()
        plt.savefig(f"{output_dir}/World_{ssp}.png")
        plt.close(fig)


def map_2015(countries=None):
    """Driver program for map plots"""
    # Get country mask
    fractionCountries = get_countries_mask(countries=countries)

    # Get grid areas for area weighted mean
    grid_area, tot_area = get_grid_area(fractionCountries)

    # Get population for population weighted mean
    pop, tot_pop = get_pop(fractionCountries)
    
    sns.set()

    bounds = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60]
    cmap = matplotlib.cm.get_cmap("jet", lut=len(bounds) + 1)

    vmin = 0
    vmax = 60
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # pm25_path = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/annual_0.5x0.5"
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(18, 8)
    fig.suptitle(f"PM2.5 concentration in 2015")
    year = 2015

    conc = []

    for i, ssp in enumerate(ssps):

        all_conc, all_awm, all_pwm = mean(
            ssp, year, fractionCountries, grid_area, tot_area, pop, tot_pop
        )
        conc.append(all_conc)
    
    conc = np.mean(conc, axis=0)

    ax.pcolormesh(
        longitude, latitude, conc, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm
    )
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS)
    ax.coastlines(resolution="10m")

    output_dir = "/home/ybenp/CMIP6_Images/PM2.5/map"
    os.makedirs(output_dir, exist_ok=True)
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        ticks=bounds,
        spacing="proportional",
        shrink=0.9,
    )
    cbar.set_label("Concentration (μg / m^3)")
    plt.savefig(f"{output_dir}/World_2015.png")
    plt.close(fig)


def map_delta(countries=None):
    """Driver program for delta PM2.5 map plots"""
    # Get country mask
    fractionCountries = get_countries_mask(countries=countries)

    # Get grid areas for area weighted mean
    grid_area, tot_area = get_grid_area(fractionCountries)

    # Get population for population weighted mean
    pop, tot_pop = get_pop(fractionCountries)

    fig, axes = plt.subplots(2, 2, subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(18, 8)
    fig.suptitle(f"Change in PM2.5 concentration")

    bounds = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
    cmap = matplotlib.cm.get_cmap("jet", lut=len(bounds) + 1)

    vmin = -100
    vmax = 300
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    for i, ssp in enumerate(ssps):
        
        conc_2015, awm_2015, pwm_2015 = mean(
            ssp, "2015", fractionCountries, grid_area, tot_area, pop, tot_pop
        )
        conc_2040, awm_2040, pwm_2040 = mean(
            ssp, "2040", fractionCountries, grid_area, tot_area, pop, tot_pop
        )

        conc = (conc_2040 - conc_2015) / conc_2015 * 100
        
        ax_i = i // 2
        ax_j = i % 2
        ax = axes[ax_i, ax_j]

        ax.pcolormesh(
            longitude, latitude, conc, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm
        )
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.BORDERS)
        ax.set_title(ssp)

    output_dir = "/home/ybenp/CMIP6_Images/PM2.5/map"
    os.makedirs(output_dir, exist_ok=True)
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        ticks=bounds,
        spacing="proportional",
        shrink=0.9,
    )
    cbar.set_label("Percent Change in Concentration")
    plt.savefig(f"{output_dir}/Delta.png")
    plt.close(fig)
        
        
def main():
    # line()
    # map()
    # Get countries and regions
    country_dict = get_country_names()
    regions, region_countries, region_countries_names = get_regions()
    output_means(regions, region_countries, region_countries)
    map_delta()


if __name__ == "__main__":
    main()
