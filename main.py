# coding: utf-8


import os
import matplotlib
import xarray as xr
from glob import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
from datetime import datetime
from density import get_density

# Paths Settings
mmrpm2p5_path = "D:/PM2.5"
density_path = "F:/Computer Programming/Projects/CMIP6/data/"
output_image_path = "D:/PM2.5_images_avg"

def year_helper(date):
    if isinstance(date, np.datetime64):
        date = pd.to_datetime(date)
    return date.year

def plot_combined(): 
    
    fig, axes = plt.subplots(nrows=2, subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(12, 8)

    # Plot start of the range
    im = axes[0].pcolormesh(dsi.lon.values, dsi.lat.values, avg_first_10, vmin = 0, vmax = max_conc, cmap="magma_r")
    start_year = year_helper(dsi.concpm2p5.time.values[0])
    end_year = year_helper(dsi.concpm2p5.time.values[9])
    axes[0].coastlines()
    axes[0].set_title(
        f"{ssp}, model {model}, realization {real}\n"
        f"Average of PM2.5 Concentration from {start_year} to {end_year}"
    )

    # Plot end of the range
    im = axes[1].pcolormesh(dsi.lon.values, dsi.lat.values, avg_last_10, vmin = 0, vmax = max_conc, cmap="magma_r")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Concentration of PM2.5 in kg $m^{-3}$")
    end_year = year_helper(dsi.concpm2p5.time.values[-1])
    start_year = year_helper(dsi.concpm2p5.time.values[-10])
    axes[1].coastlines()
    axes[1].set_title(
        f"Average of PM2.5 Concentration from {start_year} to {end_year}"
    )
    output_image_file_name = "combined"      
    plt.show()
    # plt.savefig(output_image_dir + output_image_file_name)
    # plt.close(fig)

def plot_start():

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(12, 8)
    im = ax.pcolormesh(dsi.lon.values, dsi.lat.values, avg_first_10, cmap="magma_r")
    start_year = year_helper(dsi.concpm2p5.time.values[0])
    end_year = year_helper(dsi.concpm2p5.time.values[9])
    ax.coastlines()
    ax.set_title(
        f"{ssp}, model {model}, realization {real}\n"
        f"Average of PM2.5 Concentration from {start_year} to {end_year}"
    )
    fig.colorbar(im, ax=ax, label="Concentration of PM2.5 in kg $m^{-3}$", shrink=0.6)
    output_image_file_name = f"{start_year}_{end_year}"
    plt.show()
    # plt.savefig(output_image_dir + output_image_file_name)
    # plt.close(fig)

def plot_end(): 
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(12, 8)
    im = ax.pcolormesh(dsi.lon.values, dsi.lat.values, avg_last_10, cmap="magma_r")
    end_year = year_helper(dsi.concpm2p5.time.values[-1])
    start_year = year_helper(dsi.concpm2p5.time.values[-10])
    ax.coastlines()
    ax.set_title(
        f"{ssp}, model {model}, realization {real}\n"
        f"Average of PM2.5 Concentration from {start_year} to {end_year}"
    )
    fig.colorbar(im, ax=ax, label="Concentration of PM2.5 in kg $m^{-3}$", shrink=0.6)
    output_image_file_name = f"{start_year}_{end_year}"
    plt.show()
    # plt.savefig(output_image_dir + output_image_file_name)
    # plt.close(fig)

def plot_diff(): 
    diff = avg_last_10 - avg_first_10
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(12, 8)
    max_diff = abs(np.max(diff))
    min_diff = abs(np.min(diff))
    im = ax.pcolormesh(dsi.lon.values, dsi.lat.values, diff, vmin=-max(max_diff, min_diff), vmax=max(max_diff, min_diff), cmap="seismic")
    end_year = year_helper(dsi.concpm2p5.time.values[-1])
    start_year = year_helper(dsi.concpm2p5.time.values[0])
    ax.coastlines()
    ax.set_title(
        f"{ssp}, model {model}, realization {real}\n"
        f"Change in PM2.5 Concentration from {start_year} to {end_year}"
    )
    fig.colorbar(im, ax=ax, label="Change in Concentration of PM2.5 in kg $m^{-3}$", shrink=0.6)
    output_image_file_name = f"diff"
    # plt.show()
    plt.savefig(output_image_dir + output_image_file_name)
    plt.close(fig)

# Generate density data for conversion
# get_density(density_path)

# Import density data and generate matrix for calculation
density_ds = xr.open_mfdataset(density_path + "density_201412.nc")
density_value = density_ds.density.values
density_dynamic = np.broadcast_to(
    density_ds.density.values, (86,) + density_ds.density.values.shape
)

ssps = os.listdir(mmrpm2p5_path)
for ssp in ssps:
    if not "ssp" in ssp:
        continue
    # Loop over models
    models = os.listdir(f"{mmrpm2p5_path}/{ssp}/mmrpm2p5")
    for model in models:

        # Loop over realization
        reals = os.listdir(f"{mmrpm2p5_path}/{ssp}/mmrpm2p5/{model}")
        for real in reals:

            # Loop over glabels
            glabel = os.listdir(f"{mmrpm2p5_path}/{ssp}/mmrpm2p5/{model}/{real}")
            data_dir = f"{mmrpm2p5_path}/{ssp}/mmrpm2p5/{model}/{real}/{glabel[0]}"

            files = sorted(glob(data_dir + "/*nc"))

            # resolve concat_dim error in some MRI-ESM2-0 models
            if (ssp == "ssp126" or ssp == "ssp585") and model == "MRI-ESM2-0":
                files = files[0:9]
            
            # Initialize output directories
            output_image_dir = (
                f"{output_image_path}/{ssp}/mmrpm2p5/{model}/{real}/{glabel[0]}/"
            )
            if not os.path.isdir(output_image_dir):
                os.makedirs(output_image_dir)      

            ds = xr.open_mfdataset(files, concat_dim="time")

            # Annualize
            annual_ds = ds.resample(time="1YS").mean()
            
            # Regrid
            new_lat = np.arange(-90, 90, 0.5)
            new_lon = np.arange(0, 360, 0.5)
            dsi = annual_ds.interp(lat=new_lat, lon=new_lon)

            # Fill in missing values
            dsi = dsi.interpolate_na(dim="lon", fill_value="extrapolate")
            dsi = dsi.interpolate_na(dim="lat", fill_value="extrapolate")

            # Concentration = MMR * Density
            density_matrix = density_dynamic[: len(dsi.time)]
            dsi.mmrpm2p5.values *= density_matrix
            dsi = dsi.rename({"mmrpm2p5": "concpm2p5"})

            avg_first_10 = np.mean(dsi.concpm2p5.values[:10], axis=0)[0]
            avg_last_10 = np.mean(dsi.concpm2p5.values[-10:], axis=0)[0]
            max_conc = max(np.max(avg_first_10), np.max(avg_last_10))

            # plot_combined()
            # plot_start()
            # plot_end()
            plot_diff()
        
            print(
                f"{datetime.now()} DONE: {ssp}, model {model}, realization {real}"
            )
            # plt.tight_layout()
            # plt.close(fig)
            