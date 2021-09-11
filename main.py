# coding: utf-8


import os
import xarray as xr
from glob import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import datetime
from density import get_density

# Paths Settings
mmrpm2p5_path = "D:/PM2.5"
density_path = "F:/Computer Programming/Projects/CMIP6/data/"
output_image_path = "D:/PM2.5_images"

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

            output_image_dir = (
                f"{output_image_path}/{ssp}/mmrpm2p5/{model}/{real}/{glabel[0]}/"
            )
            output_image_file_name = "result"
            if not os.path.isdir(output_image_dir):
                os.makedirs(output_image_dir)

            # Plot last datapoint in time and save the figure
            fig, axis = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
            dsi.concpm2p5.isel(time=len(dsi.concpm2p5.time) - 1).plot(
                ax=axis,
                transform=ccrs.PlateCarree(),
                cbar_kwargs={"label": "Concentration of PM2.5 in kg $m^{-3}$"},
            )
            fig.set_size_inches(12, 8)
            axis.coastlines()
            output_date = dsi.concpm2p5.time.values[-1]
            plt.title(f"{ssp}, model {model}, realization {real} on {output_date}")
            plt.savefig(output_image_dir + output_image_file_name)
            plt.close(fig)
            print(
                f"{datetime.datetime.now()} DONE: {ssp}, model {model}, realization {real}"
            )
            # plt.show()
