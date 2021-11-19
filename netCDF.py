# coding: utf-8


import os
import matplotlib
import xarray as xr
from glob import glob
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# from density import get_density
import multiprocessing

# Paths Settings
mmrpm2p5_path = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/"
density_path = "/home/ybenp/CMIP6_src-master/"
output_file_path = "/home/ybenp/CMIP6_src-master/annual"

os.environ["OMP_NUM_THREADS"] = "1"


def year_helper(date):
    if isinstance(date, np.datetime64):
        date = pd.to_datetime(date)
    return date.year


# Generate density data for conversion
# get_density(density_path)

# Import density data and generate matrix for calculation
density = xr.open_mfdataset(density_path + "density_201412.nc")
# print(density.shape())
# density
# density_dynamic = np.broadcast_to(
#    density_ds.density.values, (86,) + density_ds.density.values.shape
# )


def process(t):
    ssp = t[0]
    model = t[1]
    real = t[2]

    # Loop over glabels
    glabel = os.listdir(f"{mmrpm2p5_path}/{ssp}/mmrpm2p5/{model}/{real}")
    data_dir = f"{mmrpm2p5_path}/{ssp}/mmrpm2p5/{model}/{real}/{glabel[0]}"

    print(f"{datetime.now()} Processing {data_dir}")
    files = sorted(glob(data_dir + "/*nc"))

    # resolve concat_dim error in some MRI-ESM2-0 models
    if (ssp == "ssp126" or ssp == "ssp585") and model == "MRI-ESM2-0":
        files = files[0:9]

    # Initialize output directories
    output_file_dir = f"{output_file_path}/{ssp}/mmrpm2p5/{model}/{real}/"
    if not os.path.isdir(output_file_dir):
        os.makedirs(output_file_dir)

    ds = xr.open_mfdataset(files, concat_dim="time")

    # Annualize
    annual_ds = ds.resample(time="1YS").mean()

    # Regrid
    new_lat = np.arange(-90, 90, 0.5)
    new_lon = np.arange(0, 360, 0.5)
    dsi = annual_ds.interp(lat=new_lat, lon=new_lon).squeeze()

    # Fill in missing values
    dsi = dsi.interpolate_na(dim="lon", fill_value="extrapolate")
    dsi = dsi.interpolate_na(dim="lat", fill_value="extrapolate")

    # Concentration = MMR * Density
    for n in range(0, len(dsi.time)):
        # density_matrix = density_dynamic[: len(dsi.time)]
        dsi["mmrpm2p5"][n, :, :].values *= density["density"][0, :, :].values
    dsi = dsi.rename({"mmrpm2p5": "concpm2p5"})

    # write netcdf file with annual averages

    var = dsi["concpm2p5"][0:10, :, :].mean(axis=0)
    var.to_netcdf(output_file_dir + "annual_avg_2015-2025.nc")
    var = dsi["concpm2p5"][10:20, :, :].mean(axis=0)
    var.to_netcdf(output_file_dir + "annual_avg_2025-2035.nc")
    var = dsi["concpm2p5"][20:30, :, :].mean(axis=0)
    var.to_netcdf(output_file_dir + "annual_avg_2035-2045.nc")
    var = dsi["concpm2p5"][30:40, :, :].mean(axis=0)
    var.to_netcdf(output_file_dir + "annual_avg_2045-2055.nc")
    var = dsi["concpm2p5"][40:50, :, :].mean(axis=0)
    var.to_netcdf(output_file_dir + "annual_avg_2055-2065.nc")
    var = dsi["concpm2p5"][50:60, :, :].mean(axis=0)
    var.to_netcdf(output_file_dir + "annual_avg_2065-2075.nc")
    var = dsi["concpm2p5"][60:70, :, :].mean(axis=0)
    var.to_netcdf(output_file_dir + "annual_avg_2075-2085.nc")
    var = dsi["concpm2p5"][70:80, :, :].mean(axis=0)
    var.to_netcdf(output_file_dir + "annual_avg_2085-2095.nc")
    var = dsi["concpm2p5"][-10:, :, :].mean(axis=0)
    var.to_netcdf(output_file_dir + "annual_avg_2090-2100.nc")


def main():
    global mmrpm2p5_path, density_path, output_file_path
    ssps = os.listdir(mmrpm2p5_path)
    for ssp in ssps:
        if not "ssp" in ssp:
            continue
        # Loop over models
        models = os.listdir(f"{mmrpm2p5_path}/{ssp}/mmrpm2p5")
        for model in models:
            # Use multiprocessing to loop over realization
            reals = os.listdir(f"{mmrpm2p5_path}/{ssp}/mmrpm2p5/{model}")
            pool = multiprocessing.Pool(8)
            pool.map(process, [(ssp, model, real) for real in reals])
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
