"""
Generates density data from historical pressure and temperature data
"""

import xarray as xr
import numpy as np

def get_density(parentdir="F:\\Computer Programming\\Projects\\CMIP6\\data\\"):

    # Import pressure/temperature data and calculate density
    file_name = parentdir + "tas_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc"
    data_ts = xr.open_dataset(file_name)
    file_name = parentdir + "ps_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc"
    data_ps = xr.open_dataset(file_name)
    Rgas = 287.04
    all_density = xr.ones_like(data_ps)
    all_density = data_ps['ps']/(Rgas*data_ts['tas'])

    # Select newest density data in 2014 and regrid
    newest_density = all_density.loc['2014-12-16']
    new_lat = np.arange(-90, 90, 0.5)
    new_lon = np.arange(0, 360, 0.5)
    density = newest_density.interp(lat=new_lat, lon=new_lon)

    # Fill in NaN
    density = density.interpolate_na(dim="lon", fill_value="extrapolate")
    density = density.interpolate_na(dim="lat", fill_value="extrapolate")
    density_ds = density.to_dataset(name="density")

    # Output
    density_ds.to_netcdf(path=parentdir + "density_201412.nc")

    # Plot
    # import matplotlib.pyplot as plt
    # import cartopy.crs as ccrs
    # fig, (ax1, ax2) = plt.subplots(nrows=2, subplot_kw={'projection': ccrs.PlateCarree()})
    # newest_density.plot(ax = ax1, transform = ccrs.PlateCarree())
    # density.plot(ax = ax2, transform = ccrs.PlateCarree())
    # ax1.coastlines()
    # ax2.coastlines()
    # plt.show()

get_density()