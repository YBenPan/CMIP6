from netCDF4 import Dataset
import numpy as np
import math
import os
from lib.helper import pop_ssp_dict


def get_countries_mask(
    base_path="/home/ybenp/CMIP6_data/population/national_pop/",
    base_file="countryFractions_2010_0.5x0.5.nc",
    countries=None,
    output=False,
):
    """Return a mask of the input countries"""
    # If no country is supplied, then return uniform mask
    if countries == None or countries == [-1]:
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
            fractionCountry[:, :, len(longitude) // 2 :],
            fractionCountry[:, :, : len(longitude) // 2],
        ],
        axis=2,
    )
    longitude = np.arange(0.25, 360, 0.5)

    # Add country fractions one by one
    fraction = np.zeros((len(latitude), len(longitude)))
    for country in countries:
        fraction += fractionCountry[country]
    if not output:
        return fraction

    # Output US fraction if required. Not part of main program
    us_fraction = fraction
    output_path = "/home/ybenp/CMIP6_data/population/national_pop"
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
    ds.description = (
        "Country mask for the United States on a grid with resolution 0.5 deg x0.5 deg"
    )
    ds.contact = "Yuhao (Ben) Pan - ybenp8104@gmail.com"
    ds.close()
    return us_fraction


def get_grid_area(fractionCountries=np.ones((360, 720))):
    """Return the area of the grids of a mask and its total area"""
    lon_start = -179.75
    lat_start = -89.75
    earth_radius2 = 6371**2
    deg2rad = math.pi / 180.0
    dx = 0.5
    dy = 0.5

    grid_areas = np.zeros((int(180 / dy), int(360 / dx)))
    for i, lat in enumerate(np.arange(lat_start, 90, dy)):
        grid_areas[i] = (
            earth_radius2 * math.cos(lat * deg2rad) * (dx * deg2rad) * (dy * deg2rad)
        )
    grid_areas = grid_areas * fractionCountries
    tot_area = np.sum(grid_areas)
    return grid_areas, tot_area


def get_pop(ssp="ssp245", year=2015, fractionCountries=np.ones((360, 720))):
    """Return the population of the grids of a mask and its total population"""
    pop_path = "/home/ybenp/CMIP6_data/population/gridded_pop"
    pop_ssp = pop_ssp_dict[ssp]
    pop_year = str(year // 10 * 10)
    pop_file = os.path.join(pop_path, pop_ssp, f"{pop_ssp}_tot_{pop_year}.nc")
    
    f1 = Dataset(pop_file, "r")
    pop = f1["population"][:] * fractionCountries
    tot_pop = np.sum(pop)
    f1.close()
    return pop, tot_pop
