# -*- coding: utf-8 -*-
from netCDF4 import Dataset
import numpy as np
import os
import xarray as xr
from datetime import datetime

####################################################################################################
#### CALCULATES POPULATION DATA OF DIFFERENT AGE GROUPS (0-19, 20-39, 40-64, 65+) IN A FUTURE YEAR (2010, 2020, ... 2100)
#### PROJECTS DATA ONTO A 0.5X0.5 DEGREE GRID
####################################################################################################

####################################################################################################
#### 05x05 COUNTRY FRACTIONS (PUT THE 6 FILES TOGETHER)
#### USER INPUT:
#### WHERE IS THE 0.5X0.5 DEGREE FILE WITH COUNTRY FRACTIONS?
base_path = "F:\\Computer Programming\\Projects\\CMIP6\\data\\population\\"
#### WHAT IS THE NAME OF THE FILE WITH THE COUNTRY FRACTIONS?
#### IF 05X05
base_file = "countryFractions_2010_0.5x0.5.nc"
#### WHERE IS THE COUNTRY LEVEL .CSV FILE?
country_path = "F:\\Computer Programming\\Projects\\CMIP6\\data\\population\\"
#### WHAT'S THE NAME OF THE COUNTRY LEVEL BASELINE FILE?
country_file = ["SSP1.nc", "SSP2.nc", "SSP3.nc"]
#### WHERE ARE THE FILES WITH GRIDDED POPULATION?
pop_path = "D:\\CMIP6_data\\Pop\\"
#### Where to save the output
output_path = "D:\\CMIP6_data\\Pop_Result"
####################################################################################################

#### IMPORT COUNTRY FRACTIONS
f1 = Dataset(base_path + base_file, "r")
fractionCountry = f1.variables["fractionCountry"][:, :, :]  # countryIndex, latitude, longitude
latitude = f1.variables["latitude"][:]
longitude = f1.variables["longitude"][:]
f1.close()

fractionCountry[fractionCountry < 0.0] = 0.0
fractionCountry[fractionCountry > 1.0] = 0.0

# Change Longitude from -180 to 180 to 0 to 360 for ease of computation
fractionCountry = np.concatenate(
    [
        fractionCountry[:, :, len(longitude) // 2 :],
        fractionCountry[:, :, : len(longitude) // 2],
    ],
    axis=2,
)
longitude = np.arange(0.25, 360, 0.5)

####################################################################################################

ssps = ["ssp1", "ssp2", "ssp3"]
age_groups = [5, 7, 9]
age_group_names = np.array(["25+", "65+", "80+"], dtype=object)

years = np.arange(2010, 2110, 10)

for ssp in range(len(ssps)):

    data = country_path + country_file[ssp]
    country_pop_ds = xr.open_dataset(data)
    country_pop_all_age_values = country_pop_ds.data_vars["Population"].values
    country_pop_ds.close()

    # Select age groups
    # Shape: Year, Age Group, Country ID
    country_pop_values = country_pop_all_age_values[:, [1, 3, 6, 8], :]

    # Total Population of a Country:
    country_pop_sum = np.sum(country_pop_values, axis=1)

    for year in range(len(years)):
        # Import gridded population data
        grid_pop_file_name = f"{ssps[ssp]}_tot_{str(years[year])}.nc"
        grid_pop_ds = xr.open_dataset(f"{pop_path}\\{ssps[ssp]}\\{grid_pop_file_name}")
        grid_pop_values = grid_pop_ds.data_vars["population"].values
        grid_pop_ds.close()

        # Calculate and fill in the new array
        pop_array = np.zeros(shape=(len(latitude), len(longitude), len(age_groups)))
        for age in range(len(age_groups)):

            for i in range(len(country_pop_values[year][age])): # Number of countries

                ratio = (
                    country_pop_values[year, age, i] / country_pop_sum[year, i]
                    if country_pop_sum[year, i] != 0
                    else 0
                )

                pop_array[:, :, age] += (
                    fractionCountry[i, :, :] * grid_pop_values[:, :] * ratio
                )

        # Create Dataset
        ds = xr.Dataset(
            data_vars=dict(
                population=(["lat", "lon", "age_group"], pop_array),
            ),
            coords=dict(
                lat=latitude,
                lon=longitude,
                age_group=age_group_names,
            ),
            attrs=dict(
                description="Population of 25+, 60+, and 80+ converted to a 0.5x0.5 degree grid",
            ),
        )

        # Output
        os.makedirs(f"{output_path}\\{ssps[ssp]}", exist_ok=True)
        ds.to_netcdf(f"{output_path}\\{ssps[ssp]}\\{str(years[year])}.nc")
        ds.close()

        print(f"{datetime.now()} DONE: {ssps[ssp]}, {years[year]}")
