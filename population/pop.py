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
base_file = "countryFractions_2010_0.5x0.5.nc"
#### WHERE IS THE COUNTRY LEVEL .CSV FILE?
country_path = "F:\\Computer Programming\\Projects\\CMIP6\\data\\population\\"
#### WHAT'S THE NAME OF THE COUNTRY LEVEL POPULATION FILE WITH AGE GROUPS?
country_file = ["SSP1.nc", "SSP2.nc", "SSP3.nc"]
#### WHERE ARE THE FILES WITH GRIDDED POPULATION?
pop_path = "D:\\CMIP6_data\\Pop\\"
#### Where to save the output
output_path = "D:\\CMIP6_data\\Pop_Result"
####################################################################################################

#### IMPORT COUNTRY FRACTIONS
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
        fractionCountry[:, :, len(longitude) // 2 :],
        fractionCountry[:, :, : len(longitude) // 2],
    ],
    axis=2,
)
longitude = np.arange(0.25, 360, 0.5)

####################################################################################################

ssps = ["ssp1", "ssp2", "ssp3"]
age_groups = [5, 7, 9, 10]

# (2010 - 1950) / 5 = 12
years = np.arange(12, 12 + 2 * 10, 2)
year_names = np.arange(1950, 2105, 5)

for ssp in range(len(ssps)):

    data = country_path + country_file[ssp]
    country_pop_ds = xr.open_dataset(data)
    country_pop_ds.close()

    country_pop_values_all_ages = np.zeros((31, 11, 193))
    country_pop_values_all_ages[:, 0:10, :] = country_pop_ds.data_vars[
        "Population"
    ].values
    country_pop_values_all_ages[:, 10, :] = np.sum(country_pop_values_all_ages[:, [0, 2], :], axis=1)

    # Select age groups
    # Shape: Year, Age Group, Country ID
    country_pop_values = country_pop_values_all_ages[:, age_groups, :]

    # Total Population of a Country:
    country_pop_sum = country_pop_values[:, -1, :]

    for year_index in years:

        cur_year = year_names[year_index]

        # Import gridded population data
        grid_pop_file_name = f"{ssps[ssp]}_tot_{cur_year}.nc"
        grid_pop_ds = xr.open_dataset(f"{pop_path}\\{ssps[ssp]}\\{grid_pop_file_name}")
        grid_pop_values = grid_pop_ds.data_vars["population"].values
        grid_pop_ds.close()

        # Calculate and fill in the new array
        pop_array = np.zeros(shape=(len(latitude), len(longitude), len(age_groups)))
        for age in range(len(age_groups)):

            for j in range(
                len(country_pop_values[year_index][age])
            ):  # Number of countries

                ratio = (
                    country_pop_values[year_index, age, j]
                    / country_pop_sum[year_index, j]
                    if country_pop_sum[year_index, j] != 0
                    else 0
                )

                pop_array[:, :, age] += (
                    fractionCountry[j, :, :] * grid_pop_values[:, :] * ratio
                )

        # Create Dataset
        ds = xr.Dataset(
            data_vars=dict(
                post25=(["lat", "lon"], pop_array[:, :, 0]),
                post60=(["lat", "lon"], pop_array[:, :, 1]),
                post80=(["lat", "lon"], pop_array[:, :, 2]),
                all=(["lat", "lon"], pop_array[:, :, 3]),
            ),
            coords=dict(
                lat=latitude,
                lon=longitude,
            ),
            attrs=dict(
                description="Population of 25+, 60+, and 80+ converted to a 0.5x0.5 degree grid",
            ),
        )

        # Output
        os.makedirs(f"{output_path}\\{ssps[ssp]}", exist_ok=True)
        ds.to_netcdf(f"{output_path}\\{ssps[ssp]}\\{cur_year}.nc")
        ds.close()

        print(f"{datetime.now()} DONE: {ssps[ssp]}, {cur_year}")
