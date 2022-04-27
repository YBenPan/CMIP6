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
base_path = "D:/CMIP6_data/population/national_pop/"
#### WHAT IS THE NAME OF THE FILE WITH THE COUNTRY FRACTIONS?
base_file = "countryFractions_2010_0.5x0.5.nc"
#### WHERE IS THE COUNTRY LEVEL .CSV FILE?
country_path = "D:/CMIP6_data/population/national_pop/"
#### WHAT'S THE NAME OF THE COUNTRY LEVEL POPULATION FILE WITH AGE GROUPS?
country_file = ["SSP1.nc", "SSP2.nc", "SSP3.nc"]
#### WHERE ARE THE FILES WITH GRIDDED POPULATION?
pop_path = "D:/CMIP6_data/population/gridded_pop/"
#### Where to save the output
output_path = "D:/CMIP6_data/population/gridded_pop_age_5_years/"
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
age_groups = [
    5,  # 25-29
    6,  # 30-34
    7,  # 35-39
    8,  # 40-44
    9,  # 45-49
    11, # 50-54
    12, # 55-59
    13, # 60-64
    14, # 65-69
    15, # 70-74
    16, # 75-79
    17, # 80-84
    18, # 85-89
    19, # 90-94
    20, # 95-99
    2,  # 100+
    21  # All
]

# (2010 - 1950) / 5 = 12
years = np.arange(12, 12 + 2 * 10, 2)
year_names = np.arange(1950, 2105, 5)

for ssp in range(len(ssps)):

    data = country_path + country_file[ssp]
    country_pop_ds = xr.open_dataset(data)
    country_pop_ds.close()

    country_pop_values_all_ages = np.zeros((31, 22, 193))
    country_pop_values_all_ages = country_pop_ds.data_vars[
        "Population"
    ].values

    # Select age groups
    # Shape: Year, Age Group, Country ID
    country_pop_values = country_pop_values_all_ages[:, age_groups, :]

    # Total Population of a Country:
    country_pop_sum = country_pop_values[:, -1, :]

    for year_index in years:

        cur_year = year_names[year_index]

        # Import gridded population data
        grid_pop_file_name = f"{ssps[ssp]}_tot_{cur_year}.nc"
        grid_pop_ds = xr.open_dataset(f"{pop_path}/{ssps[ssp]}/{grid_pop_file_name}")
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

        print(f"SSP {ssp}, Year {1950 + 5 * year_index} ",
              round(np.sum(pop_array[:, :, 0:16]) / 10 ** 9, 1),
              round(np.sum(pop_array[:, :, 7:16]) / 10 ** 9, 1),
              round(np.sum(pop_array[:, :, 11:16]) / 10 ** 9, 1),
              round(np.sum(pop_array[:, :, -1]) / 10 ** 9, 1),
        )

        # Create Dataset
        ds = xr.Dataset(
            data_vars=dict(
                age_25_29=(["lat", "lon"], pop_array[:, :, 0]),
                age_30_34=(["lat", "lon"], pop_array[:, :, 1]),
                age_35_39=(["lat", "lon"], pop_array[:, :, 2]),
                age_40_44=(["lat", "lon"], pop_array[:, :, 3]),
                age_45_49=(["lat", "lon"], pop_array[:, :, 4]),
                age_50_54=(["lat", "lon"], pop_array[:, :, 5]),
                age_55_59=(["lat", "lon"], pop_array[:, :, 6]),
                age_60_64=(["lat", "lon"], pop_array[:, :, 7]),
                age_65_69=(["lat", "lon"], pop_array[:, :, 8]),
                age_70_74=(["lat", "lon"], pop_array[:, :, 9]),
                age_75_79=(["lat", "lon"], pop_array[:, :, 10]),
                age_80_84=(["lat", "lon"], pop_array[:, :, 11]),
                age_85_89=(["lat", "lon"], pop_array[:, :, 12]),
                age_90_94=(["lat", "lon"], pop_array[:, :, 13]),
                post95=(["lat", "lon"], pop_array[:, :, 14] + pop_array[:, :, 15]),
                all=(["lat", "lon"], pop_array[:, :, 16]),
            ),
            coords=dict(
                lat=latitude,
                lon=longitude,
            ),
            attrs=dict(
                description="Population by age groups converted to a 0.5x0.5 degree grid",
            ),
        )

        # Output
        os.makedirs(f"{output_path}\\{ssps[ssp]}", exist_ok=True)
        ds.to_netcdf(f"{output_path}\\{ssps[ssp]}\\{cur_year}.nc")
        ds.close()

        print(f"{datetime.now()} DONE: {ssps[ssp]}, {cur_year}")
