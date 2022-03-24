import numpy as np
import pandas as pd
import xarray as xr
import math
from netCDF4 import Dataset

####################################################################################################
#### MAPS SUBNATIONAL MORTALITY BASELINE TO A GRID
####################################################################################################

country = "kenya"
country_long_name = "Kenya"
fraction_path = "D:/CMIP6_data/fraction/"
base_path = "D:/CMIP6_data/Subnational Data_historical/"
output_path = "D:/CMIP6_data/Subnational_mortality_baseline/"

diseases = ["COPD", "Dementia", "IschemicHeartDisease", "LowerRespiratoryInfections", "LungCancer",
            "NonCommunicableDiseases", "Stroke"]
# diseases = ["Dementia"]


def combined_output():
    """Outputs gridded mortality baseline for all countries with subnational data in select countries"""


def subnational_output():
    """Outputs gridded mortality baseline for only countries with subnational data (US, UK, etc.)"""

    # import state fractions
    fraction_file = f"{country}_state_fraction_0.5x0.5.nc"
    f1 = Dataset(fraction_path + fraction_file, "r")
    fractionState = f1.variables["fractionState"][
                    :, :, :
                    ]  # stateIndex, latitude, longitude
    latitude = f1.variables["lat"][:]
    longitude = f1.variables["lon"][:]
    states = sorted(f1.variables["state"][:])
    # print(*states, sep='\n')
    f1.close()

    for i, disease in enumerate(diseases):

        # create output dataset
        # f1 = Dataset(output_path + f'us_{disease}.nc', 'w', format='NETCDF4_CLASSIC')
        # import mortality baseline
        base_file = f"{disease}_Subnatl.csv"
        # location name, age_name, metric_name, year, val
        data = pd.read_csv(base_path + base_file, usecols=[3, 7, 11, 12, 13])
        wk = data[(data["location_name"].isin(states)) & (data["year"] == 2015) & (data["metric_name"] == "Rate")]
        wk = wk.drop(columns=["metric_name", "year"])
        wk = wk.sort_values(['location_name', 'age_name'])

        age_groups = sorted(list(set(wk['age_name'].values)))
        data = np.zeros((len(age_groups), len(latitude), len(longitude)))

        for j, state in enumerate(states):

            for k, age_group in enumerate(age_groups):

                # retrieve mortality corresponding to correct age group and state
                try:
                    mort = wk[(wk["age_name"] == age_group) & (wk["location_name"] == state)]["val"].values[0]
                except IndexError:
                    print(age_group, state)
                    break

                # data[k, :, :] += (fractionState[j, :, :] / np.sum(fractionState[j, :, :])) * mort
                data[k, :, :] += fractionState[j, :, :] * mort

        if disease in ["IschemicHeartDisease", "NonCommunicableDiseases", "Stroke"]:
            ds = xr.Dataset(
                data_vars=dict(
                    post25=(["lat", "lon"], data[0]),
                    post60=(["lat", "lon"], np.sum(data[8:], axis=0)),
                    post80=(["lat", "lon"], data[12]),
                ),
                coords=dict(
                    lat=(["lat"], latitude),
                    lon=(["lon"], longitude),
                ),
                attrs=dict(
                    description=f"Gridded (0.5x0.5) mortality rate of {disease} in {country_long_name} by age groups (25+, "
                                f"60+, 80+)"),
            )
        elif disease in ["COPD", "LowerRespiratoryInfections", "LungCancer"]:
            ds = xr.Dataset(
                data_vars=dict(
                    post25=(["lat", "lon"], data[0]),
                ),
                coords=dict(
                    lat=(["lat"], latitude),
                    lon=(["lon"], longitude),
                ),
                attrs=dict(
                    description=f"Gridded (0.5x0.5) mortality rate of {disease} in {country_long_name} by age groups (25+)"),
            )
        elif disease == "Dementia":
            ds = xr.Dataset(
                data_vars=dict(
                    post65=(["lat", "lon"], np.sum(data[[0, 2]], axis=0)),
                    post75=(["lat", "lon"], data[2]),
                ),
                coords=dict(
                    lat=(["lat"], latitude),
                    lon=(["lon"], longitude),
                ),
                attrs=dict(
                    description=f"Gridded (0.5x0.5) mortality rate of {disease} in {country_long_name} by age groups (75+)"),
            )

        output_file = f"{country}_{disease}_subnatl.nc"
        ds.to_netcdf(output_path + output_file)
        ds.close()

        print(f"DONE: {disease}")


def find_diff(states):
    base_file = f"{diseases[0]}_Subnatl.csv"
    data = pd.read_csv(base_path + base_file, usecols=[2, 3])
    data = data[(data["location_id"].isin(np.arange(35617, 35664)))]
    state_names = sorted(list(set(data["location_name"].values)))
    for state in state_names:
        if state not in states:
            print(state)
