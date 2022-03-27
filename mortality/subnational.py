import numpy as np
import pandas as pd
import xarray as xr
import math
from netCDF4 import Dataset

####################################################################################################
#### MAPS SUBNATIONAL MORTALITY BASELINE TO A GRID
####################################################################################################

fraction_path = "D:/CMIP6_data/fraction/"
base_path = "D:/CMIP6_data/Subnational Data_historical/"

diseases = ["COPD", "Dementia", "IschemicHeartDisease", "LowerRespiratoryInfections", "LungCancer",
            "NonCommunicableDiseases", "Stroke"]
# diseases = ["NonCommunicableDiseases"]


def rename_helper(df, to_be_dropped, to_be_renamed):
    location_names = df["location_name"].drop_duplicates().tolist()

    # Drop countries
    location_names = [x for x in location_names if x not in to_be_dropped]
    df = df.set_index("location_name")
    df = df.loc[location_names]

    # Rename countries
    labels = df.index.values
    new_labels = list(
        map(lambda x: to_be_renamed[x] if x in to_be_renamed else x, labels)
    )

    df = df.set_axis(new_labels, axis="index")
    df.index.name = "location_name"
    df = df.reset_index()
    df = df.sort_values(by=["location_name", "year", "age_name"])
    df = df.reset_index()
    df = df.drop(columns=["index"])

    # print(df)

    return df


def combined_output():
    """Outputs gridded mortality baseline for all countries with subnational data in select countries"""

    # Import country fraction
    fraction_file = f"countryFractions_2010_0.5x0.5.nc"
    f1 = Dataset(fraction_path + fraction_file, "r")
    fractionCountry = f1.variables["fractionCountry"][
                      :, :, :
                      ]  # countryIndex, latitude, longitude
    latitude = f1.variables["latitude"][:]
    longitude = f1.variables["longitude"][:]
    f1.close()

    fractionCountry[fractionCountry < 0.0] = 0.0
    fractionCountry[fractionCountry > 1.0] = 0.0
    # Change Longitude from -180 to 180 to 0 to 360 for ease of computation
    # fractionCountry = np.concatenate(
    #     [
    #         fractionCountry[:, :, len(longitude) // 2:],
    #         fractionCountry[:, :, : len(longitude) // 2],
    #     ],
    #     axis=2,
    # )
    # longitude = np.arange(0.25, 360, 0.5)

    national_baseline_path = "D:/CMIP6_data/National Data_historical/"
    subnational_baseline_path = "D:/CMIP6_data/Subnational_mortality_baseline/"
    output_path = "D:/CMIP6_data/Combined_mortality_baseline/"
    # Countries with subnational data
    countries = ["brazil", "indonesia", "japan", "kenya", "mexico", "uk", "us"]
    country_ids = [23, 78, 85, 88, 109, 181, 183]
    to_be_dropped = ["American Samoa", "Bermuda", "Greenland", "Guam", "Montenegro", "Northern Mariana Islands", "Palestine", "Taiwan (Province of China)", "Tokelau", "United States Virgin Islands", "South Sudan"]
    to_be_renamed = {
        "Cabo Verde": "Cape Verde",
        "Eswatini": "Swaziland",
        "North Macedonia": "ThbMacedonia", # ensure correct placement between Thailand and Timor-Leste
        "Puerto Rico": "ZZPuerto Rico",
    }

    for disease in diseases:
        national_baseline_file = f"{disease}.csv"
        try:
            data = pd.read_csv(national_baseline_path + national_baseline_file, usecols=[3, 7, 11, 12, 13])
        except:
            continue
        data = rename_helper(data, to_be_dropped, to_be_renamed)
        wk = data[(~data["location_name"].isin(to_be_dropped)) & (data["year"] == 2015) & (data["metric_name"] == "Rate")]
        wk = wk.drop(columns=["metric_name", "year"])
        wk = wk.sort_values(['location_name', 'age_name'])

        age_groups = sorted(list(set(wk['age_name'].values)))
        data = np.zeros((len(age_groups), len(latitude), len(longitude)))

        # Loop through countries
        for j in np.arange(0, 193):

            # Skip country if we have subnational data of it
            if j in country_ids:
                continue

            for k, age_group in enumerate(age_groups):

                # retrieve mortality corresponding to correct age group and state
                try:
                    tmp = wk.iloc[np.arange(j * len(age_groups), (j + 1) * len(age_groups))]
                    mort = tmp[(tmp["age_name"] == age_group)]["val"].values[0]
                except IndexError:
                    print(age_group, j)
                    break

                data[k, :, :] += fractionCountry[j, :, :] * mort

        if disease in ["IschemicHeartDisease", "NonCommunicableDiseases", "Stroke"]:
            post25 = data[0]
            post60 = np.sum(data[8:], axis=0)
            post80 = data[12]

            # Add in subnational data
            for country in countries:
                subnational_baseline_file = f"{country}_{disease}_subnatl.nc"
                subnatl_wk = xr.open_dataset(subnational_baseline_path + subnational_baseline_file)
                subnatl_data = subnatl_wk.data_vars
                post25[:, :] += subnatl_data["post25"].values
                post60[:, :] += subnatl_data["post60"].values
                post80[:, :] += subnatl_data["post80"].values

            ds = xr.Dataset(
                data_vars=dict(
                    post25=(["lat", "lon"], post25),
                    post60=(["lat", "lon"], post60),
                    post80=(["lat", "lon"], post80),
                ),
                coords=dict(
                    lat=(["lat"], latitude),
                    lon=(["lon"], longitude),
                ),
                attrs=dict(
                    description=f"Gridded (0.5x0.5) mortality rate of {disease} by age groups (25+, "
                                f"60+, 80+), with subnational-level data in Brazil, Indonesia, Japan, Kenya, Mexico, "
                                f"UK, and US"),
            )
        elif disease in ["COPD", "LowerRespiratoryInfections", "LungCancer"]:
            post25 = data[0]

            # Add in subnational data
            for country in countries:
                subnational_baseline_file = f"{country}_{disease}_subnatl.nc"
                subnatl_wk = xr.open_dataset(subnational_baseline_path + subnational_baseline_file)
                subnatl_data = subnatl_wk.data_vars
                post25[:, :] += subnatl_data["post25"].values

            ds = xr.Dataset(
                data_vars=dict(
                    post25=(["lat", "lon"], post25),
                ),
                coords=dict(
                    lat=(["lat"], latitude),
                    lon=(["lon"], longitude),

                ),
                attrs=dict(
                    description=f"Gridded (0.5x0.5) mortality rate of {disease} by age groups (25+), with "
                                f"subnational-level data in Brazil, Indonesia, Japan, Kenya, Mexico, "
                                f"UK, and US"),
            )
        elif disease == "Dementia":
            post65 = np.sum(data[[0, 2]], axis=0)
            post75 = data[2]

            # Add in subnational data
            for country in countries:
                subnational_baseline_file = f"{country}_{disease}_subnatl.nc"
                subnatl_wk = xr.open_dataset(subnational_baseline_path + subnational_baseline_file)
                subnatl_data = subnatl_wk.data_vars
                post65[:, :] += subnatl_data["post65"].values
                post75[:, :] += subnatl_data["post75"].values
            ds = xr.Dataset(
                data_vars=dict(
                    post65=(["lat", "lon"], post65),
                    post75=(["lat", "lon"], post75),
                ),
                coords=dict(
                    lat=(["lat"], latitude),
                    lon=(["lon"], longitude),
                ),
                attrs=dict(
                    description=f"Gridded (0.5x0.5) mortality rate of {disease} by age groups (65+, 75+), with "
                                f"subnational-level data in Brazil, Indonesia, Japan, Kenya, Mexico, "
                                f"UK, and US"),
            )

        output_file = f"{disease}_combined.nc"
        ds.to_netcdf(output_path + output_file)
        ds.close()

        print(f"DONE: {disease}")


def subnational_output():
    """Outputs gridded mortality baseline for only countries with subnational data (US, UK, etc.)"""

    country = "kenya"
    country_long_name = "Kenya"
    output_path = "D:/CMIP6_data/Subnational_mortality_baseline/"
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


combined_output()
