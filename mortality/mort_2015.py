import numpy as np
import pandas as pd
import xarray as xr
import math
from netCDF4 import Dataset

####################################################################################################
#### MAPS SUBNATIONAL MORTALITY BASELINE TO A GRID
####################################################################################################

fraction_path = "D:/CMIP6_data/fraction/"
base_path = "D:/CMIP6_data/Mortality/Subnational Data_historical/"

diseases = ["COPD", "Dementia", "IschemicHeartDisease", "LowerRespiratoryInfections", "LungCancer",
            "NonCommunicableDiseases", "Stroke", "Diabetes"]
data_types = ["val", "upper", "lower"]

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

national_baseline_path = "D:/CMIP6_data/Mortality/National Data_historical/"
subnational_baseline_path = "D:/CMIP6_data/Mortality/Output/Subnational_mortality_baseline_2015/"

countries = ["brazil", "indonesia", "japan", "kenya", "mexico", "uk", "us"]
country_long_names = ["Brazil", "Indonesia", "Japan", "Kenya", "Mexico", "the United Kingdom", "the United States"]
country_ids = [23, 78, 85, 88, 109, 181, 183]
to_be_dropped = ["American Samoa", "Bermuda", "Greenland", "Guam", "Montenegro", "Northern Mariana Islands",
                 "Palestine", "Taiwan (Province of China)", "Tokelau", "United States Virgin Islands",
                 "South Sudan"]
to_be_renamed = {
    "Cabo Verde": "Cape Verde",
    "Eswatini": "Swaziland",
    "North Macedonia": "ThbMacedonia",  # ensure correct placement between Thailand and Timor-Leste
    "Puerto Rico": "ZZPuerto Rico",
}


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


def gen_output(disease, data, output_description, is_combined=False):
    if disease in ["IschemicHeartDisease", "NonCommunicableDiseases", "Stroke"]:
        post25_mean = data[0][0]
        post25_upper = data[0][1]
        post25_lower = data[0][2]
        post60_mean = np.sum(data[8:], axis=0)[0]
        post60_upper = np.sum(data[8:], axis=0)[1]
        post60_lower = np.sum(data[8:], axis=0)[2]
        post80_mean = data[12][0]
        post80_upper = data[12][1]
        post80_lower = data[12][2]

        # if is_combined:

        # Add in subnational data
        # for country in countries:
        #     subnational_baseline_file = f"{country}_{disease}.nc"
        #     subnatl_wk = xr.open_dataset(subnational_baseline_path + subnational_baseline_file)
        #     subnatl_data = subnatl_wk.data_vars
        #     post25[:, :] += subnatl_data["post25"].values
        #     post60[:, :] += subnatl_data["post60"].values
        #     post80[:, :] += subnatl_data["post80"].values

        ds = xr.Dataset(
            data_vars=dict(
                post25_mean=(["lat", "lon"], post25_mean),
                post25_upper=(["lat", "lon"], post25_upper),
                post25_lower=(["lat", "lon"], post25_lower),
                post60_mean=(["lat", "lon"], post60_mean),
                post60_upper=(["lat", "lon"], post60_upper),
                post60_lower=(["lat", "lon"], post60_lower),
                post80_mean=(["lat", "lon"], post80_mean),
                post80_upper=(["lat", "lon"], post80_upper),
                post80_lower=(["lat", "lon"], post80_lower),
            ),
            coords=dict(
                lat=(["lat"], latitude),
                lon=(["lon"], longitude),
            ),
            attrs=dict(
                description=output_description,
            ),
        )

    elif disease in ["COPD", "LowerRespiratoryInfections", "LungCancer", "Diabetes"]:
        post25_mean = data[0][0]
        post25_upper = data[0][1]
        post25_lower = data[0][2]

        # if is_combined:

        # Add in subnational data
        # for country in countries:
        #     subnational_baseline_file = f"{country}_{disease}.nc"
        #     subnatl_wk = xr.open_dataset(subnational_baseline_path + subnational_baseline_file)
        #     subnatl_data = subnatl_wk.data_vars
        #     post25[:, :] += subnatl_data["post25"].values
        #     post60[:, :] += subnatl_data["post60"].values
        #     post80[:, :] += subnatl_data["post80"].values

        ds = xr.Dataset(
            data_vars=dict(
                post25_mean=(["lat", "lon"], post25_mean),
                post25_upper=(["lat", "lon"], post25_upper),
                post25_lower=(["lat", "lon"], post25_lower),
            ),
            coords=dict(
                lat=(["lat"], latitude),
                lon=(["lon"], longitude),
            ),
            attrs=dict(
                description=output_description,
            ),
        )

    elif disease == "Dementia":
        post65_mean = np.sum(data[[0, 2]], axis=0)[0]
        post65_upper = np.sum(data[[0, 2]], axis=0)[1]
        post65_lower = np.sum(data[[0, 2]], axis=0)[2]
        post75_mean = data[2][0]
        post75_upper = data[2][1]
        post75_lower = data[2][2]

        # if is_combined:

        # Add in subnational data
        # for country in countries:
        #     subnational_baseline_file = f"{country}_{disease}.nc"
        #     subnatl_wk = xr.open_dataset(subnational_baseline_path + subnational_baseline_file)
        #     subnatl_data = subnatl_wk.data_vars
        #     post25[:, :] += subnatl_data["post25"].values
        #     post60[:, :] += subnatl_data["post60"].values
        #     post80[:, :] += subnatl_data["post80"].values

        ds = xr.Dataset(
            data_vars=dict(
                post65_mean=(["lat", "lon"], post65_mean),
                post65_upper=(["lat", "lon"], post65_upper),
                post65_lower=(["lat", "lon"], post65_lower),
                post75_mean=(["lat", "lon"], post75_mean),
                post75_upper=(["lat", "lon"], post75_upper),
                post75_lower=(["lat", "lon"], post75_lower),
            ),
            coords=dict(
                lat=(["lat"], latitude),
                lon=(["lon"], longitude),
            ),
            attrs=dict(
                description=output_description,
            )
        )

    return ds


def combined_output():
    """Outputs gridded mortality baseline for all countries with subnational data in select countries"""

    output_path = "D:/CMIP6_data/Mortality/Output/Combined_mortality_baseline_2015/"

    for disease in diseases:
        national_baseline_file = f"{disease}.csv"
        try:
            # location name, age_name, metric_name, year, val, upper, lower
            data = pd.read_csv(national_baseline_path + national_baseline_file, usecols=[3, 7, 11, 12, 13])
        except:
            print(f"Error importing {disease}")
            continue
        data = rename_helper(data, to_be_dropped, to_be_renamed)
        wk = data[
            (~data["location_name"].isin(to_be_dropped)) & (data["year"] == 2015) & (data["metric_name"] == "Rate")]
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

        national_output_path = "D:/CMIP6_data/Mortality/Output/National_mortality_baseline_2015/"
        output_file = f"{disease}.nc"
        output_description = f"Gridded (0.5x0.5) mortality rate of {disease} by age groups in 2015, with national-level data and subnational-level data in Brazil, Indonesia, Japan, Kenya, Mexico, UK, and US"

        ds = gen_output(disease, data, output_description)
        ds.to_netcdf(output_path + output_file)
        ds.close()

        print(f"DONE: {disease}")


def national_output():
    """Outputs gridded mortality baseline for all countries"""

    output_path = "D:/CMIP6_data/Mortality/Output/National_mortality_baseline_2015/"

    for disease in diseases:
        national_baseline_file = f"{disease}.csv"
        try:
            data = pd.read_csv(national_baseline_path + national_baseline_file, usecols=[3, 7, 11, 12, 13, 14, 15])
        except:
            print(f"Error importing {disease}")
            continue
        data = rename_helper(data, to_be_dropped, to_be_renamed)
        wk = data[
            (~data["location_name"].isin(to_be_dropped)) & (data["year"] == 2015) & (data["metric_name"] == "Rate")]
        wk = wk.drop(columns=["metric_name", "year"])
        wk = wk.sort_values(['location_name', 'age_name'])

        age_groups = sorted(list(set(wk['age_name'].values)))
        data = np.zeros((len(age_groups), len(data_types), len(latitude), len(longitude)))

        # Loop through countries
        for j in np.arange(0, 193):

            for k, age_group in enumerate(age_groups):

                for p, data_type in enumerate(data_types):

                    # retrieve mortality corresponding to correct age group and state
                    try:
                        tmp = wk.iloc[np.arange(j * len(age_groups), (j + 1) * len(age_groups))]
                        mort = tmp[(tmp["age_name"] == age_group)][data_type].values[0]
                    except IndexError:
                        print(age_group, j)
                        break

                    data[k, p, :, :] += fractionCountry[j, :, :] * mort

        output_file = f"{disease}.nc"
        output_description = f"Gridded (0.5x0.5) mortality rate of {disease} by age groups in 2015, with national-level data only"

        ds = gen_output(disease, data, output_description)
        # ds.to_netcdf(output_path + output_file)
        ds.close()

        print(f"DONE: {disease}")


def subnational_output():
    """Outputs gridded mortality baseline for only countries with subnational data (US, UK, etc.)"""

    for country, country_long_name in zip(countries, country_long_names):

        output_path = "D:/CMIP6_data/Mortality/Output/Subnational_mortality_baseline_2015/"
        # import state fractions
        fraction_file = f"{country}_state_fraction_0.5x0.5.nc"
        f1 = Dataset(fraction_path + fraction_file, "r")
        fractionState = f1.variables["fractionState"][
                        :, :, :
                        ]  # stateIndex, latitude, longitude
        states = f1.variables["state"][:]
        f1.close()

        for i, disease in enumerate(diseases):

            # import mortality baseline
            base_file = f"{disease}_Subnatl.csv"
            try:
                data = pd.read_csv(base_path + base_file, usecols=[3, 7, 11, 12, 13, 14, 15])
            except:
                print(f"Error importing {disease}")
                continue
            wk = data[(data["location_name"].isin(states)) & (data["year"] == 2015) & (data["metric_name"] == "Rate")]
            wk = wk.drop(columns=["metric_name", "year"])
            wk = wk.sort_values(['location_name', 'age_name'])

            age_groups = sorted(list(set(wk['age_name'].values)))
            data = np.zeros((len(age_groups), len(data_types), len(latitude), len(longitude)))

            for j, state in enumerate(states):

                for k, age_group in enumerate(age_groups):

                    for p, data_type in enumerate(data_types):

                        # retrieve mortality corresponding to correct age group and state
                        try:
                            mort = wk[(wk["age_name"] == age_group) & (wk["location_name"] == state)][data_type].values[
                                0]
                        except IndexError:
                            print(age_group, state)
                            break

                        data[k, p, :, :] += fractionState[j, :, :] * mort

            output_description = f"Gridded (0.5x0.5) mortality rate of {disease} in {country_long_name} by age groups in 2015"
            output_file = f"{country}_{disease}.nc"

            ds = gen_output(disease, data, output_description)
            ds.to_netcdf(output_path + output_file)
            ds.close()

            print(f"DONE: {country_long_name}, {disease}")


# def find_diff(states):
#     base_file = f"{diseases[0]}_Subnatl.csv"
#     data = pd.read_csv(base_path + base_file, usecols=[2, 3])
#     data = data[(data["location_id"].isin(np.arange(35617, 35664)))]
#     state_names = sorted(list(set(data["location_name"].values)))
#     for state in state_names:
#         if state not in states:
#             print(state)


def main():
    subnational_output()


if __name__ == "__main__":
    main()
