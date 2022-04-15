import numpy as np
import pandas as pd
import xarray as xr
import math
from netCDF4 import Dataset

####################################################################################################
#### MAPS SUBNATIONAL MORTALITY BASELINE TO A GRID
####################################################################################################

fraction_path = "D:/CMIP6_data/fraction/"

diseases = ["copd", "dementia", "ihd", "lri", "lc", "ncd", "stroke"]
disease_names = ["COPD", "Dementia", "IschemicHeartDisease", "LowerRespiratoryInfections", "LungCancer",
                 "NonCommunicableDiseases", "Stroke"]
data_types_2015 = ["val", "upper", "lower"]
data_types_2040 = [["Deaths.2016.Number", "Deaths.2040.Number"], ["Deaths.2016.Upper", "Deaths.2040.Upper"], ["Deaths.2016.Lower", "Deaths.2040.Lower"], ]

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
national_projection_path = "D:/CMIP6_data/Mortality/Mortality Projections_2040/"
subnational_baseline_path = "D:/CMIP6_data/Mortality/Output/Subnational_mortality_baseline_2040/"

# Countries with subnational data
countries = ["brazil", "indonesia", "japan", "kenya", "mexico", "uk", "us"]
country_ids = [23, 78, 85, 88, 109, 181, 183]
country_long_names = ["Brazil", "Indonesia", "Japan", "Kenya", "Mexico", "the United Kingdom", "the United States"]

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
    """Drop and rename countries"""
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
    """Generate output Dataset"""
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

        if is_combined:
            # Add in subnational data
            for country in countries:
                subnational_baseline_file = f"{country}_{disease}.nc"
                try:
                    subnatl_wk = xr.open_dataset(subnational_baseline_path + subnational_baseline_file)
                    subnatl_data = subnatl_wk.data_vars
                except:
                    print(f"Error importing {disease} {country} subnational data")
                    continue
                post25_mean[:, :] += subnatl_data["post25_mean"].values
                post25_upper[:, :] += subnatl_data["post25_upper"].values
                post25_lower[:, :] += subnatl_data["post25_lower"].values
                post60_mean[:, :] += subnatl_data["post60_mean"].values
                post60_upper[:, :] += subnatl_data["post60_upper"].values
                post60_lower[:, :] += subnatl_data["post60_lower"].values
                post80_mean[:, :] += subnatl_data["post80_mean"].values
                post80_upper[:, :] += subnatl_data["post80_upper"].values
                post80_lower[:, :] += subnatl_data["post80_lower"].values

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

        if is_combined:
            # Add in subnational data
            for country in countries:
                subnational_baseline_file = f"{country}_{disease}.nc"
                try:
                    subnatl_wk = xr.open_dataset(subnational_baseline_path + subnational_baseline_file)
                    subnatl_data = subnatl_wk.data_vars
                except:
                    print(f"Error importing {disease} {country} subnational data")
                    continue
                post25_mean[:, :] += subnatl_data["post25_mean"].values
                post25_upper[:, :] += subnatl_data["post25_upper"].values
                post25_lower[:, :] += subnatl_data["post25_lower"].values

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

        if is_combined:
            # Add in subnational data
            for country in countries:
                subnational_baseline_file = f"{country}_{disease}.nc"
                try:
                    subnatl_wk = xr.open_dataset(subnational_baseline_path + subnational_baseline_file)
                    subnatl_data = subnatl_wk.data_vars
                except:
                    print(f"Error importing {disease} {country} subnational data")
                    continue
                post65_mean[:, :] += subnatl_data["post65_mean"].values
                post65_upper[:, :] += subnatl_data["post65_upper"].values
                post65_lower[:, :] += subnatl_data["post65_lower"].values
                post75_mean[:, :] += subnatl_data["post75_mean"].values
                post75_upper[:, :] += subnatl_data["post75_upper"].values
                post75_lower[:, :] += subnatl_data["post75_lower"].values

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
    else:
        raise Exception("Undefined disease")

    return ds


def combined_output():
    """Outputs gridded mortality baseline for all countries with subnational data in select countries in 2040"""

    output_path = "D:/CMIP6_data/Mortality/Output/Combined_mortality_baseline_2040/"

    for disease, disease_name in zip(diseases, disease_names):

        # Import national baseline data from 2015
        national_baseline_file = f"{disease_name}.csv"
        try:
            natl_2015 = pd.read_csv(national_baseline_path + national_baseline_file, usecols=[3, 7, 11, 12, 13, 14, 15])
        except:
            print("Error importing 2015 national baseline files", disease, national_baseline_path + national_baseline_file)
            continue
        natl_2015 = rename_helper(natl_2015, to_be_dropped, to_be_renamed)
        natl_2015 = natl_2015[
            (~natl_2015["location_name"].isin(to_be_dropped)) & (natl_2015["year"] == 2015) & (natl_2015["metric_name"] == "Rate")]
        natl_2015 = natl_2015.drop(columns=["metric_name", "year"])
        natl_2015 = natl_2015.sort_values(['location_name', 'age_name'])
        age_groups = sorted(list(set(natl_2015['age_name'].values)))

        # Import national baseline projection from 2040
        national_projection_file = f"{disease}_rate.csv"
        try:
            natl_2040 = pd.read_csv(national_projection_path + national_projection_file, usecols=[2, 18, 19, 20, 21, 22, 23])
        except:
            print("Error importing 2040 baseline projection files", disease, national_projection_path + national_projection_file)
            continue
        natl_2040 = natl_2040.fillna(0)

        data = np.zeros((len(age_groups), len(data_types_2040), len(latitude), len(longitude)))

        # Loop through countries
        for j in np.arange(0, 193):

            # Skip country if we have subnational data of it
            if j in country_ids:
                continue

            for k, age_group in enumerate(age_groups):

                for p, data_type_2015 in enumerate(data_types_2015):

                    data_type_2040 = data_types_2040[p]

                    # Calculate ratio to evolve age group data from 2015 to 2040
                    natl_2015_val = natl_2040.iloc[j][data_type_2040[0]]
                    natl_2040_val = natl_2040.iloc[j][data_type_2040[1]]
                    if natl_2015_val == 0 or np.isnan(natl_2015_val) or natl_2040_val == 0 or np.isnan(natl_2040_val):
                        ratio = 0
                    else:
                        ratio = natl_2040_val / natl_2015_val

                    # retrieve mortality corresponding to correct age group and state
                    try:
                        tmp = natl_2015.iloc[np.arange(j * len(age_groups), (j + 1) * len(age_groups))]
                        mort = tmp[(tmp["age_name"] == age_group)][data_type_2015].values[0]
                    except IndexError:
                        print("Error importing 2015 national baseline files", age_group, j)
                        break

                    data[k, p, :, :] += fractionCountry[j, :, :] * mort * ratio

        output_file = f"{disease_name}.nc"
        output_description = f"Gridded (0.5x0.5) mortality rate of {disease} by age groups in 2040, with " \
                             f"subnational-level data in Brazil, Indonesia, Japan, Kenya, " \
                             f"Mexico, UK, and US and national-level data"
        ds = gen_output(disease_name, data, output_description, is_combined=True)
        ds.to_netcdf(output_path + output_file)
        ds.close()

        print(f"DONE: {disease_name}")


def subnational_output():
    """Outputs gridded mortality baseline for only countries with subnational data (US, UK, etc.)"""
    output_path = "D:/CMIP6_data/Mortality/Output/Subnational_mortality_baseline_2040/"
    national_projection_path = "D:/CMIP6_data/Mortality/Mortality Projections_2040/"

    for country, country_long_name, country_id in zip(countries, country_long_names, country_ids):

        # import state fractions
        fraction_file = f"{country}_state_fraction_0.5x0.5.nc"
        f1 = Dataset(fraction_path + fraction_file, "r")
        fractionState = f1.variables["fractionState"][
                        :, :, :
                        ]  # stateIndex, latitude, longitude
        states = f1.variables["state"][:]
        f1.close()

        for i, disease in enumerate(diseases):
            disease_name = disease_names[i]

            # Import projection and calculate ratio
            national_projection_file = f"{disease}_rate.csv"
            try:
                natl_2040 = pd.read_csv(national_projection_path + national_projection_file, usecols=[2, 18, 19, 20, 21, 22, 23])
            except:
                print("Error importing 2040 baseline projection files", disease, national_projection_path + national_projection_file)
                continue
            natl_2040 = natl_2040.fillna(0)

            # import mortality baseline
            subnatl_baseline_path = "D:/CMIP6_data/Mortality/Subnational Data_historical/"
            subnatl_baseline_file = f"{disease_name}_Subnatl.csv"
            # location name, age_name, metric_name, year, val
            subnatl_2015 = pd.read_csv(subnatl_baseline_path + subnatl_baseline_file, usecols=[3, 7, 11, 12, 13, 14, 15])
            subnatl_2015 = subnatl_2015[(subnatl_2015["location_name"].isin(states)) & (subnatl_2015["year"] == 2015) & (subnatl_2015["metric_name"] == "Rate")]
            subnatl_2015 = subnatl_2015.drop(columns=["metric_name", "year"])
            subnatl_2015 = subnatl_2015.sort_values(['location_name', 'age_name'])

            age_groups = sorted(list(set(subnatl_2015['age_name'].values)))
            data = np.zeros((len(age_groups), len(data_types_2040), len(latitude), len(longitude)))

            for j, state in enumerate(states):

                for k, age_group in enumerate(age_groups):

                    for p, data_type_2015 in enumerate(data_types_2015):

                        data_type_2040 = data_types_2040[p]
                        # Calculate ratio to evolve age group data from 2015 to 2040
                        natl_2015_val = natl_2040.iloc[country_id][data_type_2040[0]]
                        natl_2040_val = natl_2040.iloc[country_id][data_type_2040[1]]
                        # print(j, age_group, data_type_2015, natl_2015_val, natl_2040_val)
                        if natl_2015_val == 0 or np.isnan(natl_2015_val) or natl_2040_val == 0 or np.isnan(
                                natl_2040_val):
                            ratio = 0
                        else:
                            ratio = natl_2040_val / natl_2015_val

                        # Retrieve mortality corresponding to correct age group and state
                        try:
                            mort = subnatl_2015[(subnatl_2015["age_name"] == age_group) & (subnatl_2015["location_name"] == state)][data_type_2015].values[0]
                        except IndexError:
                            print(age_group, state)
                            break

                        data[k, p, :, :] += fractionState[j, :, :] * mort * ratio

            output_file = f"{country}_{disease_name}.nc"
            output_description = f"Gridded (0.5x0.5) mortality rate of {disease} in {country_long_name} by age groups" \
                                 f" in 2040 "
            ds = gen_output(disease_name, data, output_description)
            ds.to_netcdf(output_path + output_file)
            ds.close()

            print(f"DONE: {country_long_name}, {disease_name}")


def national_output():

    """Outputs gridded mortality baseline for all countries in 2040"""

    for disease, disease_name in zip(diseases, disease_names):

        # Import national baseline data from 2015
        national_baseline_file = f"{disease_name}.csv"
        try:
            natl_2015 = pd.read_csv(national_baseline_path + national_baseline_file, usecols=[3, 7, 11, 12, 13, 14, 15])
        except:
            print(national_baseline_path + national_baseline_file)
            continue
        natl_2015 = rename_helper(natl_2015, to_be_dropped, to_be_renamed)
        natl_2015 = natl_2015[
            (~natl_2015["location_name"].isin(to_be_dropped)) & (natl_2015["year"] == 2015) & (natl_2015["metric_name"] == "Rate")]
        natl_2015 = natl_2015.drop(columns=["metric_name", "year"])
        natl_2015 = natl_2015.sort_values(['location_name', 'age_name'])
        age_groups = sorted(list(set(natl_2015['age_name'].values)))

        # Import national baseline projection from 2040
        national_projection_file = f"{disease}_rate.csv"
        try:
            natl_2040 = pd.read_csv(national_projection_path + national_projection_file, usecols=[2, 18, 19, 20, 21, 22, 23])
        except:
            continue
        natl_2040 = natl_2040.fillna(0)

        data = np.zeros((len(age_groups), len(data_types_2040), len(latitude), len(longitude)))

        # Loop through countries
        for j in np.arange(0, 193):

            for k, age_group in enumerate(age_groups):

                for p, data_type_2015 in enumerate(data_types_2015):

                    data_type_2040 = data_types_2040[p]

                    # Calculate ratio to evolve age group data from 2015 to 2040
                    natl_2015_val = natl_2040.iloc[j][data_type_2040[0]]
                    natl_2040_val = natl_2040.iloc[j][data_type_2040[1]]
                    # print(j, age_group, data_type_2015, natl_2015_val, natl_2040_val)
                    if natl_2015_val == 0 or np.isnan(natl_2015_val) or natl_2040_val == 0 or np.isnan(natl_2040_val):
                        ratio = 0
                    else:
                        ratio = natl_2040_val / natl_2015_val

                    # Retrieve mortality corresponding to correct age group and state
                    try:
                        tmp = natl_2015.iloc[np.arange(j * len(age_groups), (j + 1) * len(age_groups))]
                        mort = tmp[(tmp["age_name"] == age_group)][data_type_2015].values[0]
                    except IndexError:
                        print(age_group, j)
                        break

                    data[k, p, :, :] += fractionCountry[j, :, :] * mort * ratio

        output_description = f"Gridded (0.5x0.5) mortality rate of {disease_name} by age groups (25+, 60+, 80+) with national-level data only in 2040"
        output_path = "D:/CMIP6_data/Mortality/Output/National_mortality_baseline_2040/"
        output_file = f"{disease_name}.nc"
        ds = gen_output(disease_name, data, output_description)
        ds.to_netcdf(output_path + output_file)
        ds.close()

        print(f"DONE: {disease}, {disease_name}")


def main():
    subnational_output()
    combined_output()
    # national_output()


if __name__ == "__main__":
    main()
