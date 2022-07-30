import numpy as np
import pandas as pd
import xarray as xr
import os
from netCDF4 import Dataset
from mort_2015 import rename_helper, gen_output

####################################################################################################
#### MAPS SUBNATIONAL MORTALITY BASELINE TO A GRID
####################################################################################################

fraction_path = "D:/CMIP6_data/fraction/"

diseases = ["Allcause", "COPD", "Dementia", "T2D", "IHD", "LRI", "LC", "NCD", "Stroke"]
disease_names = [
    "Allcause",
    "COPD",
    "Dementia",
    "Diabetes",
    "IschemicHeartDisease",
    "LowerRespiratoryInfections",
    "LungCancer",
    "NonCommunicableDiseases",
    "Stroke",
]
data_types_2015 = ["val", "upper", "lower"]
data_types_2040 = ["Value", "Upper bound", "Lower bound"]
age_groups = [
    "25-29 years",
    "30-34 years",
    "35-39 years",
    "40-44 years",
    "45-49 years",
    "50-54 years",
    "55-59 years",
    "60-64 years",
    "65-69 years",
    "70-74 years",
    "75-79 years",
    "80-84 years",
    "85-89 years",
    "90-94 years",
    "95+ years",
]


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

national_baseline_path = "D:/CMIP6_data/Mortality/National Data_historical_5_years/"
national_projection_path = "D:/CMIP6_data/Mortality/Age-specific Mortality Projections_2040/"
subnational_nc_path = (
    "D:/CMIP6_data/Mortality/Output/Subnational_mortality_baseline_2040/"
)

# Countries with subnational data
countries = [
    "brazil",
    "ethiopia",
    "indonesia",
    "iran",
    "japan",
    "kenya",
    "mexico",
    "pakistan",
    "south_africa",
    "uk",
    "us",
]
country_long_names = [
    "Brazil",
    "Ethiopia",
    "Indonesia",
    "Iran",
    "Japan",
    "Kenya",
    "Mexico",
    "Pakistan",
    "South Africa",
    "the United Kingdom",
    "the United States",
]
country_ids = [23, 58, 78, 79, 85, 88, 109, 127, 158, 181, 183]


def combined_output():
    """Outputs gridded mortality baseline for all countries with subnational data in select countries in 2040"""

    output_path = "D:/CMIP6_data/Mortality/Output/Combined_mortality_baseline_2040/"

    for disease, disease_name in zip(diseases, disease_names):

        # Import national baseline data from 2015
        national_baseline_file = f"{disease_name}.csv"
        try:
            natl_2015 = pd.read_csv(
                national_baseline_path + national_baseline_file,
                usecols=[3, 7, 11, 12, 13, 14, 15],
            )
        except:
            print(
                "Error importing 2015 national baseline files",
                disease,
                national_baseline_path + national_baseline_file,
            )
            continue
        natl_2015 = rename_helper(natl_2015)
        natl_2015 = natl_2015[
            (natl_2015["year"] == 2015) & (natl_2015["metric_name"] == "Rate")
        ]
        natl_2015 = natl_2015.drop(columns=["metric_name", "year"])
        natl_2015 = natl_2015.sort_values(["location_name", "age_name"])
        age_groups = sorted(list(set(natl_2015["age_name"].values)))

        # Import national baseline projection from 2040
        national_projection_file = f"{disease}_rate.csv"
        try:
            natl_2040 = pd.read_csv(
                national_projection_path + national_projection_file,
                usecols=[2, 18, 19, 20, 21, 22, 23],
            )
        except:
            print(
                "Error importing 2040 baseline projection files",
                disease,
                national_projection_path + national_projection_file,
            )
            continue
        natl_2040 = natl_2040.fillna(0)

        data = np.zeros((15, len(data_types_2040), len(latitude), len(longitude)))

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
                    if (
                        natl_2015_val == 0
                        or np.isnan(natl_2015_val)
                        or natl_2040_val == 0
                        or np.isnan(natl_2040_val)
                    ):
                        ratio = 0
                    else:
                        ratio = natl_2040_val / natl_2015_val

                    # retrieve mortality corresponding to correct age group and state
                    try:
                        tmp = natl_2015.iloc[
                            np.arange(j * len(age_groups), (j + 1) * len(age_groups))
                        ]
                        mort = tmp[(tmp["age_name"] == age_group)][
                            data_type_2015
                        ].values[0]
                    except IndexError:
                        print(
                            "Error importing 2015 national baseline files", age_group, j
                        )
                        break

                    if disease_name != "Dementia":
                        data[k, p, :, :] += fractionCountry[j, :, :] * mort * ratio
                    else:
                        data[k + 3, p, :, :] += fractionCountry[j, :, :] * mort * ratio

        output_file = f"{disease_name}.nc"
        output_description = (
            f"Gridded (0.5x0.5) mortality rate of {disease} by age groups in 2040, with "
            f"subnational-level data in Brazil, Ethiopia, Indonesia, Iran, Japan, Kenya, "
            f"Mexico, Pakistan, South Africa, UK, and US and national-level data"
        )
        ds = gen_output(
            disease_name,
            data,
            output_description,
            subnational_nc_path,
            is_combined=True,
        )
        ds.to_netcdf(output_path + output_file)
        ds.close()

        print(f"DONE: {disease_name}")


def subnational_output():
    """Outputs gridded mortality baseline for only countries with subnational data (US, UK, etc.)"""
    output_path = "D:/CMIP6_data/Mortality/Output/Subnational_mortality_baseline_2040/"
    national_projection_path = "D:/CMIP6_data/Mortality/Mortality Projections_2040/"

    for country, country_long_name, country_id in zip(
        countries, country_long_names, country_ids
    ):

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
                natl_2040 = pd.read_csv(
                    national_projection_path + national_projection_file,
                    usecols=[2, 18, 19, 20, 21, 22, 23],
                )
            except:
                print(
                    "Error importing 2040 baseline projection files",
                    disease,
                    national_projection_path + national_projection_file,
                )
                continue
            natl_2040 = natl_2040.fillna(0)

            # import mortality baseline
            subnatl_baseline_path = (
                "D:/CMIP6_data/Mortality/Subnational Data_historical_5_years/"
            )
            subnatl_baseline_file = f"{disease_name}.csv"
            # location name, age_name, metric_name, year, val
            subnatl_2015 = pd.read_csv(
                subnatl_baseline_path + subnatl_baseline_file,
                usecols=[3, 7, 11, 12, 13, 14, 15],
            )
            subnatl_2015 = subnatl_2015[
                (subnatl_2015["location_name"].isin(states))
                & (subnatl_2015["year"] == 2015)
                & (subnatl_2015["metric_name"] == "Rate")
            ]
            subnatl_2015 = subnatl_2015.drop(columns=["metric_name", "year"])
            subnatl_2015 = subnatl_2015.sort_values(["location_name", "age_name"])

            age_groups = sorted(list(set(subnatl_2015["age_name"].values)))
            data = np.zeros((15, len(data_types_2040), len(latitude), len(longitude)))

            for j, state in enumerate(states):

                if country_long_name == "Pakistan" and (
                    state == "Azad Jammu & Kashmir" or state == "Gilgit-Baltistan"
                ):
                    continue

                for k, age_group in enumerate(age_groups):

                    for p, data_type_2015 in enumerate(data_types_2015):

                        data_type_2040 = data_types_2040[p]
                        # Calculate ratio to evolve age group data from 2015 to 2040
                        natl_2015_val = natl_2040.iloc[country_id][data_type_2040[0]]
                        natl_2040_val = natl_2040.iloc[country_id][data_type_2040[1]]
                        # print(j, age_group, data_type_2015, natl_2015_val, natl_2040_val)
                        if (
                            natl_2015_val == 0
                            or np.isnan(natl_2015_val)
                            or natl_2040_val == 0
                            or np.isnan(natl_2040_val)
                        ):
                            ratio = 0
                        else:
                            ratio = natl_2040_val / natl_2015_val

                        # Retrieve mortality corresponding to correct age group and state
                        try:
                            mort = subnatl_2015[
                                (subnatl_2015["age_name"] == age_group)
                                & (subnatl_2015["location_name"] == state)
                            ][data_type_2015].values[0]
                        except IndexError:
                            print(age_group, state)
                            break

                        if disease_name != "Dementia":
                            data[k, p, :, :] += fractionState[j, :, :] * mort * ratio
                        else:
                            data[k + 3, p, :, :] += (
                                fractionState[j, :, :] * mort * ratio
                            )

            output_file = f"{country}_{disease_name}.nc"
            output_description = (
                f"Gridded (0.5x0.5) mortality rate of {disease} in {country_long_name} by age groups"
                f" in 2040 "
            )
            ds = gen_output(
                disease_name, data, output_description, subnatl_baseline_path
            )
            ds.to_netcdf(output_path + output_file)
            ds.close()

            print(f"DONE: {country_long_name}, {disease_name}")


def national_output():
    """Outputs gridded mortality baseline for all countries in 2040"""

    for disease, disease_name in zip(diseases, disease_names):

        data = np.zeros((15, len(data_types_2040), len(latitude), len(longitude)))

        # Loop through countries
        for j in np.arange(0, 193):

            for k, age_group in enumerate(age_groups):

                # Import national baseline projection from 2040
                national_projection_file = os.path.join(
                    national_projection_path, "2040", disease, f"{age_group}.csv"
                )
                natl_2040 = pd.read_csv(national_projection_file, usecols=[1, 7, 8, 9])
                natl_2040 = natl_2040.fillna(0)

                for p, data_type_2040 in enumerate(data_types_2040):

                    # Retrieve mortality corresponding to the correct age group and state
                    try:
                        tmp = natl_2040.iloc[j]
                        mort = tmp[data_type_2040]
                    except IndexError:
                        print(age_group, j)
                        break

                    data[k, p, :, :] += fractionCountry[j, :, :] * mort

        output_description = f"Gridded (0.5x0.5) mortality rate of {disease_name} by 5-year age groups with national-level data only in 2040"
        output_path = (
            "D:/CMIP6_data/Mortality/Output/Age_specific_National_mortality_baseline_2040/"
        )
        output_file = f"{disease_name}.nc"
        ds = gen_output(disease_name, data, output_description)
        ds.to_netcdf(output_path + output_file)
        ds.close()

        print(f"DONE: {disease}, {disease_name}")


def main():
    # subnational_output()
    # combined_output()
    national_output()


if __name__ == "__main__":
    main()
