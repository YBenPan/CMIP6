import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

####################################################################################################
#### MAPS SUBNATIONAL MORTALITY BASELINE TO A GRID
####################################################################################################

fraction_path = "D:/CMIP6_data/fraction/"
base_path = "D:/CMIP6_data/Mortality/Subnational Data_historical_5_years/"

diseases = ["Allcause", "COPD", "Dementia", "IschemicHeartDisease", "LowerRespiratoryInfections", "LungCancer",
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

national_baseline_path = "D:/CMIP6_data/Mortality/National Data_historical_5_years/"
subnational_nc_path = "D:/CMIP6_data/Mortality/Output/Subnational_mortality_baseline_2015/"

countries = ["brazil", "ethiopia", "indonesia", "iran", "japan", "kenya", "mexico", "pakistan", "south_africa", "uk", "us"]
country_long_names = ["Brazil", "Ethiopia", "Indonesia", "Iran", "Japan", "Kenya", "Mexico", "Pakistan", "South Africa", "the United Kingdom", "the United States"]
country_ids = [23, 58, 78, 79, 85, 88, 109, 127, 158, 181, 183]


def rename_helper(df):
    """Drop and rename countries"""
    to_be_dropped = ["American Samoa", "Bermuda", "Greenland", "Guam", "Montenegro", "Northern Mariana Islands",
                     "Palestine", "Taiwan (Province of China)", "Tokelau", "United States Virgin Islands",
                     "South Sudan"]
    to_be_renamed = {
        "Cabo Verde": "Cape Verde",
        "Eswatini": "Swaziland",
        "North Macedonia": "ThbMacedonia",  # ensure correct placement between Thailand and Timor-Leste
        "Puerto Rico": "ZzPuerto Rico",
    }

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


def gen_output(disease, data, output_description, subnatl_path=None, is_combined=False):
    """Generate output Dataset"""
    if disease in ["Allcause", "COPD", "Diabetes", "IschemicHeartDisease", "LowerRespiratoryInfections", "LungCancer", "NonCommunicableDiseases", "Stroke"]:
        age_25_29_mean = data[0][0]
        age_25_29_upper = data[0][1]
        age_25_29_lower = data[0][2]
        age_30_34_mean = data[1][0]
        age_30_34_upper = data[1][1]
        age_30_34_lower = data[1][2]
        age_35_39_mean = data[2][0]
        age_35_39_upper = data[2][1]
        age_35_39_lower = data[2][2]
        age_40_44_mean = data[3][0]
        age_40_44_upper = data[3][1]
        age_40_44_lower = data[3][2]
        age_45_49_mean = data[4][0]
        age_45_49_upper = data[4][1]
        age_45_49_lower = data[4][2]
        age_50_54_mean = data[5][0]
        age_50_54_upper = data[5][1]
        age_50_54_lower = data[5][2]
        age_55_59_mean = data[6][0]
        age_55_59_upper = data[6][1]
        age_55_59_lower = data[6][2]
        age_60_64_mean = data[7][0]
        age_60_64_upper = data[7][1]
        age_60_64_lower = data[7][2]
        age_65_69_mean = data[8][0]
        age_65_69_upper = data[8][1]
        age_65_69_lower = data[8][2]
        age_70_74_mean = data[9][0]
        age_70_74_upper = data[9][1]
        age_70_74_lower = data[9][2]
        age_75_79_mean = data[10][0]
        age_75_79_upper = data[10][1]
        age_75_79_lower = data[10][2]
        age_80_84_mean = data[11][0]
        age_80_84_upper = data[11][1]
        age_80_84_lower = data[11][2]
        age_85_89_mean = data[12][0]
        age_85_89_upper = data[12][1]
        age_85_89_lower = data[12][2]
        age_90_94_mean = data[13][0]
        age_90_94_upper = data[13][1]
        age_90_94_lower = data[13][2]
        post95_mean = data[14][0]
        post95_upper = data[14][1]
        post95_lower = data[14][2]

        if is_combined:
            # Add in subnational data
            for country in countries:
                subnatl_file = f"{country}_{disease}.nc"
                try:
                    subnatl_wk = xr.open_dataset(subnatl_path + subnatl_file)
                    subnatl_data = subnatl_wk.data_vars
                except:
                    print(f"Error importing {disease} {country} subnational data")
                    continue
                age_25_29_mean += subnatl_data["age_25_29_mean"].values
                age_25_29_upper += subnatl_data["age_25_29_upper"].values
                age_25_29_lower += subnatl_data["age_25_29_lower"].values
                age_30_34_mean += subnatl_data["age_30_34_mean"].values
                age_30_34_upper += subnatl_data["age_30_34_upper"].values
                age_30_34_lower += subnatl_data["age_30_34_lower"].values
                age_35_39_mean += subnatl_data["age_35_39_mean"].values
                age_35_39_upper += subnatl_data["age_35_39_upper"].values
                age_35_39_lower += subnatl_data["age_35_39_lower"].values
                age_40_44_mean += subnatl_data["age_40_44_mean"].values
                age_40_44_upper += subnatl_data["age_40_44_upper"].values
                age_40_44_lower += subnatl_data["age_40_44_lower"].values
                age_45_49_mean += subnatl_data["age_45_49_mean"].values
                age_45_49_upper += subnatl_data["age_45_49_upper"].values
                age_45_49_lower += subnatl_data["age_45_49_lower"].values
                age_50_54_mean += subnatl_data["age_50_54_mean"].values
                age_50_54_upper += subnatl_data["age_50_54_upper"].values
                age_50_54_lower += subnatl_data["age_50_54_lower"].values
                age_55_59_mean += subnatl_data["age_55_59_mean"].values
                age_55_59_upper += subnatl_data["age_55_59_upper"].values
                age_55_59_lower += subnatl_data["age_55_59_lower"].values
                age_60_64_mean += subnatl_data["age_60_64_mean"].values
                age_60_64_upper += subnatl_data["age_60_64_upper"].values
                age_60_64_lower += subnatl_data["age_60_64_lower"].values
                age_65_69_mean += subnatl_data["age_65_69_mean"].values
                age_65_69_upper += subnatl_data["age_65_69_upper"].values
                age_65_69_lower += subnatl_data["age_65_69_lower"].values
                age_70_74_mean += subnatl_data["age_70_74_mean"].values
                age_70_74_upper += subnatl_data["age_70_74_upper"].values
                age_70_74_lower += subnatl_data["age_70_74_lower"].values
                age_75_79_mean += subnatl_data["age_75_79_mean"].values
                age_75_79_upper += subnatl_data["age_75_79_upper"].values
                age_75_79_lower += subnatl_data["age_75_79_lower"].values
                age_80_84_mean += subnatl_data["age_80_84_mean"].values
                age_80_84_upper += subnatl_data["age_80_84_upper"].values
                age_80_84_lower += subnatl_data["age_80_84_lower"].values
                age_85_89_mean += subnatl_data["age_85_89_mean"].values
                age_85_89_upper += subnatl_data["age_85_89_upper"].values
                age_85_89_lower += subnatl_data["age_85_89_lower"].values
                age_90_94_mean += subnatl_data["age_90_94_mean"].values
                age_90_94_upper += subnatl_data["age_90_94_upper"].values
                age_90_94_lower += subnatl_data["age_90_94_lower"].values
                post95_mean += subnatl_data["post95_mean"].values
                post95_upper += subnatl_data["post95_upper"].values
                post95_lower += subnatl_data["post95_lower"].values

        ds = xr.Dataset(
            data_vars=dict(
                age_25_29_mean=(["lat", "lon"], age_25_29_mean),
                age_25_29_upper=(["lat", "lon"], age_25_29_upper),
                age_25_29_lower=(["lat", "lon"], age_25_29_lower),
                age_30_34_mean=(["lat", "lon"], age_30_34_mean),
                age_30_34_upper=(["lat", "lon"], age_30_34_upper),
                age_30_34_lower=(["lat", "lon"], age_30_34_lower),
                age_35_39_mean=(["lat", "lon"], age_35_39_mean),
                age_35_39_upper=(["lat", "lon"], age_35_39_upper),
                age_35_39_lower=(["lat", "lon"], age_35_39_lower),
                age_40_44_mean=(["lat", "lon"], age_40_44_mean),
                age_40_44_upper=(["lat", "lon"], age_40_44_upper),
                age_40_44_lower=(["lat", "lon"], age_40_44_lower),
                age_45_49_mean=(["lat", "lon"], age_45_49_mean),
                age_45_49_upper=(["lat", "lon"], age_45_49_upper),
                age_45_49_lower=(["lat", "lon"], age_45_49_lower),
                age_50_54_mean=(["lat", "lon"], age_50_54_mean),
                age_50_54_upper=(["lat", "lon"], age_50_54_upper),
                age_50_54_lower=(["lat", "lon"], age_50_54_lower),
                age_55_59_mean=(["lat", "lon"], age_55_59_mean),
                age_55_59_upper=(["lat", "lon"], age_55_59_upper),
                age_55_59_lower=(["lat", "lon"], age_55_59_lower),
                age_60_64_mean=(["lat", "lon"], age_60_64_mean),
                age_60_64_upper=(["lat", "lon"], age_60_64_upper),
                age_60_64_lower=(["lat", "lon"], age_60_64_lower),
                age_65_69_mean=(["lat", "lon"], age_65_69_mean),
                age_65_69_upper=(["lat", "lon"], age_65_69_upper),
                age_65_69_lower=(["lat", "lon"], age_65_69_lower),
                age_70_74_mean=(["lat", "lon"], age_70_74_mean),
                age_70_74_upper=(["lat", "lon"], age_70_74_upper),
                age_70_74_lower=(["lat", "lon"], age_70_74_lower),
                age_75_79_mean=(["lat", "lon"], age_75_79_mean),
                age_75_79_upper=(["lat", "lon"], age_75_79_upper),
                age_75_79_lower=(["lat", "lon"], age_75_79_lower),
                age_80_84_mean=(["lat", "lon"], age_80_84_mean),
                age_80_84_upper=(["lat", "lon"], age_80_84_upper),
                age_80_84_lower=(["lat", "lon"], age_80_84_lower),
                age_85_89_mean=(["lat", "lon"], age_85_89_mean),
                age_85_89_upper=(["lat", "lon"], age_85_89_upper),
                age_85_89_lower=(["lat", "lon"], age_85_89_lower),
                age_90_94_mean=(["lat", "lon"], age_90_94_mean),
                age_90_94_upper=(["lat", "lon"], age_90_94_upper),
                age_90_94_lower=(["lat", "lon"], age_90_94_lower),
                post95_mean=(["lat", "lon"], post95_mean),
                post95_upper=(["lat", "lon"], post95_upper),
                post95_lower=(["lat", "lon"], post95_lower),
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
        age_40_44_mean = data[3][0]
        age_40_44_upper = data[3][1]
        age_40_44_lower = data[3][2]
        age_45_49_mean = data[4][0]
        age_45_49_upper = data[4][1]
        age_45_49_lower = data[4][2]
        age_50_54_mean = data[5][0]
        age_50_54_upper = data[5][1]
        age_50_54_lower = data[5][2]
        age_55_59_mean = data[6][0]
        age_55_59_upper = data[6][1]
        age_55_59_lower = data[6][2]
        age_60_64_mean = data[7][0]
        age_60_64_upper = data[7][1]
        age_60_64_lower = data[7][2]
        age_65_69_mean = data[8][0]
        age_65_69_upper = data[8][1]
        age_65_69_lower = data[8][2]
        age_70_74_mean = data[9][0]
        age_70_74_upper = data[9][1]
        age_70_74_lower = data[9][2]
        age_75_79_mean = data[10][0]
        age_75_79_upper = data[10][1]
        age_75_79_lower = data[10][2]
        age_80_84_mean = data[11][0]
        age_80_84_upper = data[11][1]
        age_80_84_lower = data[11][2]
        age_85_89_mean = data[12][0]
        age_85_89_upper = data[12][1]
        age_85_89_lower = data[12][2]
        age_90_94_mean = data[13][0]
        age_90_94_upper = data[13][1]
        age_90_94_lower = data[13][2]
        post95_mean = data[14][0]
        post95_upper = data[14][1]
        post95_lower = data[14][2]

        if is_combined:
            # Add in subnational data
            for country in countries:
                subnatl_file = f"{country}_{disease}.nc"
                try:
                    subnatl_wk = xr.open_dataset(subnatl_path + subnatl_file)
                    subnatl_data = subnatl_wk.data_vars
                except:
                    print(f"Error importing {disease} {country} subnational data")
                    continue
                age_40_44_mean += subnatl_data["age_40_44_mean"].values
                age_40_44_upper += subnatl_data["age_40_44_upper"].values
                age_40_44_lower += subnatl_data["age_40_44_lower"].values
                age_45_49_mean += subnatl_data["age_45_49_mean"].values
                age_45_49_upper += subnatl_data["age_45_49_upper"].values
                age_45_49_lower += subnatl_data["age_45_49_lower"].values
                age_50_54_mean += subnatl_data["age_50_54_mean"].values
                age_50_54_upper += subnatl_data["age_50_54_upper"].values
                age_50_54_lower += subnatl_data["age_50_54_lower"].values
                age_55_59_mean += subnatl_data["age_55_59_mean"].values
                age_55_59_upper += subnatl_data["age_55_59_upper"].values
                age_55_59_lower += subnatl_data["age_55_59_lower"].values
                age_60_64_mean += subnatl_data["age_60_64_mean"].values
                age_60_64_upper += subnatl_data["age_60_64_upper"].values
                age_60_64_lower += subnatl_data["age_60_64_lower"].values
                age_65_69_mean += subnatl_data["age_65_69_mean"].values
                age_65_69_upper += subnatl_data["age_65_69_upper"].values
                age_65_69_lower += subnatl_data["age_65_69_lower"].values
                age_70_74_mean += subnatl_data["age_70_74_mean"].values
                age_70_74_upper += subnatl_data["age_70_74_upper"].values
                age_70_74_lower += subnatl_data["age_70_74_lower"].values
                age_75_79_mean += subnatl_data["age_75_79_mean"].values
                age_75_79_upper += subnatl_data["age_75_79_upper"].values
                age_75_79_lower += subnatl_data["age_75_79_lower"].values
                age_80_84_mean += subnatl_data["age_80_84_mean"].values
                age_80_84_upper += subnatl_data["age_80_84_upper"].values
                age_80_84_lower += subnatl_data["age_80_84_lower"].values
                age_85_89_mean += subnatl_data["age_85_89_mean"].values
                age_85_89_upper += subnatl_data["age_85_89_upper"].values
                age_85_89_lower += subnatl_data["age_85_89_lower"].values
                age_90_94_mean += subnatl_data["age_90_94_mean"].values
                age_90_94_upper += subnatl_data["age_90_94_upper"].values
                age_90_94_lower += subnatl_data["age_90_94_lower"].values
                post95_mean += subnatl_data["post95_mean"].values
                post95_upper += subnatl_data["post95_upper"].values
                post95_lower += subnatl_data["post95_lower"].values

        ds = xr.Dataset(
            data_vars=dict(
                age_40_44_mean=(["lat", "lon"], age_40_44_mean),
                age_40_44_upper=(["lat", "lon"], age_40_44_upper),
                age_40_44_lower=(["lat", "lon"], age_40_44_lower),
                age_45_49_mean=(["lat", "lon"], age_45_49_mean),
                age_45_49_upper=(["lat", "lon"], age_45_49_upper),
                age_45_49_lower=(["lat", "lon"], age_45_49_lower),
                age_50_54_mean=(["lat", "lon"], age_50_54_mean),
                age_50_54_upper=(["lat", "lon"], age_50_54_upper),
                age_50_54_lower=(["lat", "lon"], age_50_54_lower),
                age_55_59_mean=(["lat", "lon"], age_55_59_mean),
                age_55_59_upper=(["lat", "lon"], age_55_59_upper),
                age_55_59_lower=(["lat", "lon"], age_55_59_lower),
                age_60_64_mean=(["lat", "lon"], age_60_64_mean),
                age_60_64_upper=(["lat", "lon"], age_60_64_upper),
                age_60_64_lower=(["lat", "lon"], age_60_64_lower),
                age_65_69_mean=(["lat", "lon"], age_65_69_mean),
                age_65_69_upper=(["lat", "lon"], age_65_69_upper),
                age_65_69_lower=(["lat", "lon"], age_65_69_lower),
                age_70_74_mean=(["lat", "lon"], age_70_74_mean),
                age_70_74_upper=(["lat", "lon"], age_70_74_upper),
                age_70_74_lower=(["lat", "lon"], age_70_74_lower),
                age_75_79_mean=(["lat", "lon"], age_75_79_mean),
                age_75_79_upper=(["lat", "lon"], age_75_79_upper),
                age_75_79_lower=(["lat", "lon"], age_75_79_lower),
                age_80_84_mean=(["lat", "lon"], age_80_84_mean),
                age_80_84_upper=(["lat", "lon"], age_80_84_upper),
                age_80_84_lower=(["lat", "lon"], age_80_84_lower),
                age_85_89_mean=(["lat", "lon"], age_85_89_mean),
                age_85_89_upper=(["lat", "lon"], age_85_89_upper),
                age_85_89_lower=(["lat", "lon"], age_85_89_lower),
                age_90_94_mean=(["lat", "lon"], age_90_94_mean),
                age_90_94_upper=(["lat", "lon"], age_90_94_upper),
                age_90_94_lower=(["lat", "lon"], age_90_94_lower),
                post95_mean=(["lat", "lon"], post95_mean),
                post95_upper=(["lat", "lon"], post95_upper),
                post95_lower=(["lat", "lon"], post95_lower),
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
        raise Exception(f"{disease} is an undefined disease")

    return ds


def combined_output():
    """Outputs gridded mortality baseline for all countries with subnational data in select countries"""

    output_path = "D:/CMIP6_data/Mortality/Output/Combined_mortality_baseline_2015/"

    for disease in diseases:
        national_baseline_file = f"{disease}.csv"
        try:
            # location name, age_name, metric_name, year, val, upper, lower
            data = pd.read_csv(national_baseline_path + national_baseline_file, usecols=[3, 7, 11, 12, 13, 14, 15])
        except:
            print(f"Error importing {disease}")
            continue
        data = rename_helper(data)
        wk = data[(data["year"] == 2015) & (data["metric_name"] == "Rate")]
        wk = wk.drop(columns=["metric_name", "year"])
        wk = wk.sort_values(["location_name", "age_name"])

        age_groups = sorted(list(set(wk["age_name"].values)))
        data = np.zeros((15, len(data_types), len(latitude), len(longitude)))

        # Loop through countries
        for j in np.arange(0, 193):

            # Skip country if we have subnational data of it
            if j in country_ids:
                continue

            for k, age_group in enumerate(age_groups):

                for p, data_type in enumerate(data_types):

                    # retrieve mortality corresponding to correct age group and state
                    try:
                        tmp = wk.iloc[np.arange(j * len(age_groups), (j + 1) * len(age_groups))]
                        mort = tmp[(tmp["age_name"] == age_group)][data_type].values[0]
                    except IndexError:
                        print("Error importing 2015 national baseline files", age_group, j)
                        break

                    if disease != "Dementia":
                        data[k, p, :, :] += fractionCountry[j, :, :] * mort
                    else:
                        data[k + 3, p, :, :] += fractionCountry[j, :, :] * mort

        national_output_path = "D:/CMIP6_data/Mortality/Output/Combined_mortality_baseline_2015/"
        output_file = f"{disease}.nc"
        output_description = f"Gridded (0.5x0.5) mortality rate of {disease} by age groups in 2015, with " \
                             f"subnational-level data in Brazil, Indonesia, Japan, Kenya, " \
                             f"Mexico, UK, and US and national-level data"

        ds = gen_output(disease, data, output_description, subnational_nc_path, is_combined=True)
        ds.to_netcdf(output_path + output_file)
        ds.close()

        print(f"DONE: {disease}")


def national_output():
    """Outputs gridded mortality baseline for all countries"""

    output_path = "D:/CMIP6_data/Mortality/Output/National_mortality_baseline_2015/"

    for disease in diseases:
        national_baseline_file = f"{disease}.csv"
        try:
            data = pd.read_csv(national_baseline_path + national_baseline_file)
        except:
            print(f"Error importing {disease}")
            continue
        data = rename_helper(data)
        wk = data[(data["year"] == 2015) & (data["metric_name"] == "Rate")]
        wk = wk.drop(columns=["metric_name", "year"])
        wk = wk.sort_values(["location_name", "age_name"])

        age_groups = sorted(list(set(wk["age_name"].values)))
        data = np.zeros((15, len(data_types), len(latitude), len(longitude)))

        # Loop through countries
        for j in np.arange(0, 193):

            for k, age_group in enumerate(age_groups):

                for p, data_type in enumerate(data_types):

                    # retrieve mortality corresponding to correct age group and state
                    try:
                        tmp = wk.iloc[np.arange(j * len(age_groups), (j + 1) * len(age_groups))]
                        mort = tmp[(tmp["age_name"] == age_group)][data_type].values[0]
                        # print(j, age_group, data_type, mort)
                        # input()
                    except IndexError:
                        print(age_group, j)
                        break

                    if disease != "Dementia":
                        data[k, p, :, :] += fractionCountry[j, :, :] * mort
                    else:
                        data[k + 3, p, :, :] += fractionCountry[j, :, :] * mort

        output_file = f"{disease}.nc"
        output_description = f"Gridded (0.5x0.5) mortality rate of {disease} by age groups in 2015, with national-level data only"

        ds = gen_output(disease, data, output_description, subnational_nc_path)
        ds.to_netcdf(output_path + output_file)
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
            base_file = f"{disease}.csv"
            try:
                data = pd.read_csv(base_path + base_file, usecols=[3, 7, 11, 12, 13, 14, 15])
            except:
                print(f"Error importing {disease}")
                continue
            wk = data[(data["location_name"].isin(states)) & (data["year"] == 2015) & (data["metric_name"] == "Rate")]
            wk = wk.drop(columns=["metric_name", "year"])
            wk = wk.sort_values(["location_name", "age_name"])

            age_groups = sorted(list(set(wk["age_name"].values)))
            data = np.zeros((15, len(data_types), len(latitude), len(longitude)))

            for j, state in enumerate(states):

                if country_long_name == "Pakistan" and (state == "Azad Jammu & Kashmir" or state == "Gilgit-Baltistan"):
                    continue

                for k, age_group in enumerate(age_groups):

                    for p, data_type in enumerate(data_types):

                        # retrieve mortality corresponding to correct age group and state
                        try:
                            mort = wk[(wk["age_name"] == age_group) & (wk["location_name"] == state)][data_type].values[
                                0]
                        except IndexError:
                            print(age_group, state)
                            break

                        if disease != "Dementia":
                            data[k, p, :, :] += fractionState[j, :, :] * mort
                        else:
                            data[k + 3, p, :, :] += fractionState[j, :, :] * mort

            output_description = f"Gridded (0.5x0.5) mortality rate of {disease} in {country_long_name} by age groups " \
                                 f"in 2015 "
            output_file = f"{country}_{disease}.nc"

            ds = gen_output(disease, data, output_description, subnational_nc_path)
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
    combined_output()
    # national_output()


if __name__ == "__main__":
    main()
