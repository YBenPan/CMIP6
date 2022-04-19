import numpy as np
import pandas as pd
import xarray as xr
from mort_2015 import rename_helper

####################################################################################################
#### CREATE AGE GROUP RATES IN MORTALITY BASELINE (FOR POST60 etc.)
####################################################################################################

national_baseline_path = "D:/CMIP6_data/Mortality/National Data_historical/"
pop_path = "D:/CMIP6_data/population/national_pop/"
pop_file = "SSP1.nc"

diseases = ["copd", "dementia", "ihd", "lri", "lc", "ncd", "stroke", "diabetes"]
disease_names = ["COPD", "Dementia", "IschemicHeartDisease", "LowerRespiratoryInfections", "LungCancer",
                 "NonCommunicableDiseases", "Stroke", "Diabetes"]
data_types_2015 = ["val", "upper", "lower"]


def national():
    output_path = "D:/CMIP6_data/Mortality/National Data_historical_with_post60/"

    # Import population data
    pop_ds = xr.open_dataset(pop_path + pop_file)

    for disease_name in disease_names:

        # Import national baseline
        national_baseline_file = f"{disease_name}.csv"
        try:
            natl_2015 = pd.read_csv(national_baseline_path + national_baseline_file, usecols=[3, 7, 11, 12, 13, 14, 15])
        except:
            print("Error importing 2015 national baseline files", disease_name, national_baseline_path + national_baseline_file)
            continue
        natl_2015 = rename_helper(natl_2015)
        natl_2015 = natl_2015[(natl_2015["year"] == 2015)]
        natl_2015 = natl_2015.sort_values(['location_name', 'age_name'])

        all_age_groups = sorted(list(set(natl_2015["age_name"].values)))

        data = []

        for i in range(0, 193):

            wk = natl_2015.iloc[np.arange(i * (2 * len(all_age_groups)), (i + 1) * (2 * len(all_age_groups)))]

            if disease_name in ["COPD", "LowerRespiratoryInfections", "LungCancer", "Diabetes"]:
                post25_num = wk[(wk["age_name"] == "25 plus") & (wk["metric_name"] == "Number")].to_numpy()[0].tolist()
                post25_rate = wk[(wk["age_name"] == "25 plus") & (wk["metric_name"] == "Rate")].to_numpy()[0].tolist()
                data.append(post25_num)
                data.append(post25_rate)
            elif disease_name in ["IschemicHeartDisease", "NonCommunicableDiseases", "Stroke"]:
                # Set up population data and calculation for rate
                pop_data = pop_ds.sel(Year=2015, Age="60+")["Population"].values
                pop_data *= (10 ** 3)
                pop = pop_data[i]
                no_pop = (pop == 0)

                age_groups = ["60 to 64", "65 to 69", "70 to 74", "75 to 79", "80 plus"]
                num = np.zeros((len(data_types_2015)))
                rate = np.zeros((len(data_types_2015)))

                post25_num = wk[(wk["age_name"] == "25 plus") & (wk["metric_name"] == "Number")].to_numpy()[0].tolist()
                post25_rate = wk[(wk["age_name"] == "25 plus") & (wk["metric_name"] == "Rate")].to_numpy()[0].tolist()
                post80_num = wk[(wk["age_name"] == "80 plus") & (wk["metric_name"] == "Number")].to_numpy()[0].tolist()
                post80_rate = wk[(wk["age_name"] == "80 plus") & (wk["metric_name"] == "Rate")].to_numpy()[0].tolist()
                post60_num = wk.iloc[0].tolist()
                post60_num[1] = "60 plus"
                post60_rate = wk.iloc[1].tolist()
                post60_rate[1] = "60 plus"

                for j, data_type in enumerate(data_types_2015):

                    for k, age_group in enumerate(age_groups):

                        # Retrieve row according to age group and metric
                        mort_num = wk[(wk["age_name"] == age_group) & (wk["metric_name"] == "Number")][data_type].values[0]
                        mort_rate = wk[(wk["age_name"] == age_group) & (wk["metric_name"] == "Rate")][data_type].values[0]

                        # If population data was not imported (in added countries like Monaco), then calculate it here
                        if no_pop and j == 0:
                            pop += mort_num / mort_rate * (10 ** 5)
                        num[j] += mort_num

                rate[:] = num[:] / pop * (10 ** 5)
                post60_num[-3:] = num
                post60_rate[-3:] = rate

                data.append(post25_num)
                data.append(post25_rate)
                data.append(post60_num)
                data.append(post60_rate)
                data.append(post80_num)
                data.append(post80_rate)
            elif disease_name == "Dementia":
                pop_data = pop_ds.sel(Year=2015, Age="65+")["Population"].values
                pop_data *= (10 ** 3)
                pop = pop_data[i]
                no_pop = (pop == 0)

                age_groups = ["65 to 74", "75 plus"]
                num = np.zeros((len(data_types_2015)))
                rate = np.zeros((len(data_types_2015)))

                post75_num = wk[(wk["age_name"] == "75 plus") & (wk["metric_name"] == "Number")].to_numpy()[0].tolist()
                post75_rate = wk[(wk["age_name"] == "75 plus") & (wk["metric_name"] == "Rate")].to_numpy()[0].tolist()
                post65_num = wk.iloc[0].tolist()
                post65_num[1] = "65 plus"
                post65_rate = wk.iloc[1].tolist()
                post65_rate[1] = "65 plus"

                for j, data_type in enumerate(data_types_2015):

                    for k, age_group in enumerate(age_groups):

                        mort_num = wk[(wk["age_name"] == age_group) & (wk["metric_name"] == "Number")][data_type].values[0]
                        mort_rate = wk[(wk["age_name"] == age_group) & (wk["metric_name"] == "Rate")][data_type].values[0]
                        if no_pop and j == 0:
                            pop += mort_num / mort_rate * (10 ** 5)
                        num[j] += mort_num

                rate[:] = num[:] / pop * (10 ** 5)
                post65_num[-3:] = num
                post65_rate[-3:] = rate

                data.append(post75_num)
                data.append(post75_rate)
                data.append(post65_num)
                data.append(post65_rate)
            else:
                raise Exception(f"Undefined disease {disease_name}")

        # Output
        df = pd.DataFrame(data, columns=natl_2015.columns.tolist())
        # print(df.head())
        output_file = f"{disease_name}.csv"
        df.to_csv(output_path + output_file, index=False)
        print(f"DONE: {disease_name}")


def main():
    national()


if __name__ == "__main__":
    main()
