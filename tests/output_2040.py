import os
import pandas as pd
import numpy as np

home_path = "D:/CMIP6_data/tmp/"
ssps = ["ssp585"]
# ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585"]
# Maps mortality ssp to population ssp
ssp_pop_mapping = {
    "ssp119": "SSP1",
    "ssp126": "SSP1",
    "ssp245": "SSP2",
    "ssp370": "SSP3",
    "ssp434": "SSP2",
    "ssp460": "SSP2",
    "ssp585": "SSP1"
}
pop_years = ["Pop_ssp1_2010"]
pop_file = "D:/CMIP6_data/population/national_pop/wcde_data_world.csv"
pop_data = pd.read_csv(pop_file, skiprows=8)

for ssp in ssps:

    pop_ssp = ssp_pop_mapping[ssp]
    pop_df = pop_data[pop_data["Scenario"] == pop_ssp]

    # Merge 95-99 and 100+
    pop_df.loc[pop_df["Age"] == "95--99", "Population"] = pop_df.loc[pop_df["Age"] == "100+", "Population"].values[0] + pop_df.loc[pop_df["Age"] == "95--99", "Population"].values[0]
    pop_df.loc[pop_df["Age"] == "95--99", "Age"] = "95+"
    pop_df = pop_df[pop_df["Age"] != "100+"]
    pop = pop_df["Population"].values

    for pop_year in pop_years:

        diseases_path = f"{home_path}/{ssp}/{pop_year}/CountryMortalityAbsolute/"
        diseases = os.listdir(diseases_path)

        for disease in diseases:
            mort_path = f"{diseases_path}/{disease}/"
            age_groups = os.listdir(mort_path)
            age_groups.remove("all_ages")
            age_groups = sorted(age_groups)

            data = np.zeros((len(age_groups)))
            for i, age_group in enumerate(age_groups):
                wk_file = f"{mort_path}/{age_group}/CESM2-WACCM6_r1i1p1f1_2040_GEMM.csv"
                wk = pd.read_csv(wk_file, header=None, skiprows=1, index_col=0)
                data[i] = wk.loc[-9999][1]
            rate = data / pop
            if not all(i <= j for i, j in zip(rate, rate[1:])):
                print(f"Error in {ssp}, {disease}")
                for i, j in zip(age_groups, rate):
                    print(i, j)
            else:
                print(f"Done: {ssp}, {disease} is OK.")

