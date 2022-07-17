import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lib.country import get_regions
from lib.helper import pct_change


def disease():
    base_path = "/home/ybenp/CMIP6_data/Mortality/Mortality Projections_2040"
    diseases = ["allcause", "copd", "ihd", "lc", "lri", "stroke", "diabetes"]
    regions, region_countries, region_countries_names = get_regions()

    for disease in diseases:
        base_file = os.path.join(base_path, f"{disease}_rate.csv")
        df = pd.read_csv(base_file, usecols=[4, 5, 18, 21])
        data = df.to_numpy()

        pop = data[:, 0:2]
        mort_rate = data[:, 2:4]
        mort_num = pop * mort_rate

        df = pd.DataFrame(
            columns=["Region", "Percent Change in Baseline Mortality Rate"]
        )

        for region, countries in zip(regions, region_countries):
            if region == "World":
                region_mort_num = np.nansum(mort_num, 0)
                region_pop = np.nansum(pop, 0)
            else:
                # Slice mortality numbers by regions
                region_mort_num = np.nansum(mort_num[countries], 0)
                region_pop = np.nansum(pop[countries], 0)
            region_mort_rate = region_mort_num / region_pop

            mort_num_pct_change = pct_change(region_mort_rate[0], region_mort_rate[1])
            df = df.append(
                {
                    "Region": region,
                    "Percent Change in Baseline Mortality Rate": mort_num_pct_change,
                },
                ignore_index=True,
            )

        # CSV Output
        output_path = "/home/ybenp/CMIP6_Images/Mortality/baseline"
        output_file = os.path.join(output_path, f"{disease}.csv")
        df.to_csv(output_file, index=False)

        print(f"Done: {disease}")


# def age():


def main():
    disease()


if __name__ == "__main__":
    main()
