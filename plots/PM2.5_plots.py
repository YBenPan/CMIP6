import os
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from china import get_country_mask, get_pop, get_grid_area, mean

# ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585"]
ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]
years = [2015, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
years = np.arange(2015, 2105, 10)

# Get country mask for the United States
fractionCountry = get_country_mask(country=183)

# Get grid areas for area weighted mean
grid_area, tot_area = get_grid_area(fractionCountry)

# Get population for population weighted mean
pop, tot_pop = get_pop(fractionCountry)


def line():
    pm25_path = "D:/CMIP6_data/PM2.5_annual"
    sns.set_theme()

    for i, ssp in enumerate(ssps):
        data = np.zeros(len(years))

        for j, year in enumerate(years):

            models = os.listdir(f"{pm25_path}/{ssp}/mmrpm2p5")
            all_conc, all_awm, all_pwm = mean(models, ssp, year, fractionCountry, grid_area, tot_area, pop, tot_pop)

            # Multi-model mean
            print(f"{ssp} {year} inter-model PWM: {np.round(all_pwm, 2)}, AWM: {np.round(all_awm, 2)}")
            data[j] = all_awm

        df = pd.DataFrame({"year": years, "concentration": data})
        sns.lineplot(data=df, x="year", y="concentration", label=ssp, marker="o")
        print(f"Done: {ssp}")
    plt.legend(bbox_to_anchor=(1.25, 1), loc="upper right", borderaxespad=0)
    plt.ylabel("PM2.5 concentration, weighted by area (μg / m^3)")
    plt.tight_layout()
    # plt.show()
    plt.savefig("D:/CMIP6_Images/PM2.5/us_awm.png")


def main():
    line()


if __name__ == "__main__":
    main()
