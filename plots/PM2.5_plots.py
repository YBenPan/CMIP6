import os
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from netCDF4 import Dataset

ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585"]
years = [2015, 2020, 2025, 2030, 2035, 2040]


def line():
    pm25_path = "D:/CMIP6_data/PM2.5_annual"
    sns.set_theme()

    for i, ssp in enumerate(ssps):
        data = np.zeros(len(years))

        for j, year in enumerate(years):

            pm25_file = f"{pm25_path}/{ssp}/mmrpm2p5/*/*/annual_avg_{year}.nc"
            files = sorted(glob(pm25_file))
            all_conc = []
            for file in files:
                if "EC-Earth3" in file:
                    continue # Outlier: extremely large data
                if "MIROC" in file or "GISS" in file:
                    continue # Test skip
                # Import netCDF dataset
                wk = Dataset(file, "r")
                conc = wk["concpm2p5"][:]
                # unit: kilogram per m^3
                sum_conc = np.sum(conc) * 10 ** 9
                # unit: microgram per m^3
                all_conc.append(sum_conc)
            # Inter-model mean
            data[j] = np.mean(all_conc)

        df = pd.DataFrame({"year": years, "concentration": data})
        sns.lineplot(data=df, x="year", y="concentration", label=ssp, marker="o")
    plt.legend(bbox_to_anchor=(1.25, 1), loc="upper right", borderaxespad=0)
    plt.ylabel("concentration (μm / m^3)")
    plt.tight_layout()
    plt.show()




def main():
    line()


if __name__ == "__main__":
    main()
