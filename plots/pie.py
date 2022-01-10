from glob import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os

####################################################################################################
#### CREATE PIE CHARTS BASED ON COUNTRY MORTALITY DATA
####################################################################################################

####################################################################################################
#### USER INPUT:
#### WHERE ARE THE COUNTRY MORTALITY DATA?
base_path = "D:/CMIP6_data/Mortality/"
#### WHERE IS THE OUTPUT FOLDER?
output_path = "D:/CMIP6_Images/Mortality/pie/"
####################################################################################################

ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]
# ssps = ["ssp370"]

years = np.arange(2015, 2101)
year_bins = [
    np.arange(0, 5),  # 2015-2019
    np.arange(5, 15),   # 2020-2029
    np.arange(15, 25),  # 2030-2039
    np.arange(25, 35),  # 2040-2049
    np.arange(35, 45),  # 2050-2059
    np.arange(45, 55),  # 2060-2069
    np.arange(55, 65),  # 2070-2079
    np.arange(65, 75),  # 2080-2089
    np.arange(75, 85),  # 2090-2099
    [85]                # 2100
]

year_bin_names = [str(x) + '0s' for x in np.arange(201, 211)]

diseases = ["Allcause", "COPD", "IHD", "LC", "LRI", "Stroke", "T2D"]
disease_labels = ["Others", "COPD", "IHD", "LC", "LRI", "Stroke", "T2D"]
# diseases = ["COPD"]

countries = [183, 77, 35, 193]


def pie():
    for ssp in ssps:

        mortality_path = f'{base_path}{ssp}/'
        mortality_files = glob(f'{mortality_path}*')

        for year_bin_ind in range(len(year_bins)):
            year_bin = year_bins[year_bin_ind]  # 0, 1, 2, 3, 4 ...
            year_bin_str = [str(x + 2015) for x in year_bin]  # "2015", "2016" ...

            fig = plt.figure(figsize=(15, 15))
            axes = fig.subplots(nrows=2, ncols=2)
            axes[0, 0].set_title("USA")
            axes[0, 1].set_title("India")
            axes[1, 0].set_title("China")
            axes[1, 1].set_title("World")
            title = f'Mortality Distribution in the {year_bin_names[year_bin_ind]} in {ssp} Scenario'
            fig.suptitle(title)

            data = np.empty((len(diseases), len(countries)))

            for disease_ind in range(len(diseases)):
                disease = diseases[disease_ind]
                # Choose all files with correct disease and year within the year bin
                disease_files = [x for x in mortality_files if disease in x and any(year in x for year in year_bin_str)]

                # Import and sum over all files
                column = 13 if disease in ["Allcause", "IHD", "Stroke"] else 1
                mortality = [np.genfromtxt(x, delimiter=',', usecols=column) for x in disease_files]
                mortality = np.sum(mortality, axis=0)
                mortality /= len(disease_files)

                # Select USA, India, China, and World
                mortality = [mortality[183], mortality[77], mortality[35], mortality[0:].sum()]
                data[disease_ind] = mortality

            data[0, :] -= data[1:, :].sum(axis=0)

            axes[0, 0].pie(data[:, 0], labels=disease_labels, autopct='%.1f%%')
            axes[0, 1].pie(data[:, 1], labels=disease_labels, autopct='%.1f%%')
            axes[1, 0].pie(data[:, 2], labels=disease_labels, autopct='%.1f%%')
            axes[1, 1].pie(data[:, 3], labels=disease_labels, autopct='%.1f%%')

            output_file = f'{output_path}{ssp}/{year_bin_names[year_bin_ind]}.png'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Create directory if it does not exist
            plt.savefig(output_file)
            # plt.show()
            plt.close(fig)

            print(f'DONE: {ssp}, {year_bin_names[year_bin_ind]}, {datetime.now()}')


def main():
    pie()


if __name__ == "__main__":
    main()
