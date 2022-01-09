from glob import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

####################################################################################################
#### CREATE PLOTS BASED ON COUNTRY MORTALITY DATA
####################################################################################################

####################################################################################################
#### USER INPUT:
#### WHERE ARE THE COUNTRY MORTALITY DATA?
base_path = "D:/CMIP6_data/Mortality/"
#### WHERE IS THE OUTPUT FOLDER?
output_path = "D:/CMIP6_Images/Mortality/plot1/"
####################################################################################################

ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]
# ssps = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585"]
years = np.arange(2015, 2101)
# years = [2100]
year_bins = [
    np.arange(0, 5),
    np.arange(5, 15),
    np.arange(15, 25),
    np.arange(25, 35),
    np.arange(35, 45),
    np.arange(45, 55),
    np.arange(55, 65),
    np.arange(65, 75),
    np.arange(75, 85),
    [85]
]

year_bin_names = [str(x) + '*' for x in np.arange(201, 211)]

diseases = ["Allcause", "COPD", "IHD", "LC", "LRI", "Stroke", "T2D"]
# diseases = ["Allcause"]

# {"USA": 183, "India": 77, "China": 35, "World": 193}
countries = [183, 77, 35, 193]


def all_years():

    for disease in diseases:

        fig = plt.figure(figsize=(15, 15))
        axes = fig.subplots(nrows=2, ncols=2)
        axes[0, 0].set_title("USA")
        axes[0, 1].set_title("India")
        axes[1, 0].set_title("China")
        axes[1, 1].set_title("World")

        title = f'{disease} from 2015 to 2100'
        fig.suptitle(title)

        for ssp in ssps:

            mortality_path = f'{base_path}{ssp}/'
            mortality_files = glob(f'{mortality_path}*')
            data = np.empty((len(years), len(countries)))

            for year_ind in range(len(years)):

                year = str(year_ind + years[0])
                disease_files = [x for x in mortality_files if disease in x and str(year) in x]

                # Import and sum over all models
                column = 13 if disease in ["Allcause", "IHD", "Stroke"] else 1
                mortality = [np.genfromtxt(x, delimiter=',', usecols=column) for x in disease_files]
                mortality = np.sum(mortality, axis=0)
                mortality /= len(disease_files)     # Average over all models

                # Extract regional data
                data[year_ind, 0] = mortality[183]       # USA
                data[year_ind, 1] = mortality[77]        # India
                data[year_ind, 2] = mortality[35]        # China
                data[year_ind, 3] = mortality[0:].sum()  # World

                print(f'DONE: {disease}, {ssp}, {year}, {datetime.now()}')

            axes[0, 0].plot(years, data[:, 0], label=ssp)
            axes[0, 1].plot(years, data[:, 1], label=ssp)
            axes[1, 0].plot(years, data[:, 2], label=ssp)
            axes[1, 1].plot(years, data[:, 3], label=ssp)

        for ax in axes.flatten():
            ax.legend(loc="upper right")
            ax.set_xlabel("Year")
            ax.set_ylabel("Number of deaths")

        output_file = f'{output_path}{disease}.png'
        plt.savefig(output_file)
        # plt.show()
        plt.close(fig)


def avg():

    for disease in diseases:

        fig = plt.figure(figsize=(15, 15))
        axes = fig.subplots(nrows=2, ncols=2)
        axes[0, 0].set_title("USA")
        axes[0, 1].set_title("India")
        axes[1, 0].set_title("China")
        axes[1, 1].set_title("World")

        title = f'{disease} from 2015 to 2100'
        fig.suptitle(title)

        for ssp in ssps:

            mortality_path = f'{base_path}{ssp}/'
            mortality_files = glob(f'{mortality_path}*')
            data = np.empty((len(year_bins), len(countries)))

            for year_bin_ind in range(len(year_bins)):
                year_bin = year_bins[year_bin_ind]
                year_bin_mortality = np.zeros(4)
                for year_ind in year_bin:
                    year = str(year_ind + years[0])

                    disease_files = [x for x in mortality_files if disease in x and str(year) in x]

                    # Import and sum over all models
                    column = 13 if disease in ["Allcause", "IHD", "Stroke"] else 1
                    mortality = [np.genfromtxt(x, delimiter=',', usecols=column) for x in disease_files]
                    mortality = np.sum(mortality, axis=0)
                    mortality /= len(disease_files)  # Average over all models

                    # Select USA, India, China, and World
                    mortality = [mortality[183], mortality[77], mortality[35], mortality[0:].sum()]
                    # Add average mortality of a year to the total mortality

                    year_bin_mortality += mortality

                data[year_bin_ind] = year_bin_mortality / len(year_bin)
                print(f'DONE: {ssp}, {disease}, {len(year_bin)}, {datetime.now()}')

            axes[0, 0].plot(year_bin_names, data[:, 0], label=ssp)
            axes[0, 1].plot(year_bin_names, data[:, 1], label=ssp)
            axes[1, 0].plot(year_bin_names, data[:, 2], label=ssp)
            axes[1, 1].plot(year_bin_names, data[:, 3], label=ssp)

        for ax in axes.flatten():
            ax.legend(loc="upper right")
            ax.set_xlabel("Year")
            ax.set_ylabel("Number of deaths")

        output_file = f'{output_path}{disease}_avg.png'
        # plt.savefig(output_file)
        plt.show()
        plt.close(fig)


def main():
    all_years()


if __name__ == "__main__":
    main()

