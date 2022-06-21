import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from decomposition import multi_year_mort, init_by_factor

####################################################################################################
#### CREATE STACKED BAR PLOT OF GLOBAL MORTALITY IN 2015, 2030, AND 2040
####################################################################################################

# General Settings
parent_dir = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/Health"
output_dir = "/home/ybenp/CMIP6_Images/Mortality/global_mort"
pop_ssp_dict = {
    "ssp119": "ssp1",
    "ssp126": "ssp1",
    "ssp245": "ssp2",
    "ssp370": "ssp3",
    "ssp434": "ssp2",
    "ssp460": "ssp2",
    "ssp585": "ssp1",
}

# Run Settings
ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]
years = [2015, 2030, 2040]
diseases = ["COPD", "IHD", "LC", "LRI", "Stroke", "T2D"]
age_groups = ["25-60", "60-80", "80+"]


def diseases_stacked(factor_name, factors, pop, baseline, country=-1, country_long_name="World"):
    df = pd.DataFrame(columns=["Year", "SSP", factor_name, "Mean"])
    xlabels = []
    for i, year in enumerate(years):

        for j, ssp in enumerate(ssps):
            pop_ssp = pop_ssp_dict[ssp]

            for p, factor in enumerate(factors):
                ages, disease = init_by_factor(factor_name, factor)

                factor_mean, std = multi_year_mort(
                    pop=pop,
                    baseline=baseline,
                    year=year,
                    ssp=ssp,
                    ages=ages,
                    disease=disease,
                    country=country,
                    return_values=True,
                )
                df = df.append(
                    {
                        "Year": year,
                        "SSP": ssp,
                        factor_name: factor,
                        "Mean": factor_mean,
                        "STD": std,
                    },
                    ignore_index=True,
                )
                print(f"{year}, {ssp}, {pop}, {baseline}, {factor} mean: {factor_mean}; STD: {std}")
            xlabels.append(f"{year}, {ssp}")

    sns.set()
    g = sns.catplot(kind="bar", data=df, col="Year", x="SSP", y="Mean", hue=factor_name)
    g.set_axis_labels("SSP", "Number of Deaths")
    g.set_xticklabels(["1", "2", "3", "5"])
    g.set_titles("{col_name}")
    fig = g.fig
    fig.suptitle(country_long_name)
    fig.tight_layout(rect=[0, 0.03, 0.93, 0.95])

    # Add error bars
    for i, year in enumerate(years):
        ax=g.axes[0][i]
        year_df = df[df["Year"] == year]
        year_stds = year_df["STD"].values
        x_coords = sorted([p.get_x() + 0.5 * p.get_width() for p in ax.patches])
        y_coords = year_df["Mean"].values
        ax.errorbar(x=x_coords, y=y_coords, yerr=year_stds, fmt="none", color="black")

    output_file = f"{output_dir}/{country_long_name}_{factor_name}_{pop}_{baseline}.png"
    plt.savefig(output_file)
    # plt.show()


def main():
    diseases_stacked(factor_name="Age", factors=age_groups, pop="var", baseline="2015", country=35, country_long_name="China")


if __name__ == "__main__":
    main()
