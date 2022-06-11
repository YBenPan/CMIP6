import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from decomposition import get_models, mort, init_by_factor

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


def diseases_stacked(factor_name, factors, pop_baselines, country=-1, country_long_name="World"):
    df = pd.DataFrame(columns=["Year", "SSP", "Pop_Baseline", factor_name, "Mean"])
    xlabels = []
    for i, year in enumerate(years):

        for j, ssp in enumerate(ssps):
            pop_ssp = pop_ssp_dict[ssp]

            for k, pop_baseline in enumerate(pop_baselines):
                pop, baseline = pop_baseline

                for p, factor in enumerate(factors):
                    ages, disease = init_by_factor(factor_name, factor)

                    factor_mean = mort(
                        pop=pop,
                        baseline=baseline,
                        year=year,
                        ssp=ssp,
                        ages=ages,
                        disease=disease,
                        country=country,
                    )
                    df = df.append(
                        {
                            "Year": year,
                            "SSP": ssp,
                            "Pop_Baseline": pop_baseline,
                            factor_name: factor,
                            "Mean": factor_mean,
                        },
                        ignore_index=True,
                    )
                    print(f"{year}, {ssp}, {pop_baseline}, {factor} mean: {factor_mean}")
                xlabels.append(f"{year}, {ssp}")

    df = df.groupby(["Year", "SSP", "Pop_Baseline", factor_name]).sum().unstack(factor_name)
    fig, ax = plt.subplots(figsize=(10, 10))
    df.plot.bar(ax=ax, stacked=True)
    ax.set_xticks(ticks=np.arange(0, 12), labels=xlabels)
    ax.set_ylabel("Global Mortality")

    output_file = f"{output_dir}/{country_long_name}_{factor_name}_{pop_baselines[0][0]}_{pop_baselines[0][1]}.png"
    plt.savefig(output_file)
    # plt.show()


def main():
    diseases_stacked(factor_name="Age", factors=age_groups, pop_baselines=[("var", "2015")], country=183, country_long_name="US")
    # china_tmp()


if __name__ == "__main__":
    main()
