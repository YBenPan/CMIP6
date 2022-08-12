import os
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import csv
from lib.map import get_countries_mask, get_grid_area, get_pop
from lib.country import get_regions
from lib.helper import pop_ssp_dict


pm25_path = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/annual_0.5x0.5"
mort_path = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/Health"
ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]


def mean(ssp, year, fractionCountries, type):
    """Compute the mean PM2.5 concentration, given SSP, year, and countries fractions"""
    if type not in ["PM2.5 Concentration", "Mortality"]:
        raise Exception(f"Unknown type {type}")
    grid_area, tot_area = get_grid_area(fractionCountries)
    pop, tot_pop = get_pop(ssp, year, fractionCountries)
    pop_ssp = pop_ssp_dict[ssp]

    all_data = []  # Unweighted mean
    all_awm = []  # Area weighted mean
    all_pwm = []  # Population weighted mean
    models = os.listdir(os.path.join(pm25_path, ssp, "mmrpm2p5"))
    models = [
        model
        for model in models
        if model
        in [
            "CESM2-WACCM6",
            "GFDL-ESM4",
            "GISS-E2-1-G",
            "MIROC-ES2L",
            "MIROC6",
            "MRI-ESM2-0",
            "NorESM2-LM",
        ]
    ]

    for model in models:

        # Compute mean PM2.5 concentration of all realizations
        if type == "PM2.5 Concentration":
            search_str = os.path.join(
                pm25_path, ssp, "mmrpm2p5", model, "*", f"annual_avg_{year}.nc"
            )
            files = sorted(glob(search_str))
        elif type == "Mortality":
            search_year = 2040 if year == 2040 else 2015
            search_str = os.path.join(
                mort_path,
                f"Baseline_Ben_{search_year}_National",
                "5_years",
                ssp,
                f"Pop_{pop_ssp}_var",
                "MortalityAbsolute",
                "Allcause_mean",
                f"{model}_*_{year}_GEMM.nc",
            )
            files = sorted(glob(search_str))
        if len(files) == 0:
            raise Exception(f"{search_str} not found!")
        model_data = []
        model_awm = []
        model_pwm = []

        for file in files:
            # Import concentration NC file
            wk = Dataset(file, "r")
            if type == "PM2.5 Concentration":
                data = wk["concpm2p5"][:]
                country_data = (
                    data * fractionCountries * (10**9)
                )  # Apply mask to concentration array
            elif type == "Mortality":
                data = wk["deaths__mean"]
                country_data = data * fractionCountries

            area_weighted_mean = np.sum(grid_area * country_data) / tot_area
            pop_weighted_mean = np.sum(pop * country_data) / tot_pop

            # Compute mean concentration of every province
            # state_means = np.zeros(len(states))
            # for k, state in enumerate(states):
            #     state_conc = conc * fractionState[k] * (10 ** 9)
            #     state_area = np.sum(fractionState[k])
            #     state_means[k] = np.sum(state_conc) / state_area
            # all_conc.append(state_means)

            model_data.append(country_data)
            model_awm.append(area_weighted_mean)
            model_pwm.append(pop_weighted_mean)
        model_data = np.mean(model_data, axis=0)
        model_awm = np.mean(model_awm, axis=0)
        model_pwm = np.mean(model_pwm, axis=0)
        all_data.append(model_data)
        all_awm.append(model_awm)
        all_pwm.append(model_pwm)
        # print(f"{model}: PWM: {np.round(model_pwm, 2)}, AWM: {np.round(model_awm, 2)}")

    # Compute mean and standard deviation
    conc_std = np.std(all_data, axis=0)
    awm_std = np.std(all_awm, axis=0)
    pwm_std = np.std(all_pwm, axis=0)

    conc = np.mean(all_data, axis=0)
    awm = np.mean(all_awm, axis=0)
    pwm = np.mean(all_pwm, axis=0)
    return conc, awm, pwm, conc_std, awm_std, pwm_std


def get_means(regions, region_countries, region_countries_names, ssp, year, type):
    """Return mean values of input regions"""
    awms = np.zeros(len(regions))
    pwms = np.zeros(len(regions))
    awms_std = np.zeros(len(regions))
    pwms_std = np.zeros(len(regions))
    np.zeros(len(regions))
    for i, (region, countries, countries_names) in enumerate(
        zip(regions, region_countries, region_countries_names)
    ):
        # Get country mask
        fractionCountries = get_countries_mask(countries=countries)

        # Get grid areas for area weighted mean
        grid_area, tot_area = get_grid_area(fractionCountries)

        # Get population for population weighted mean for verification
        pop, tot_pop = get_pop(ssp, year, fractionCountries)
        # print(
        #     f"{region} population: {int(np.round(tot_pop, -6))} area: {int(tot_area)}"
        # )

        conc, awm, pwm, conc_std, awm_std, pwm_std = mean(
            ssp, year, fractionCountries, type
        )
        # print(f"{ssp} Region {region} has AWM {awm}, PWM {pwm}")
        awms[i] = awm
        pwms[i] = pwm
        awms_std[i] = awm_std
        pwms_std[i] = pwm_std

    return awms, pwms, awms_std, pwms_std


def output_means(year, region_source):
    """Output the means and standard deivations of input regions in ssp245"""
    regions, region_countries, region_countries_names = get_regions(region_source)

    output_dir = "/home/ybenp/CMIP6_Images/PM2.5/map"
    output_file = os.path.join(output_dir, f"{year}_{region_source}_mean.csv")
    data = pd.DataFrame()

    for ssp in ssps:
        awms, pwms, awms_std, pwms_std = get_means(regions, region_countries, region_countries_names, ssp, year, "PM2.5 Concentration")

        df = pd.DataFrame(
            {
                "SSP": ssp,
                "Region Name": regions,
                "AWM": awms,
                "PWM": pwms,
                "AWM STD": awms_std,
                "PWM STD": pwms_std,
            }
        )
        data = data.append(df)

        # with open(output_file, "w") as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     csvwriter.writerow(["Region Name", "AWM", "PWM", "AWM STD", "PWM STD"])
        #     for i, region in enumerate(regions):
        #         csvwriter.writerow([region, awms[i], pwms[i], awms_std[i], pwms_std[i]])
        #         print(
        #             f"Region {region}: has AWM {awms[i]} with {awms_std[i]} uncertainty and PWM {pwms[i]} with {pwms_std[i]} uncertainty"
        #         )
    data.to_csv(output_file)
