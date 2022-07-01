import os
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import csv
from lib.map import get_countries_mask, get_grid_area, get_pop

pm25_path = "/project/ccr02/lamar/CMIP6_analysis/PM2.5/annual_0.5x0.5"
ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]

def mean(ssp, year, fractionCountries, grid_area, tot_area, pop, tot_pop):
    """Compute the mean PM2.5 concentration, given SSP, year, and countries fractions"""
    all_conc = []  # Unweighted mean
    all_awm = []  # Area weighted mean
    all_pwm = []  # Population weighted mean
    models = os.listdir(os.path.join(pm25_path, ssp, "mmrpm2p5"))

    for model in models:
        # Outlier: extremely large data
        if "EC-Earth3" in model:
            continue
        if "IPSL" in model or "MPI" in model:
            continue
        # Skip models that do not include natural PM2.5 sources (anthropogenic only)
        # if model not in ["GFDL-ESM4", "MRI-ESM2-0"]:
        #     continue

        # Compute mean PM2.5 concentration of all realizations
        files = sorted(
            glob(
                os.path.join(
                    pm25_path, ssp, "mmrpm2p5", model, "*", f"annual_avg_{year}.nc"
                )
            )
        )
        if len(files) == 0:
            continue
        model_conc = []
        model_awm = []
        model_pwm = []

        for file in files:
            # Import concentration NC file
            wk = Dataset(file, "r")
            conc = wk["concpm2p5"][:]

            # Calculate concentration and means
            country_conc = (
                conc * fractionCountries * (10**9)
            )  # Apply mask to concentration array
            area_weighted_mean = np.sum(grid_area * country_conc) / tot_area
            pop_weighted_mean = np.sum(pop * country_conc) / tot_pop

            # Compute mean concentration of every province
            # state_means = np.zeros(len(states))
            # for k, state in enumerate(states):
            #     state_conc = conc * fractionState[k] * (10 ** 9)
            #     state_area = np.sum(fractionState[k])
            #     state_means[k] = np.sum(state_conc) / state_area
            # all_conc.append(state_means)

            model_conc.append(country_conc)
            model_awm.append(area_weighted_mean)
            model_pwm.append(pop_weighted_mean)

            # real = file.split("mmrpm2p5/")[1].split("\\annual_avg")[0]

        model_conc = np.mean(model_conc, axis=0)
        model_awm = np.mean(model_awm, axis=0)
        model_pwm = np.mean(model_pwm, axis=0)
        all_conc.append(model_conc)
        all_awm.append(model_awm)
        all_pwm.append(model_pwm)
        # print(f"{model}: PWM: {np.round(model_pwm, 2)}, AWM: {np.round(model_awm, 2)}")
    all_conc = np.mean(all_conc, axis=0)
    all_awm = np.mean(all_awm, axis=0)
    all_pwm = np.mean(all_pwm, axis=0)
    return all_conc, all_awm, all_pwm


def get_means(regions, region_countries, region_countries_names, ssp, year):
    """Return mean values of input regions"""
    awms = np.zeros(len(regions))
    pwms = np.zeros(len(regions))
    for i, (region, countries, countries_names) in enumerate(zip(
        regions, region_countries, region_countries_names)
    ):
        # Get country mask
        fractionCountries = get_countries_mask(countries=countries)

        # Get grid areas for area weighted mean
        grid_area, tot_area = get_grid_area(fractionCountries)

        # Get population for population weighted mean
        pop, tot_pop = get_pop(fractionCountries)
        conc, awm, pwm = mean(
            ssp, year, fractionCountries, grid_area, tot_area, pop, tot_pop
        )
        # print(f"{ssp} Region {region} has AWM {awm}, PWM {pwm}")
        awms[i] = awm
        pwms[i] = pwm
    return awms, pwms


def output_means(regions, region_countries, region_countries_names):
    awms = []
    pwms = []
    for ssp in ssps:
        awm, pwm = get_means(regions, region_countries, region_countries_names, ssp=ssp, year=2015)
        awms.append(awm)
        pwms.append(pwm)
    awms = np.mean(awms, axis=0)
    pwms = np.mean(pwms, axis=0)
    
    output_dir = "/home/ybenp/CMIP6_Images/PM2.5/map"
    output_file = os.path.join(output_dir, "2015_mean.csv")
    with open(output_file, "w") as csvfile: 
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Region Name", "AWM", "PWM"])
        for i, region in enumerate(regions):
            csvwriter.writerow([region, awms[i], pwms[i]])
            print(f"Region {region}: has AWM {awms[i]} and PWM {pwms[i]}")
