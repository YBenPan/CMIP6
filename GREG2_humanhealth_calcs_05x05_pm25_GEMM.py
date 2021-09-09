#!/usr/bin/env python
# -*- coding: utf-8 -*-
# """
# Created on Thu Jul  5 22:42:39 2018
#
# @author: mr245
# Greg Faluvegi is altering to use on NCSS supercomputer and use GISS model output.
# """

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import sys

# TODO: Greg, but these switcher routines into a separate file so that the mapping can
# be used in other programs...
# In fact, I could do that with all these paths, etc. i.e. move all the GISS-specific
# sutff "under the hood."


def name2(dis):
    # to get the name string that starts the baseline mortality input file
    switcher = {
        "Allcause": "allcause",
        "IHD": "ihd",
        "Stroke": "stroke",
        "COPD": "copd",
        "LC": "lc",
        "LRI": "lri",
        "T2D": "diabetes2",
    }
    return switcher.get(dis, "Invalid name2 disease")


def subdir(dis):
    # to get the name of baseline mortality subdirectory
    switcher = {
        "Allcause": "all_cause/",
        "IHD": "ihd/",
        "Stroke": "stroke/",
        "COPD": "copd/",
        "LC": "lc/",
        "LRI": "lri/",
        "T2D": "diab_type2/",
    }
    return switcher.get(dis, "Invalid subdir disease")


def mortTag(dis):
    # to get basically the date stamp on the baseline mortality filename
    # lc used to be 28.1.19 but when NaNs were fixed for some reason it became
    # 25.1.19...
    switcher = {
        "Allcause": "25.3.19",
        "IHD": "25.1.19",
        "Stroke": "25.1.19",
        "COPD": "28.1.19",
        "LC": "25.1.19",
        "LRI": "25.1.19",
        "T2D": "30.1.19",
    }
    return switcher.get(dis, "Invalid mortTag disease")


def main(argv):
    runName, runYear, tag, disease = sys.argv[1:]
    # passing simulation name, representative year tag (like ANN2026-2035) and the disease name
    # e.g. E21UasiaWNCLEa 2030 ANN2026-2035 LC

    ## disease SHOULD BE WITHIN ONE OF THE FOLLOWING:
    ## Allcause
    ## IHD
    ## Stroke
    ## COPD (NOT SPECIFYING AGE GROUP)
    ## LC, (NOT SPECIFYING AGE GROUP)
    ## LRI, (NOT SPECIFYING AGE GROUP)
    ## T2D, (NOT SPECIFYING AGE GROUP)

    ####################################################################################################
    #### CALCULATES GLOBAL HUMAN-HEALTH MORTALITY FROM VARIOUS CAUSES DUE TO PM2.5 BASED ON RISK FUNCTIONS
    #### FOR TYPE 2 DIABETES, THE IER FUNCTION FORM IS USED
    #### FOR ALL THE OTHER DISEASES, GEMM MODEL (BURNETT 2018) IS USED
    #### OUTPUT INCLUDE:
    ####     1. GLOBAL NETCDF FILES OF CALCULATED RELATIVE RISK (RR) AND ATTRIBUTABLE FRACTIONS(AF) AND
    ####        .CSV SUMMARY FILES
    ####     2. GLBOAL NETCEF FILES OF MORTALITIES AND MORTALITY RATES BY CAUSES AND .CSV SUMMARY FILES
    ####################################################################################################

    ####################################################################################################
    #### OTHER USER SETTINGS:
    #### SPECIFY YEAR:
    # Mortality is always using Duke files; for now, for population, there is the option to use
    # the GISS files:
    popChoice = "GISS"  ## popChoice='Duke'
    if popChoice == "Duke":
        if runYear > 2015:
            popYear = "2015"
            print(
                "Warning: year "
                + runYear
                + " outside of "
                + popChoice
                + " range. Using: "
                + popYear
            )
        elif runYear < 2000:
            popYear = "2000"
            print(
                "Warning: year "
                + runYear
                + " outside of "
                + popChoice
                + " range. Using: "
                + popYear
            )
        else:
            popYear = runYear
    elif popChoice == "GISS":
        popYear = runYear
    bmortYear = runYear
    print("popYear = " + popYear + " (" + popChoice + ") ; bmortYear = " + bmortYear)
    #### WHAT IS THE BASE PATH TO THE GENERIC INPUT DATA (POPULATION, MORTALITY, COUNTRY FRACTIONS, ETC.)?
    base_path = "/discover/nobackup/projects/giss_ana/users/gfaluveg/newDecarbonHealth/From_Duke/PM_health_GEMM/healthdata/Generic/"
    out_path = "/discover/nobackup/projects/giss_ana/users/gfaluveg/newDecarbonHealth/PM_Output/"
    popTag = "16.4.18"  # note alternate path:
    pop_path = "/discover/nobackup/projects/giss_ana/users/gfaluveg/newDecarbonHealth/From_Duke/4health/"
    #### WHAT IS THE BASE PATH TO THE PM25 CONC. DATA? MUST BE GRIDDED 05X05 GLOBAL FILES, WITH APPROPRIATE AVERAGING
    pm25_path = "/discover/nobackup/projects/giss_ana/users/gfaluveg/newDecarbonHealth/getPM2p5input/DATA/"
    #### WHAT IS THE BASE PATH TO THE BASELINE MORTALITY DATA, 05X05
    mort_path = "/discover/nobackup/projects/giss_ana/users/gfaluveg/newDecarbonHealth/From_Duke/4health/baseline_mortality/gbd2017/"
    #### WHAT IS CONCENTRATION FILE NAME?
    burnett_file = (
        "XtSS_DSWlpUf_PM2.5_microgram_m-3_" + tag + "_" + runName + "_0.5x0.5.nc"
    )
    #### WHAT IS THE NAME OF THE COUNTRY FRACTION FILE?
    fc_file = "countryFractions_2010_0.5x0.5.nc"
    #### IF TO INLCUDE THE CHINESE MALE COHORT, INPUT BETWEEN 'Y' OR 'N'.
    #### IF 'Y', THE C-R FUNCTION WILL BE THE ONE GENERATED WITH THE CHINESE MALE DATA. DEFAULT IS 'Y'
    chinesemale = "Y"
    ####################################################################################################
    ageBins = [
        "25_29",
        "30_34",
        "35_39",
        "40_44",
        "45_49",
        "50_54",
        "55_59",
        "60_64",
        "65_69",
        "70_74",
        "75_79",
        "80",
    ]
    # Note: for the RR and AF parts, there is one extra age bin at the start for the over-25 population.
    # This is the reason for the usages of len(ageBins)+1 below. I.e. original list looked like this:
    # age_bins = ('post25','25_29','30_34','35_39','40_44','45_49','50_54','55_59','60_64','65_69','70_74','75_79','post80')
    # but nothing was actually being done with the values within. Note the post25 at front.
    nAvgLowHigh = 3  # for loops over average, low-estimate, high-estimate
    ####################################################################################################
    #### IMPORT NON-AGE BINNED DATASETS
    #### ALL DATA MUST BE IN 05X05 GRIDDED RESOLUTION.
    ####

    #### IMPORT ANNUAL AVG. PM25 DATA
    f1 = Dataset(pm25_path + burnett_file, "r")
    BurnettPM25 = f1.variables["sfcPM2p5"][:, :]
    latitude = f1.variables["lat"][:]
    longitude = f1.variables["lon"][:]
    BurnettPM25[np.isnan(BurnettPM25)] = 0.0  ## z value in the formula
    n_lat = len(latitude)
    n_lon = len(longitude)
    f1.close()

    #### IMPORT COUNTRY FRACTIONS
    f1 = Dataset(base_path + fc_file, "r")
    fractionCountry = f1.variables["fractionCountry"][:, :, :]
    num_countries = len(f1.variables["countryIndex"][:])  ## country index starts from 1
    f1.close()
    fractionCountry[fractionCountry < 0.0] = 0.0

    ####################################################################################################
    #### CALCULATE RR AND AF USING ANNUAL AVERAGE OF PM2.5
    #### COLUMNS: #of Country, AVG, LOW, HIGH
    Burnett_af = np.zeros((num_countries, len(ageBins) + 1, nAvgLowHigh + 1))

    ####  INITIALIZE ARRAYS
    #### PARAMETERS: mu, v, theta
    #### CHARACTERS: cause,age

    # 1st index: 0:average, 1:low, 2: high
    Cause_calcs_RR = np.zeros((nAvgLowHigh, len(ageBins) + 1, n_lat, n_lon))
    Cause_calcs_AF = np.zeros((nAvgLowHigh, len(ageBins) + 1, n_lat, n_lon))

    #### USE IER FOR T2D
    if disease == "T2D":
        #### IMPORT PARAMETERS
        data = base_path + "params_t2_dm_99.csv"
        Cause_para = np.genfromtxt(
            data, delimiter=",", skip_header=1, usecols=(1, 2, 3, 4)
        )  # last col is counterfactual values
        Cause_calcs_temp = np.zeros((1000, n_lat, n_lon))  # first col is for iterations

        for i in range(0, 1000):
            BurnettPM25_temp = BurnettPM25[:, :] - Cause_para[i, 3]
            BurnettPM25_temp[BurnettPM25_temp[:, :] < 0] = 0.0
            Cause_calcs_temp[i, :, :] = 1 + Cause_para[i, 0] * (
                1
                - np.exp(-Cause_para[i, 1] * BurnettPM25_temp[:, :] ** Cause_para[i, 2])
            )  # function form of IER

        #### CALCULATE RR
        Cause_calcs_RR[0, 0, :, :] = np.average(
            Cause_calcs_temp[:, :, :], axis=0
        )  # average RR
        Cause_calcs_RR[1, 0, :, :] = Cause_calcs_RR[0, 0, :, :] - 2 * np.std(
            Cause_calcs_temp[:, :, :], axis=0
        )  # lower bound
        Cause_calcs_RR[2, 0, :, :] = Cause_calcs_RR[0, 0, :, :] + 2 * np.std(
            Cause_calcs_temp[:, :, :], axis=0
        )  # upper bound

        #### CALCULATE AF
        for m in range(nAvgLowHigh):
            Cause_calcs_AF[m, 0, :, :] = (
                Cause_calcs_RR[m, 0, :, :] - 1
            ) / Cause_calcs_RR[m, 0, :, :]

        #### CALCULATE COUNTRY-LEVEL AF
        for j in range(num_countries):  ## report AF, only filling in the 1st age bin:
            Burnett_af[j, 0, 0] = j + 1
            for m in range(nAvgLowHigh):
                Burnett_af[j, 0, m + 1] = np.sum(
                    fractionCountry[j, :, :] * Cause_calcs_AF[m, 0, :, :]
                ) / np.sum(fractionCountry[j, :, :])

    else:  #### USE GEMM FOR OTHER CAUSES
        data = base_path + "GEMM_CRCurve_parameters.csv"
        if chinesemale == "Y":
            Cause_para = pd.read_csv(
                data,
                usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8),
                header=1,
                skiprows=(0),
                dtype={"z": np.float32},
            )
        else:
            Cause_para = pd.read_csv(
                data,
                usecols=(0, 1, 2, 3, 9, 10, 11, 12, 13),
                header=1,
                skiprows=(0),
                dtype={"z": np.float32},
            )

        Cause_para = Cause_para[Cause_para["cause"] == disease]
        Cause_para = Cause_para.reset_index(drop=True)  ## reset the index from 0

        #### calculate RR and AF, using GEMM(z)=exp{θlog(z/α+1)/(1+exp{-(z-µ)/ν}), where z=max(0, PM2.5-2.4µg/m3)
        zcf = 2.4
        Cause_calcs_temp = np.zeros((n_lat, n_lon))
        Cause_calcs_temp = BurnettPM25[:, :] - zcf  ##  z=max(0, PM2.5-2.4µg/m3)
        Cause_calcs_temp[Cause_calcs_temp[:, :] < 0] = 0.0  ## f(z)=z
        ## alternatively, f(z)=log(z+1): Cause_calcs_f_temp = np.log(Cause_calcs_temp+1)

        mu = Cause_para.loc[0, "u"]
        v = Cause_para.loc[0, "v"]
        a = Cause_para.loc[0, "a"]

        logit_w = 1 / (
            1 + np.exp(-(Cause_calcs_temp[:, :] - mu) / v)
        )  ## mu and v is the same value for all age groups

        if (
            disease == "Allcause" or disease == "IHD" or disease == "Stroke"
        ):  ## these three diseases have different age groups
            ngroups = len(ageBins) + 1
        else:  ## the other diseases only have one age group, so the second dimension of the arrays are all set to 0, i.e.post-25 group
            ngroups = 1

        for i in list(range(ngroups)):
            np.random.seed(0)  ## set the seed for the random number generations later
            sim_coef = np.random.normal(
                Cause_para.loc[i, "theta"], Cause_para.loc[i, "theta_SE"], 1000
            )

            Cause_calcs_RR[0, i, :, :] = np.exp(
                np.log(Cause_calcs_temp / a + 1) * logit_w * Cause_para.loc[i, "theta"]
            )  ## calculate RR for each age group
            Cause_calcs_RR[1, i, :, :] = np.exp(
                np.log(Cause_calcs_temp / a + 1)
                * logit_w
                * np.percentile(sim_coef, 5.0)
            )  # 5th
            Cause_calcs_RR[2, i, :, :] = np.exp(
                np.log(Cause_calcs_temp / a + 1)
                * logit_w
                * np.percentile(sim_coef, 95.0)
            )  # 95th
            for m in range(nAvgLowHigh):
                Cause_calcs_AF[m, i, :, :] = (
                    Cause_calcs_RR[m, i, :, :] - 1
                ) / Cause_calcs_RR[m, i, :, :]

            #### Calculate country level results of AF
            for j in range(num_countries):  ## report AF
                Burnett_af[j, i, 0] = j + 1
                for m in range(nAvgLowHigh):
                    Burnett_af[j, i, m + 1] = np.sum(
                        fractionCountry[j, :, :] * Cause_calcs_AF[m, i, :, :]
                    ) / np.sum(fractionCountry[j, :, :])

    #### Output NC file of AF and RR, total of all age groups
    f1 = Dataset(
        out_path + disease + "_" + runName + "_" + tag + "_RR+AF_05x05_GEMM.nc",
        "w",
        format="NETCDF4_CLASSIC",
    )

    # define coordinates/coordinate variables:
    lat = f1.createDimension("lat", n_lat)
    lon = f1.createDimension("lon", n_lon)
    lat = f1.createVariable("lat", np.double, ("lat",))
    lon = f1.createVariable("lon", np.double, ("lon",))

    # define other variables:
    suff = ["", "_low", "_up"]
    Cause_RR_out = []
    Cause_AF_out = []
    for m in range(nAvgLowHigh):
        Cause_RR_out.append(
            f1.createVariable("Cause_calcs_RR" + suff[m], np.float32, ("lat", "lon"))
        )  # ,fill_value=0))
        Cause_AF_out.append(
            f1.createVariable("Cause_calcs_AF" + suff[m], np.float32, ("lat", "lon"))
        )  # ,fill_value=0))

    # Global Attributes
    f1.description = (
        "Calculated grid level RR and AF for post 25 years old people for each disease"
    )
    f1.contact = "MuyeRu - muye.ru@duke.edu"

    # Variable Attributes
    lat.units = "degrees_north"
    lat.long_name = "latitude"
    lat.comment = "center of grid cell"
    lon.units = "degrees_east"
    lon.long_name = "longitude"
    lon.comment = "center of grid cell"
    for m in range(nAvgLowHigh):
        Cause_RR_out[m].units = "no units"
        Cause_RR_out[
            m
        ].long_name = "relative risk attributable to PM2.5, using GEMM model in PNAS by Burnett et al. (2018)"
        Cause_AF_out[m].units = "fraction out of 1"
        Cause_AF_out[
            m
        ].long_name = "attributable fraction of a disease attributable to PM2.5"

    # Fill variables:
    f1.variables["lat"][:] = latitude[:]
    f1.variables["lon"][:] = longitude[:]
    for m in range(nAvgLowHigh):
        f1.variables["Cause_calcs_RR" + suff[m]][:, :] = Cause_calcs_RR[m, 0, :, :]
        f1.variables["Cause_calcs_AF" + suff[m]][:, :] = Cause_calcs_AF[m, 0, :, :]

    f1.close()

    Cause_calcs_temp = None
    Cause_calcs_RR = None

    #####################################################################################################
    #### CALCULATE COUNTRY-LEVEL DEATHS AND MORTALITY RATES DUE TO PM2.5 ATTRIBUTABLE TO EACH DISEASE
    #### IMPORT COUNTRY BASELINE MORTATLIY RATES (fraction) TO GRID
    #### IMPORT POPULATION FOR EACH GRID
    #####################################################################################################

    #### IMPORT GRIDDED BASELINE MORTALITY
    # NOTE: input files say % units, but these are in fact fraction units.
    fname = str(
        mort_path
        + subdir(disease)
        + name2(disease)
        + "_"
        + bmortYear
        + "_rate_05x05_"
        + mortTag(disease)
        + ".nc"
    )
    print("Using baseline mortality file: " + fname)
    f1 = Dataset(fname, "r")
    # note this is re-reading in the lat and lon, and we are not checking that they match previous!
    latitude = f1.variables["lat"][:]
    longitude = f1.variables["lon"][:]
    Base_mortrate = []
    for n in range(len(ageBins)):
        Base_mortrate.append(f1.variables["rate_" + ageBins[n]][:, :])
    f1.close()

    #### IMPORT GRIDDED POPULATION DATA (UNITS = # PEOPLE)
    if popChoice == "Duke":
        f_pop = Dataset(
            pop_path + "population/population_" + popYear + "_05x05_" + popTag + ".nc",
            "r",
        )
        tot_pop = f_pop.variables["pop_tot"][:, :]
        pop = []
        for n in range(len(ageBins)):
            pop.append(f_pop.variables["pop_" + ageBins[n]][:, :])
        f_pop.close()

    elif popChoice == "GISS":
        # read in the total population and the fractional age bins (17 of them starting at age
        # 0 then 5, 10, 15, ... 80) and multiply them to get the population per bin into same
        # array as Duke case does:
        offset = 5  # i.e. the 6th age bin in the GISS input starts at age 25
        f_popTots = Dataset(
            "/discover/nobackup/gfaluveg/SHARE/SUPPORTING/populationData/griddedPopulation_"
            + popYear
            + "_0.5x0.5_OCT2017corrected.nc",
            "r",
        )
        f_popFrac = Dataset(
            "/discover/nobackup/gfaluveg/SHARE/SUPPORTING/populationData/griddedAgeBinFracs_"
            + popYear
            + "_0.5x0.5_OCT2017corrected.nc",
            "r",
        )
        tot_pop = f_popTots.variables["population"][:, :]
        pop = []
        for n in range(len(ageBins)):
            pop.append(
                tot_pop[:, :] * f_popFrac.variables["ageFractions"][n + offset, :, :]
            )
        f_popTots.close()
        f_popFrac.close()

    #### FOR DISEASES THAT DO NOT HAVE AGE BINS, CALCULATE ALL AGE GROUP MORTALITY RATE
    # Let's calculation no matter the disease but only use for some:
    # BaselineMortalitySum25 used to be BaselineMortalityRates and was over the full population.
    # Muye noted that we don't add in the age 0 to 24 mortalities because they are 0. So effectively, we ARE
    # adding those in here and that's why we divide by the total population instead of the over-25 population.
    # Greg changed to BaselineMortalitySum25 and then removed the division by to_pop and the multiplication by
    # tot_pop when it is used. See e-mail w/ Drew Oct 7 2019:
    BaselineMortalitySum25 = Base_mortrate[0][:, :] * pop[0][:, :]
    for n in range(len(ageBins) - 1):
        BaselineMortalitySum25 += Base_mortrate[n + 1][:, :] * pop[n + 1][:, :]
    BaselineMortalitySum25[np.isnan(BaselineMortalitySum25)] = 0.0

    #### CALCULATE GRID-LEVEL DEATHS
    # TODO(Greg): I believe this section and it's "else" section could be combined with some
    # forethought on the number of dimensions and the use of BaselineMortalitySum25[] or not.
    # (I.e. don't calculate BaselineMortalitySum25 at all, just sum over them when filling in
    # GEMM_death_val but always using the post 25 Cause_calcs_AF)
    if disease == "COPD" or disease == "LC" or disease == "LRI" or disease == "T2D":
        GEMM_death_val = np.zeros((nAvgLowHigh, n_lat, n_lon))
        for m in range(nAvgLowHigh):
            GEMM_death_val[m, :, :] = (
                Cause_calcs_AF[m, 0, :, :] * BaselineMortalitySum25[:, :]
            )
        GEMM_death_val[np.isnan(GEMM_death_val)] = 0.0

        #### PRINT A GLOBAL SUM FOR CHECK
        print("Burnett global:")
        print(
            np.sum(GEMM_death_val[0, :, :]),
            "(",
            np.sum(GEMM_death_val[1, :, :]),
            "-",
            np.sum(GEMM_death_val[2, :, :]),
            ")",
        )
        print(" ")

        #### CALCULATE MORTALITY RATES
        GEMM_mortrate_val = np.zeros(
            (nAvgLowHigh, n_lat, n_lon)
        )  ##  1st dim for mid, lower, and upper values
        for m in range(nAvgLowHigh):
            GEMM_mortrate_val[m, :, :] = GEMM_death_val[m, :, :] / tot_pop

        #### GENERATE A .NC OUTPUT FILE OF DEATH AND MORTALITY RATES
        f1 = Dataset(
            out_path + disease + "_" + runName + "_" + tag + "_Mortality_05x05_GEMM.nc",
            "w",
            format="NETCDF4_CLASSIC",
        )

        # Define coordinates/coordinate variables:
        ci = f1.createDimension(
            "ci", nAvgLowHigh
        )  ## 0 = mean value, 1 = lower value, 2 = upper value.
        lat = f1.createDimension("lat", n_lat)
        lon = f1.createDimension("lon", n_lon)
        lat = f1.createVariable("lat", np.double, ("lat",))
        lon = f1.createVariable("lon", np.double, ("lon",))
        ci = f1.createVariable(
            "ci", np.double, ("ci",)
        )  # used to be int and that failed on NCCS machines

        # Define other variables:
        Cause_death_post25 = f1.createVariable(
            "Cause_death_post25", np.float32, ("ci", "lat", "lon")
        )  # ,fill_value=0)    ##  deaths
        Cause_mortrate_post25 = f1.createVariable(
            "Cause_mortrate_post25", np.float32, ("ci", "lat", "lon")
        )  # ,fill_value=0)    ##  mortality rate (fraction) of total pop

        # Global Attributes
        f1.description = "Calculated grid level deaths and mortality rate for the cause of disease, only average values are shown, the uncertainty ranges are in the .csv country output"
        f1.contact = "MuyeRu - muye.ru@duke.edu"

        # Variable Attributes
        lat.units = "degrees_north"
        lat.long_name = "latitude"
        lat.comment = "center of grid cell"
        lon.units = "degrees_east"
        lon.long_name = "longitude"
        lon.comment = "center of grid cell"
        ci.long_name = "mean, lower, upper"
        ci.comment = "ci = 0 mean value, ci = 1 lower value, ci =2, upper value"
        Cause_death_post25.units = "person"
        Cause_death_post25.long_name = (
            "annual total deaths due to the cause of disease in post 25 population"
        )
        Cause_mortrate_post25.units = "fraction"
        Cause_mortrate_post25.long_name = (
            "annual mortality rate out of total population"
        )

        # Fill variables:
        f1.variables["lat"][:] = latitude[:]
        f1.variables["lon"][:] = longitude[:]
        f1.variables["ci"][:] = np.arange(nAvgLowHigh)
        for m in range(nAvgLowHigh):
            f1.variables["Cause_death_post25"][m, :, :] = GEMM_death_val[m, :, :]
            f1.variables["Cause_mortrate_post25"][m, :, :] = GEMM_mortrate_val[m, :, :]

        f1.close()

        #### CALCULATE COUNTRY LEVEL RESULTS AND GENERATE A .CSV FILE
        data = pop_path + "countryvalue_blank.csv"
        Cause_characters = np.genfromtxt(
            data, delimiter=",", usecols=(0), dtype=str, skip_header=1
        )

        GEMM_death_cty = np.zeros(
            (num_countries, nAvgLowHigh + 1)
        )  ## first column is country NO., then 3 cols for average, low, and high
        GEMM_death_cty[:, 0] = -1.0  # b/c I've replaced by names below as per Drew
        GEMM_mortrate_cty = np.zeros((num_countries, nAvgLowHigh + 1))
        GEMM_mortrate_cty[:, 0] = -1.0  # b/c I've replaced by names below as per Drew

        for i in range(num_countries):
            temp_Pop = np.sum(
                fractionCountry[i, :, :] * tot_pop[:, :]
            )  ## Country population
            for j in list(range(nAvgLowHigh)):
                GEMM_death_cty[i, j + 1] = np.sum(
                    fractionCountry[i, :, :] * GEMM_death_val[j, :, :]
                )
                GEMM_mortrate_cty[i, j + 1] = (
                    GEMM_death_cty[i, j + 1] / temp_Pop
                )  ## mortality rate out of total population

        GEMM_death_cty = np.round(GEMM_death_cty[:], 2)
        GEMM_mortrate_cty = np.round(GEMM_mortrate_cty[:], 7)

        #### country level
        headerline1 = "CountryID, Mean, Low, High"
        output_file = (
            out_path
            + disease
            + "_"
            + runName
            + "_"
            + tag
            + "_CountryMortalityAbsolute_GEMM.csv"
        )
        np.savetxt(
            output_file,
            np.column_stack(
                (Cause_characters[:], GEMM_death_cty[:, 1 : nAvgLowHigh + 1])
            ),
            delimiter=",",
            fmt="%s",
            header=headerline1,
        )
        output_file = (
            out_path
            + disease
            + "_"
            + runName
            + "_"
            + tag
            + "_CountryMortalityFraction_GEMM.csv"
        )
        np.savetxt(
            output_file,
            np.column_stack(
                (Cause_characters[:], GEMM_mortrate_cty[:, 1 : nAvgLowHigh + 1])
            ),
            delimiter=",",
            fmt="%s",
            header=headerline1,
        )

    #### CALCULATE GRID-LEVEL DEATHS FOR DISEASES THAT HAVE AGE BINS
    else:
        GEMM_death_val = np.zeros(
            (len(ageBins) + 1, nAvgLowHigh, n_lat, n_lon)
        )  ##  1st dim for different age groups, 2nd dims for mid, lower, and upper values
        # grid level deaths in absolute numbers for each age group:
        for n in range(len(ageBins)):
            for m in range(nAvgLowHigh):
                GEMM_death_val[n, m, :, :] = (
                    Base_mortrate[n][:, :] * Cause_calcs_AF[m, n, :, :] * pop[n][:, :]
                )
        # summing age groups to get post-25 case:
        for m in range(nAvgLowHigh):
            GEMM_death_val[len(ageBins), m, :, :] = np.sum(
                GEMM_death_val[list(range(len(ageBins))), m, :, :], axis=0
            )
        GEMM_death_val[np.isnan(GEMM_death_val)] = 0.0

        #### CALCULATE GRID-LEVEL MORTALITY RATE
        GEMM_mortrate_val = np.zeros(
            (len(ageBins) + 1, nAvgLowHigh, n_lat, n_lon)
        )  ##  1st dim for different age groups, 2nd dims for mid, lower, and upper values
        # grid level deaths in fractional for each age group:
        for n in range(len(ageBins)):
            for m in range(nAvgLowHigh):
                GEMM_mortrate_val[n, m, :, :] = (
                    Base_mortrate[n][:, :] * Cause_calcs_AF[m, n, :, :]
                )
        # summing age groups to get post-25 case:
        for m in range(nAvgLowHigh):
            GEMM_mortrate_val[len(ageBins), m, :, :] = (
                np.sum(GEMM_death_val[list(range(len(ageBins))), m, :, :], axis=0)
                / tot_pop
            )
        GEMM_mortrate_val[np.isinf(GEMM_mortrate_val)] = 0.0

        #### PRINT A GLOBAL SUM FOR CHECK
        print("Burnett global:")
        print(
            np.sum(GEMM_death_val[len(ageBins), 0, :, :]),
            "(",
            np.sum(GEMM_death_val[len(ageBins), 1, :, :]),
            "-",
            np.sum(GEMM_death_val[len(ageBins), 2, :, :]),
            ")",
        )
        print(" ")

        #### GENERATE GRID-LEVEL OUTPUT FILE
        #### Output NC file of deaths and mortality rates
        f1 = Dataset(
            out_path + disease + "_" + runName + "_" + tag + "_Mortality_05x05_GEMM.nc",
            "w",
            format="NETCDF4_CLASSIC",
        )

        # Define coordinates/coordinate variables:
        ci = f1.createDimension(
            "ci", nAvgLowHigh
        )  ## value = 0 mean value, value = 1 lower value, value =2, upper value.
        lat = f1.createDimension("lat", len(latitude))
        lon = f1.createDimension("lon", len(longitude))
        lat = f1.createVariable("lat", np.double, ("lat",))
        lon = f1.createVariable("lon", np.double, ("lon",))
        ci = f1.createVariable(
            "ci", np.double, ("ci",)
        )  # formerly int, but that crashed on NCCS machine

        # Define rest of variables:
        Cause_death_out = []  # list probably not needed at all...
        Cause_mortrate_out = []  # list probably not needed at all...
        for n in range(len(ageBins)):
            Cause_death_out.append(
                f1.createVariable(
                    "Cause_death_" + ageBins[n], np.float32, ("ci", "lat", "lon")
                )
            )  # ,fill_value=0))
            Cause_mortrate_out.append(
                f1.createVariable(
                    "Cause_mortrate_" + ageBins[n], np.float32, ("ci", "lat", "lon")
                )
            )  # ,fill_value=0))
        Cause_death_out.append(
            f1.createVariable("Cause_death_post25", np.float32, ("ci", "lat", "lon"))
        )  # ,fill_value=0))
        Cause_mortrate_out.append(
            f1.createVariable("Cause_mortrate_post25", np.float32, ("ci", "lat", "lon"))
        )  # ,fill_value=0))

        # Global Attributes
        f1.description = "Calculated grid level deaths and mortality rate for the cause of disease, only average values are shown, the uncertainty ranges are in the .csv country output"
        f1.contact = "MuyeRu - muye.ru@duke.edu"

        # Variable Attributes
        lat.units = "degrees_north"
        lat.long_name = "latitude"
        lat.comment = "center of grid cell"
        lon.units = "degrees_east"
        lon.long_name = "longitude"
        lon.comment = "center of grid cell"
        ci.long_name = "mean, lower, upper"
        ci.comment = "ci = 0 mean value, ci = 1 lower value, ci =2, upper value"
        for n in range(len(ageBins) + 1):
            Cause_death_out[n].units = "person"
            Cause_death_out[
                n
            ].long_name = "annual total deaths due to the cause of disease"
            Cause_mortrate_out[n].units = "fraction"
            Cause_mortrate_out[
                n
            ].long_name = "annual mortality rate in fraction of population"

        # Fill in variables:
        f1.variables["lat"][:] = latitude[:]
        f1.variables["lon"][:] = longitude[:]
        f1.variables["ci"][:] = np.arange(nAvgLowHigh)
        for n in range(len(ageBins)):
            for m in range(nAvgLowHigh):
                f1.variables["Cause_death_" + ageBins[n]][m, :, :] = GEMM_death_val[
                    n, m, :, :
                ]
                f1.variables["Cause_mortrate_" + ageBins[n]][
                    m, :, :
                ] = GEMM_mortrate_val[n, m, :, :]
        for m in range(nAvgLowHigh):
            f1.variables["Cause_death_post25"][m, :, :] = GEMM_death_val[
                len(ageBins), m, :, :
            ]
            f1.variables["Cause_mortrate_post25"][m, :, :] = GEMM_mortrate_val[
                len(ageBins), m, :, :
            ]

        f1.close()

        #### CALCULATE COUNTRY LEVEL RESULTS AND GENERATE A .CSV FILE
        data = pop_path + "countryvalue_blank.csv"
        Cause_characters = np.genfromtxt(
            data, delimiter=",", usecols=(0), dtype=str, skip_header=1
        )
        # first column is country number then there are (# age groups X {avg, lower, upper} columns:
        ncol = 1 + (len(ageBins) + 1) * nAvgLowHigh
        GEMM_death_cty = np.zeros((num_countries, ncol))
        GEMM_death_cty[:, 0] = -1.0  # b/c replaced with country name as per Drew
        GEMM_mortrate_cty = np.zeros((num_countries, ncol))
        GEMM_mortrate_cty[:, 0] = -1.0  # b/c replaced with country name as per Drew

        for i in range(num_countries):
            temp_Pop = np.sum(
                fractionCountry[i, :, :] * tot_pop[:, :]
            )  ## Country population
            for m in range(nAvgLowHigh):
                m2 = m * (len(ageBins) + 1)
                for n in range(len(ageBins) + 1):
                    j = m2 + n + 1
                    GEMM_death_cty[i, j] = np.sum(
                        fractionCountry[i, :, :] * GEMM_death_val[n, m, :, :]
                    )
                    GEMM_mortrate_cty[i, j] = (
                        GEMM_death_cty[i, j] / temp_Pop
                    )  ## mortality rate out of total population in fraction
        GEMM_death_cty = np.round(GEMM_death_cty[:], 2)
        GEMM_mortrate_cty = np.round(GEMM_mortrate_cty[:], 7)

        #### country level
        suff = ["", "_low", "_high"]
        headerline1 = "CountryID"
        for m in range(nAvgLowHigh):
            for n in range(len(ageBins)):
                headerline1 += "," + ageBins[n] + suff[m]
            headerline1 += ",post25" + suff[m]
        output_file = (
            out_path
            + disease
            + "_"
            + runName
            + "_"
            + tag
            + "_CountryMortalityAbsolute_GEMM.csv"
        )
        np.savetxt(
            output_file,
            np.column_stack((Cause_characters[:], GEMM_death_cty[:, 1:ncol])),
            delimiter=",",
            fmt="%s",
            header=headerline1,
        )
        output_file = (
            out_path
            + disease
            + "_"
            + runName
            + "_"
            + tag
            + "_CountryMortalityFraction_GEMM.csv"
        )
        np.savetxt(
            output_file,
            np.column_stack((Cause_characters[:], GEMM_mortrate_cty[:, 1:ncol])),
            delimiter=",",
            fmt="%s",
            header=headerline1,
        )

        print(
            "Warning: more thought might be needed that all the per-capita output is correct (tot_pop)."
        )


####################################################################################################
####################################################################################################

if __name__ == "__main__":
    main(sys.argv[1:])
