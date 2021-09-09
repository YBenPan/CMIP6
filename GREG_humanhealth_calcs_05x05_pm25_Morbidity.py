# -*- coding: utf-8 -*-
"""
Created on 17 May 2019

@author: mr245
Greg Faluvegi editing for use on NCCS machine Fall 2019


Muye Ru edited on 06/26/20 to update the log-linear ERFs and include non-linear ERFs
"""

from netCDF4 import Dataset
import numpy as np
import sys

####################################################################################################
#### CALCULATES GLOBAL HUMAN-HEALTH MORBIDITY FROM VARIOUS CAUSES DUE TO PM2.5 BASED ON RISK FUNCTIONS
#### FROM CARBONH/HRAPIE PROJECT
#### COUNTERFACTUAL LEVEL IS 2.4 UG/M3
#### MORBIDITY ENDPOINTS INCLUDE:
#### 1. Work lost days (ages 15 to 64 years)
#### 2. Respiratory hospital admissions (all ages)
#### 3. Cardiovascular hospital admissions (all ages)
#### 4. Restricted activity days (all ages)
#### 5. Bronchitis in children (ages 6 to 12 years) - REQUIRE PM10 TO PM2.5 TRANSFORMATION
#### 6. Asthma symptom days in asthmatic children (ages 5 to 19 years) - REQUIRE PM10 TO PM2.5 TRANSFORMATION
#### 7. Chronic bronchitis in adults (ages ≥ 27 years) - REQUIRE PM10 TO PM2.5 TRANSFORMATION
#### OUTPUT INCLUDE:
####     1. GLOBAL NETCDF FILES OF CALCULATED MORBIDITY IN THE ABOVE 7 ENDPOINTS
####     2. A COUNTRY LEVEL .CSV SUMMARY FILE
####################################################################################################


def main(argv):
    runName, runYear, tag = sys.argv[1:]
    # for now, passing in a GISS run name, representative year, and a period tag (like 'ANN1990-1999')
    global m, out, vname, units, long_name, nAvgLowHigh, n_lat, n_lon

    #################################
    #### USER INPUT:
    #### SPECIFY YEAR:
    year = runYear
    #### WHAT IS THE BASE PATH TO THE GENERIC INPUT DATA (POPULATION, MORTALITY, COUNTRY FRACTIONS, ETC.)?
    base_path = "/discover/nobackup/projects/giss_ana/users/gfaluveg/newDecarbonHealth/From_Duke/PM_health_GEMM/healthdata/Generic/"
    #### WHAT IS THE BASE PATH TO THE PM25 CONC. DATA? MUST BE GRIDDED 05X05 GLOBAL FILES, WITH APPROPRIATE AVERAGING
    pm25_path = "/discover/nobackup/projects/giss_ana/users/gfaluveg/newDecarbonHealth/getPM2p5input/DATA/"
    #### WHAT IS CONCENTRATION FILE NAME?
    # pm25_file = 'GlobalGWRc_PM25_GL_'+year+'01_'+year+'12-RH35_05x05.nc'
    pm25_file = (
        "XtSS_DSWlpUf_PM2.5_microgram_m-3_" + tag + "_" + runName + "_0.5x0.5.nc"
    )
    #### WHAT IS THE NAME OF THE COUNTRY FRACTION FILE?
    fc_file = "countryFractions_2010_0.5x0.5.nc"
    #### WHAT IS THE GRID FILE OF THE RATIO BETWEEN PM2.5 AND PM10?
    ratio_file = "Country_PM2pt5_to_PM10_ratio_GISS.nc"
    out_path = "/discover/nobackup/projects/giss_ana/users/gfaluveg/newDecarbonHealth/PM_Output/"
    ####################################################################################################

    ####################################################################################################
    #### IMPORT NON-AGE BINNED DATASETS
    #### ALL DATA MUST BE IN 05X05 GRIDDED RESOLUTION.
    ####
    #### IMPORT ANNUAL AVG. PM25 DATA

    f1 = Dataset(pm25_path + pm25_file, "r")
    pm25 = f1.variables["sfcPM2p5"][:, :]
    latitude = f1.variables["lat"][:]
    longitude = f1.variables["lon"][:]
    pm25[np.isnan(pm25)] = 0.0  ## z value in the formula
    n_lat = len(latitude)
    n_lon = len(longitude)
    f1.close()

    #### COUNTERFACTUAL LEVEL IS 2.4 UG/M3
    zcf = 2.4
    pm25 = pm25[:, :] - zcf
    pm25[pm25[:, :] < 0] = 0.0

    #### IMPORT COUNTRY FRACTIONS
    f1 = Dataset(base_path + fc_file, "r")
    fractionCountry = f1.variables["fractionCountry"][:, :, :]
    num_countries = len(f1.variables["countryIndex"][:])  ## country index starts from 1
    f1.close()
    fractionCountry[fractionCountry < 0.0] = 0.0

    #### IMPORT RATIO OF PM2.5 TO PM10 - and get pm10, since you already have pm25
    f1 = Dataset(base_path + ratio_file, "r")
    pm25_pm10_ratio = f1.variables["PM2pt5_to_PM10_ratio"][:, :]
    f1.close()
    pm10 = np.zeros((n_lat, n_lon))
    x = np.where(pm25_pm10_ratio != 0.0)
    pm10[x] = pm25[x] / pm25_pm10_ratio[x]  # elsewhere already 0

    ####################################################################################################
    #### IMPORT AGE BINNED DATASETS
    #### ALL DATA MUST BE IN 05X05 GRIDDED RESOLUTION.
    #### IMPORT GRIDDED POPULATION DATA (UNITS = # PEOPLE)

    # CAREFUL! the Mortality program currently starts with age 25, so these age-bins would be different:
    # TODO: let's synchonize those two sets and import this from a common file:
    ageBins = [
        "0_4",
        "5_9",
        "10_14",
        "15_19",
        "20_24",
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

    # read in the total population and the fractional age bins (17 of them starting at age
    # 0 then 5, 10, 15, ... 80) and multiply them to get the population per bin into same
    # array as Duke case does:
    f_popTots = Dataset(
        "/discover/nobackup/gfaluveg/SHARE/SUPPORTING/populationData/griddedPopulation_"
        + year
        + "_0.5x0.5_OCT2017corrected.nc",
        "r",
    )
    f_popFrac = Dataset(
        "/discover/nobackup/gfaluveg/SHARE/SUPPORTING/populationData/griddedAgeBinFracs_"
        + year
        + "_0.5x0.5_OCT2017corrected.nc",
        "r",
    )
    tot_pop = f_popTots.variables["population"][:, :]
    pop = []
    for n in range(len(ageBins)):
        pop.append(tot_pop[:, :] * f_popFrac.variables["ageFractions"][n, :, :])
    f_popTots.close()
    f_popFrac.close()

    nAvgLowHigh = 3  # for loops over average, low-estimate, high-estimate
    names = ["Mean", "Low", "High"]
    out = []  # begin list of output
    vname = []  # and variable names in the nc output
    units = []  # units in the nc ouput
    long_name = []  # long name of variable in nc output
    m = -1  # index/counter (i.e. first will be 0)
    # "theCase" tuples below are used to fill in the variable name, units, long_name, ... of each morbidity cause

    def morbidity(r, bv, pm, p, tc):
        # calculates the increased risk due to PM(2.5 or 10) and grid level
        # output for the given population. Appends to lists of meta-data too.
        global m, out, vname, units, long_name, nAvgLowHigh, n_lat, n_lon
        m += 1  # index of the current output
        out.append(
            np.zeros((nAvgLowHigh, n_lat, n_lon))
        )  # create new list member of output
        vname.append(tc[0])
        units.append(tc[1])
        long_name.append(tc[2])
        for n in range(nAvgLowHigh):
            out[m][n, :, :] = (
                (r[n] ** (pm[:, :] / 10.0)) * bv[n] * p[:, :]
            )  # note per 10 ug m-3 form
        print("Global sum, " + long_name[m] + ":")
        print(
            int(np.sum(out[m][0, :, :])),
            "(",
            int(np.sum(out[m][1, :, :])),
            "-",
            int(np.sum(out[m][2, :, :])),
            ")\n",
        )

    ####################################################################################################
    #### 1. WORK LOST DAYS CALCULATION
    #### RR = 1.046 [1.039, 1.053] PER 10 UG/M3 PM2.5 INCREMENT,
    #### TYPICAL DAYS LOST PER WORKER IN EUROPE IS 9.4 (2,18) DAYS
    #### WORKER POPULATION INCLUDE FOR 15-64 POPULATION
    #### OUTPUT IS THE TOTAL WORK DAYS LOST DUE TO PM2.5 POLUTION
    RR = [1.046, 1.039, 1.053]
    baseVals = [9.4, 2.0, 18.0]
    theCase = ("work", "work days", "lost work days")
    bin0 = ageBins.index("15_19")
    binN = ageBins.index("60_64")
    popNow = np.sum(pop[bin0 : binN + 1], axis=0)
    morbidity(RR, baseVals, pm25, popNow, theCase)

    ####################################################################################################
    #### 2. RESPIRATORY HOSPITAL ADMISSIONS CALCULATION
    #### RR = 1.019 [95%CI: 0.998−1.040] PER 10 UG/M3 PM2.5 INCREMENT, FOR ALL AGE POPULATION,
    #### country-specific typical cases per 100,000 population in Europe is 1,165 (range: 528−2,170 cases)
    #### OUTPUT IS THE TOTAL RESPIRATORY HOPSITAL ADMISSION DUE TO PM2.5 POLUTION
    RR = [1.019, 0.998, 1.040]
    baseVals = [1165.0 / 100000.0, 528.0 / 100000.0, 2170.0 / 100000.0]
    theCase = ("res_hos", "hospital admissions", "resipiratory hospital admissions")
    morbidity(RR, baseVals, pm25, tot_pop, theCase)

    ####################################################################################################
    #### 3. CARDIOVASCULAR HOSPITAL ADMISSIONS CALCULATION
    #### RR = 1.0091 [95%CI: 1.0017−1.0166] PER 10 UG/M3 PM2.5 INCREMENT, FOR ALL AGE POPULATION,
    #### country-specific typical cases per 100,000 population in Europe is 2,256 (range: 661−4,239 days)
    #### OUTPUT IS THE TOTAL CARDIOVASCULAR HOPSITAL ADMISSION DUE TO PM2.5 POLUTION
    RR = [1.0091, 1.0017, 1.0166]
    baseVals = [2256.0 / 100000.0, 661.0 / 100000.0, 4239.0 / 100000.0]
    theCase = ("car_hos", "hospital admissions", "cardiovascular hospital admissions")
    morbidity(RR, baseVals, pm25, tot_pop, theCase)

    ####################################################################################################
    #### 4. RESTRICTED ACTIVITY DAYS CALCULATION
    #### RR = 1.047 [95%CI: 1.042−1.053] PER 10 UG/M3 PM2.5 INCREMENT, FOR ALL AGE POPULATION,
    #### TYPICAL RESTRICTED ACTIVITY DAYS PER WORKER IN EUROPE IS 19 DAYS
    #### OUTPUT IS THE TOTAL RESTRICTED ACTIVITY DAYS DUE TO PM2.5 POLUTION
    RR = [1.047, 1.042, 1.053]
    baseVals = [19.0, 19.0, 19.0]
    theCase = ("act", "days", "restricted activity days")
    morbidity(RR, baseVals, pm25, tot_pop, theCase)

    ####################################################################################################
    #### 5. BRONCHITIS IN CHILDREN
    #### NEED TRANSFORMATION FROM PM10 TO PM2.5
    #### FOR PM10, RR = 1.08 [95%CI: 0.95−1.19] PER 10 UG/M3 PM10 INCREMENT,
    #### FOR AGE 6-12,
    #### ANNUAL PREVALENCE RATE 0.186 × fpop IN EUROPE, WHICH WE APPLY WORLDWIDE FOR NOW
    #### OUTPUT IS THE TOTAL PREVALENCE OF CHILDREN BRONCHITIS DUE TO PM2.5 POLUTION
    RR = [1.08, 0.95, 1.19]
    baseVals = [0.186, 0.186, 0.186]  # but see note below about fpop
    theCase = (
        "child_bron",
        "cases of children bronchitis",
        "prevalence of children bronchitis among 6-12 population",
    )
    bin1 = ageBins.index("5_9")
    bin2 = ageBins.index("10_14")
    popNow = np.array(pop[bin1][:, :]) * (4.0 / 5.0) + np.array(pop[bin2][:, :]) * (
        3.0 / 5.0
    )
    # Muye and Drew had:
    # output_child_bron[n,:,:] = risk_child_bron[n,:,:]*0.186*fpop_6_12*pop_6_12
    # i.e. the fpop_6_12 really goes with the "baseVals" here, not the popNow.
    # But since it is functionally the same, apply to popNow so we can use the same morbidity function:
    # SAME NOTE for cases 6, 7 below.
    popNow[tot_pop > 0] = popNow[tot_pop > 0] * (
        popNow[tot_pop > 0] / tot_pop[tot_pop > 0]
    )
    morbidity(RR, baseVals, pm10, popNow, theCase)

    ####################################################################################################
    #### 6. ASTHMA SYMPTOM DAYS IN CHILDREN
    #### NEED TRANSFORMATION FROM PM10 TO PM2.5
    #### FOR PM10, RR = 1.028 [95%CI: 1.006−1.051] PER 10 UG/M3 PM10 INCREMENT,
    #### FOR AGE 5-19,
    #### RR FOR ASTHMA SYSMPTOM DAYS 3.04 × fpop FOR WEST EUROPE, AND 2.17 × fpop ELSEWHERE IN EUROPE,
    #### WE USE THE AVERAGE OF IT WORLDWIDE FOR NOW
    #### OUTPUT IS THE TOTAL CHILDREN ASTHMA SYMPTOM DAYS DUE TO PM2.5 POLUTION
    RR = [1.028, 1.006, 1.051]
    baseVals = [
        (3.04 + 2.17) / 2.0,
        (3.04 + 2.17) / 2.0,
        (3.04 + 2.17) / 2.0,
    ]  # but see note below about fpop
    theCase = (
        "child_asth",
        "cases of children asthma",
        "prevalence of children asthma among 5-19 population",
    )
    bin0 = ageBins.index("5_9")
    binN = ageBins.index("15_19")
    popNow = np.sum(pop[bin0 : binN + 1], axis=0)
    popNow[tot_pop > 0] = popNow[tot_pop > 0] * (
        popNow[tot_pop > 0] / tot_pop[tot_pop > 0]
    )
    morbidity(RR, baseVals, pm10, popNow, theCase)

    ####################################################################################################
    #### 7. CHRONIC BRONCHITIS IN ADULTS
    #### NEED TRANSFORMATION FROM PM10 TO PM2.5
    #### FOR PM10, RR = 1.117 [95%CI: 1.040−1.189] PER 10 UG/M3 PM10 INCREMENT,
    #### FOR AGE >=27,
    #### ANNUAL PREVALENCE RATE 0.0039 × fpop IN EUROPE, WHICH WE APPLY WORLDWIDE FOR NOW
    #### OUTPUT IS THE PREVALENCE OF ADULTS CHRONIC BRONCHITIS  DUE TO PM2.5 POLUTION
    RR = [1.117, 1.040, 1.189]
    baseVals = [0.0039, 0.0039, 0.0039]
    theCase = (
        "adult_bron",
        "cases of adult bronchitis",
        "prevalence of adult bronchitis among >=27 population",
    )
    bin0 = ageBins.index("25_29")
    binN = ageBins.index("80")
    popNow = np.sum(pop[bin0 : binN + 1], axis=0) - (2.0 / 5.0) * pop[bin0][:, :]
    popNow[tot_pop > 0] = popNow[tot_pop > 0] * (
        popNow[tot_pop > 0] / tot_pop[tot_pop > 0]
    )
    morbidity(RR, baseVals, pm10, popNow, theCase)

    nCause = m + 1
    print("Check: looks like I did " + str(nCause) + " endpoints.")
    #####################################################################################################
    #### OUTPUT NETCDF FILES OF GRID-LEVEL MORBIIDTY,
    #### AND CALCULATE COUNTRY LEVEL MORBIDITY, OUTPUT TO A .CSV FILE
    #####################################################################################################
    f1 = Dataset(
        out_path + runName + "_" + tag + "_Morbidity_05x05.nc",
        "w",
        format="NETCDF4_CLASSIC",
    )
    # create dimensions:
    lat = f1.createDimension("lat", n_lat)
    lon = f1.createDimension("lon", n_lon)
    ci = f1.createDimension("ci", nAvgLowHigh)
    # and coordinate variables:
    lat = f1.createVariable("lat", np.double, ("lat",))
    lon = f1.createVariable("lon", np.double, ("lon",))
    ci = f1.createVariable(
        "ci", np.double, ("ci",)
    )  # used to be int and that failed on NCCS machine

    # create all other variables:
    outVars = []
    for m in range(nCause):
        outVars.append(f1.createVariable(vname[m], np.float32, ("ci", "lat", "lon")))

    # Global Attributes
    f1.description = (
        "Calculated grid level morbidity with " + str(nCause) + " endpoints"
    )
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
    for m in range(nCause):
        outVars[m].units = units[m]
        outVars[m].long_name = long_name[m]

    # Fill all variables:
    f1.variables["lat"][:] = latitude[:]
    f1.variables["lon"][:] = longitude[:]
    f1.variables["ci"][:] = np.arange(nAvgLowHigh)
    for m in range(nCause):
        for n in range(nAvgLowHigh):
            f1.variables[vname[m]][n, :, :] = out[m][n, :, :]

    f1.close()

    #### CALCULATE COUNTRY LEVEL RESULTS AND GENERATE A .CSV FILE
    cdata = "/discover/nobackup/projects/giss_ana/users/gfaluveg/newDecarbonHealth/From_Duke/4health/countryvalue_blank.csv"
    cnames = np.genfromtxt(cdata, delimiter=",", usecols=(0), dtype=str, skip_header=1)
    output_cty = np.zeros((num_countries, nAvgLowHigh * nCause))

    for i in range(num_countries):
        for m in range(nCause):
            for n in range(nAvgLowHigh):
                nn = nAvgLowHigh * m + n
                output_cty[i, nn] = np.round(
                    np.sum(fractionCountry[i, :, :] * out[m][n, :, :]), 1
                )
    headerline1 = "CountryID"
    for m in range(nCause):
        for n in range(nAvgLowHigh):
            headerline1 += "," + vname[m] + "_" + names[n]
    output_file = out_path + runName + "_" + tag + "_Morbidity_05x05.csv"
    np.savetxt(
        output_file,
        np.column_stack((cnames[:], output_cty[:, :])),
        delimiter=",",
        fmt="%s",
        header=headerline1,
    )

    ####################################################################################################
    ####################################################################################################
    print("WARNING: pm10 calculation will understimate as is in areas with no pm2.5.")


if __name__ == "__main__":
    main(sys.argv[1:])
