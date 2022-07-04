#cell 0
import compute_AF_ages
import numpy as np
from itertools import groupby
import os
from glob import glob
import datetime as dt
import csv
import dask as dask
from netCDF4 import Dataset
import multiprocessing as mp
import pandas as pd

#cell 1
def process_file_name(file_name):
#
# manipulate file_name string to get info needed by GEMM and MORBIDITY scripts
#
    nums=[int(''.join(i)) for is_digit, i in groupby(file_name, str.isdigit) if is_digit]
    runYear = int(np.average(nums[-1:]))
#
# Note: this will be adapted if parentdir changes
#
    wk = file_name.split("/")
    runName = wk[10] + "_" + wk[11]
#
    return runYear,runName

#cell 2
def get_value(data,parameter,type_of_data):
#
# get values from dataframe
#
    wk0 = data[data["parameter"] == parameter]
    wk1 = wk0["value"]
    wk2 = wk1.iloc[0]
    value = wk2
#
    if type_of_data == "string":
        value = str(wk2)
    if type_of_data == "logical":
        value = bool(int(wk2))
    if type_of_data == "float":
        value = float(wk2)
    if type_of_data == "integer":
        value = int(wk2)
#
#    print(parameter,value)
#
    return value

#cell 3
def get_pm25(file_name,scaling):
#
    f1 = Dataset(file_name, "r")
    lat = f1.variables["lat"][:]
    lon = f1.variables["lon"][:]
    pm25 = f1.variables["concpm2p5"][:,:]
    pm25[np.isnan(pm25)] = 0.0
    pm25[:,:] = pm25[:,:]  * scaling
    f1.close()
#
    return np.array(lat),np.array(lon),np.array(pm25)

#cell 4
def get_pop(pop_base_path,runYear,ssp_name,constant_pop,constant_pop_year):
#
# define population year, based on the runYear
#
    wk = int(runYear)
    popYear = str(wk-wk%10)
#
    if constant_pop:
        popYear = str(constant_pop_year)
#
# get population file
#
    file_name = pop_base_path + "/" + ssp_name + "/" + popYear + ".nc"
    if debug:
        print("Population_file: " + file_name)
    f1 = Dataset(file_name, "r")
    nlat = len(f1.variables["lat"])
    nlon = len(f1.variables["lon"])
    npop = 0
    pop_names = []
    for name in list(f1.variables):
        if (name.find('age') != -1):
            npop = npop + 1
            pop_names.append(name)
#   
    pop = np.zeros((npop,nlat,nlon),float)
    for i in range(npop):
        pop[i,:,:] = f1.variables[pop_names[i]][:,:]
#
# add post100 to age_95_99
#
    if debug:
        print(i)
    pop[i,:,:] += f1.variables["post100"][:,:]
#
# change name
#
    pop_names[i] = "post95"
    pop[np.isnan(pop)] = 0.0         
    f1.close()
#
    if debug:
        print("npop =",npop)
        print(pop_names)    
#
# change units to 100k to match the baseline data
#
    pop[:,:,:] = pop[:,:,:]/1.e5
#
    if debug:
        print(np.sum(pop))
#
    return np.array(pop),npop,pop_names

#cell 5
def get_baseline(full_path,ndis,nlon,nlat,npop,pop_names,baseline_est):
#
# number of baseline estimates
#
    nbas = len(baseline_est)
#        
    baseline = np.zeros((ndis,nbas,npop,nlat,nlon),float)
#
# from compute_AF.py
# 'T2D','Allcause','IHD','Stroke','COPD','LC','LRI','DEM'
# 
    file_name = full_path + "/Diabetes.nc"
    baseline[0,:,:,:,:] = read_baseline(file_name,nbas,npop,nlon,nlat,pop_names,baseline_est)
    file_name = full_path + "/NonCommunicableDiseases.nc"
    baseline[1,:,:,:,:] = read_baseline(file_name,nbas,npop,nlon,nlat,pop_names,baseline_est)
    file_name = full_path + "/IschemicHeartDisease.nc"
    baseline[2,:,:,:,:] = read_baseline(file_name,nbas,npop,nlon,nlat,pop_names,baseline_est)
    file_name = full_path + "/Stroke.nc"
    baseline[3,:,:,:,:] = read_baseline(file_name,nbas,npop,nlon,nlat,pop_names,baseline_est)
    file_name = full_path + "/COPD.nc"
    baseline[4,:,:,:,:] = read_baseline(file_name,nbas,npop,nlon,nlat,pop_names,baseline_est)
    file_name = full_path + "/LungCancer.nc"
    baseline[5,:,:,:,:] = read_baseline(file_name,nbas,npop,nlon,nlat,pop_names,baseline_est)
    file_name = full_path + "/LowerRespiratoryInfections.nc"
    baseline[6,:,:,:,:] = read_baseline(file_name,nbas,npop,nlon,nlat,pop_names,baseline_est)
    file_name = full_path + "/Dementia.nc"
    baseline[7,:,:,:,:] = read_baseline(file_name,nbas,npop,nlon,nlat,pop_names,baseline_est)
#
    return np.array(baseline)      

#cell 6
def read_baseline(file_name,nbas,npop,nlon,nlat,pop_names,baseline_est):
#
# define array
#
    baseline = np.zeros((nbas,npop,nlat,nlon),float)
#
    f1 = Dataset(file_name, "r")
#
    lon = f1.variables["lon"][:]
    lon_wrap = np.zeros(nlon,float)
#
    for i1 in range(len(pop_names)):
        for i2 in range(len(baseline_est)):
            var_name = pop_names[i1] + "_" + baseline_est[i2]
#
# make sure this variable is in the file
#
            found_variable = False
            for name in list(f1.variables):
                if (name == var_name):
                    found_variable = True
#
            if found_variable:
                if debug:
                    print("Reading ",var_name)
                wk = f1.variables[var_name][:,:]
#
# flip longitudes to be [0,360]
#
                if lon.min() < 0:
                    for j in range(nlat):
                        wk1d = wk[j,:]
                        lon_wrap[0:360] = wk1d[360:720]
                        lon_wrap[360:720] = wk1d[0:360]
                        wk[j,:] = lon_wrap
#
                baseline[i2,i1,:,:] = wk[:,:]
#
    f1.close()
#
    return np.array(baseline)

#cell 7
def read_country_fractions():
    
    file_name = "/glade/work/lamar/CMIP6_analysis/PM2.5/Health/countryFractions_2010_0.5x0.5.nc"
    f1 = Dataset(file_name, "r")
    fractionCountry = f1.variables["fractionCountry"][:, :, :]
    num_countries = len(f1.variables["countryIndex"][:])  ## country index starts from 1
    f1.close()
    fractionCountry[fractionCountry < 0.0] = 0.0
    return np.array(fractionCountry)

#cell 8
def write_output_country(output_dir,runName,runYear,baseline,disease,pop_names,mortality,fractionCountry):
#
    wk_shape = fractionCountry.shape
    num_countries = wk_shape[0]
    wk_shape = mortality.shape
    num_estimates = wk_shape[0]
    mortality_country = np.zeros((num_countries,num_estimates),float)
#
    #tic = dt.datetime.now()
#
    output_arr = np.zeros((num_countries + 1, 1 + num_estimates * (len(pop_names) + 1)))
    output_arr[:-1, 0] = np.arange(1,194,dtype=int) 
#
# create output dir
#
    output_dir_full = output_dir + "/CountryMortalityAbsolute/" + disease + "_" + baseline
    if debug:
        print(output_dir_full)
    os.makedirs(output_dir_full, exist_ok=True)
#
# write csv file
#
    output_file = output_dir_full + "/all_ages_" + runName + "_" + runYear + "_GEMM.csv"
    if debug:
        print("Output file:" + output_file)
    header_name = "CountryID"
#
    for k in range(len(pop_names)):
        header_name = f"{header_name},{pop_names[k]}_Mean,{pop_names[k]}_Low,{pop_names[k]}_High"
#
        for j in range(num_countries):
            wk0 = fractionCountry[j,:,:]
            wk1 = wk0[wk0>0]
            for n in range(num_estimates):
                wk2 = mortality[n,k,:,:]
                wk2 = wk2[wk0>0]
                mortality_country[j,n] = np.sum(wk1*wk2)
#

        output_arr[:-1, num_estimates * k + 1] = mortality_country[:, 0]
        output_arr[:-1, num_estimates * k + 2] = mortality_country[:, 1]
        output_arr[:-1, num_estimates * k + 3] = mortality_country[:, 2]
#
# Add all age
#
    header_name = f"{header_name},all_age_Mean, all_age_Low,all_age_High"
    output_arr[:-1, -3] = np.sum(output_arr[:-1, 1:-3:3], axis=1)
    output_arr[:-1, -2] = np.sum(output_arr[:-1, 2:-2:3], axis=1)
    output_arr[:-1, -1] = np.sum(output_arr[:-1, 3:-1:3], axis=1)
#
# add global total
#
    output_arr[num_countries,:] = np.sum(output_arr[:-1,:], axis=0)
#
    output_arr = np.round(output_arr[:,:], 2)
    output_arr[num_countries,0] = -99
#
    if debug:
        print("country output ",output_file)
    np.savetxt(output_file, output_arr, delimiter=",", header=header_name,fmt='%1.2f')
    #tac = dt.datetime.now()
    #print("write_output_country takes:", tac - tic)
#

#cell 9
def write_output_netcdf(output_dir,runName,runYear,baseline,disease,lat_05,lon_05,mortality):
#
# create output dir
#
    output_dir_full = output_dir + "/MortalityAbsolute/" + disease + "_" + baseline
    if debug:
        print(output_dir_full)
    os.makedirs(output_dir_full, exist_ok=True)
#
# write nc file
#
    output_file = output_dir_full + "/" + runName + "_" + runYear + "_GEMM.nc"
    if debug:
        print("Output file:" + output_file)
#
# make sure the file does not already exists
#
    if os.path.exists(output_file):
        os.remove(output_file)
#
    f1 = Dataset(output_file,"w",format="NETCDF4_CLASSIC")
#
# Global Attributes
#
    f1.description = (
        "Attributable deaths to PM2.5 for each disease"
    )
    f1.contact = "Jean-Francois Lamarque, lamar@ucar.edu"
#
# define coordinates/coordinate variables:
#
    lat = f1.createDimension("lat", len(lat_05))
    lon = f1.createDimension("lon", len(lon_05))
    lat = f1.createVariable("lat", np.float32, ("lat",))
    lon = f1.createVariable("lon", np.float32, ("lon",))
    lat.units = "degrees_north"
    lat.long_name = "latitude"
    lat.comment = "center of grid cell"
    lon.units = "degrees_east"
    lon.long_name = "longitude"
    lon.comment = "center of grid cell"
    f1.variables["lat"][:] = lat_05[:]
    f1.variables["lon"][:] = lon_05[:]
#
    wk_shape = mortality.shape
    num_estimates = wk_shape[0]
#
# only output the total for all ages
#
    wk_mortality = np.sum(mortality[:,:,:,:],axis=1)
#
# define other variables:
#
    suff = ["_mean", "_low", "_up"]
#
# define variables
#
    var_out = []
    for n in range(num_estimates):
        var_out.append(f1.createVariable("deaths_" + suff[n], np.float32, ("lat", "lon")))
#
# add Attributes
#
    cnt = 0
    for n in range(num_estimates):
        var_out[cnt].units = "deaths, all ages"
        var_out[cnt].long_name = "Mortality attributable to PM2.5"
#
        f1.variables["deaths_" + suff[n]][:, :] = wk_mortality[n, :, :]
#
        cnt += 1
#
    f1.close()
#        

#cell 10
#
# start timer
#
t0 = dt.datetime.now()
#
# read parameters from external file
#
data = pd.read_csv('data.csv')
#
chinese_cohort = get_value(data,"chinese_cohort","logical")
functional_form = get_value(data,"functional_form","string")
GEMM_path = get_value(data,"GEMM_path","string")
pop_path = get_value(data,"pop_path","string")
constant_pop = get_value(data,"constant_pop","logical")
constant_pop_year = get_value(data,"constant_pop_year","integer")
refMortYear = get_value(data,"refMortYear","string")
baseline_data = get_value(data,"baseline_data","string")
baseline_path = get_value(data,"baseline_path","string")
baseline_path = baseline_path + baseline_data + "_mortality_baseline_" + refMortYear
ssp_name = get_value(data,"ssp_name","string")
pm25_dir = get_value(data,"pm25_dir","string") + ssp_name
scaling = get_value(data,"scaling","float") # scaling for concentration (to end in ug/m3)
year_max = get_value(data,"year_max","integer")
write_output_csv = get_value(data,"write_output_csv","logical")
write_output_nc  = get_value(data,"write_output_nc","logical")
#
global debug
debug = get_value(data,"debug","logical")
#
# baseline estimate naming convention
#
baseline_est = ["mean","lower","upper"]
#
# define output directories
#
output_dir = "/glade/scratch/lamar/tmp/PM2.5/Baseline_Ben_" + refMortYear + "_" + baseline_data + "/5_years/" + ssp_name
#
# create list of PM2.5 files
#
parent_dir = pm25_dir + "/mmrpm2p5/*/*/*nc"
#
# loop over all PM2.5 files
#
initial_step = True
#
for file_name in sorted(glob(parent_dir)):
#
# extract run information from file name
#
    runYear,runName = process_file_name(file_name)
#
# process until year_max
#
    if runYear > year_max:
        continue
#
    print("Processing " + file_name)
#
# extract PM2.5
#
    if debug:
        print("get_pm25")
    lat,lon,pm25 = get_pm25(file_name,scaling)
    nlon = len(lon)
    nlat = len(lat)
    minval = pm25.min()
    if debug:
        print("Min PM25: ",minval)
    maxval = pm25.max()
    if debug:
        print("Max PM25: ",maxval)
#
# get population
#    define population year, based on the runYear
#    map SSPs (see http://dx.doi.org/10.1016/j.gloenvcha.2014.06.004)
#
    refPopYear = str(runYear - runYear%10)
    runYear    = str(runYear)
    ssp_name_pop = ssp_name[0:4]
    if ssp_name[0:4] == "ssp5":
        ssp_name_pop = "ssp1"
    if ssp_name[0:4] == "ssp4":
        ssp_name_pop = "ssp2"
#
# read population
#
    if debug:
        print("Population:",ssp_name_pop)
    pop,npop,pop_names = get_pop(pop_path,runYear,ssp_name_pop,constant_pop,constant_pop_year)
#
    if debug:
        print("npop =",npop)
        print(pop_names)
        print("Global population:",np.sum(pop))
#
# compute AF
#
    if debug:
        print("compute AF")
    diseases,disease_AF = compute_AF_ages.main(GEMM_path,chinese_cohort,functional_form,pop_names,pm25,nlon,nlat)
#
# special actions if first pass through
#
    if initial_step:
#
# update output directory to reflect population
#
#
        pop_out_name = "Pop_" + ssp_name_pop + "_var" 
        if constant_pop:
            pop_out_name = "Pop_" + ssp_name_pop + "_" + str(constant_pop_year)
        output_dir = output_dir + "/" + pop_out_name
#
# get baseline mortality
#
        if debug:
            print("get_baseline")
        baseline = get_baseline(baseline_path,len(diseases),nlon,nlat,npop,pop_names,baseline_est)
        wk_shape = baseline.shape
        nbas = wk_shape[1]
        ndis = wk_shape[0]
        del wk_shape
#
        wk_shape = disease_AF.shape
        nest = wk_shape[1]
        del wk_shape
#
# read country fractions
#
        country_fractions = read_country_fractions()
#
# define output array
#
        mortality = np.zeros((nest,npop,nlat,nlon),float)
#
# compute additional mortality
#
    pool = mp.Pool()
    #print(pool)
    feed_csv_args = []
    feed_nc_args  = []
    for i in range(nbas):
        for m in range(ndis):
            wk_bas = baseline[m,i,:,:,:]
            for n in range(nest):
                wk_AF = disease_AF[m,n,:,:,:]
                for j in range(npop):
                    mortality[n,j,:,:] = (pop[j,:,:] * wk_bas[j,:,:]) * wk_AF[j,:,:]
                    #mortality[n,j,:,:] = pop[j,:,:]
#
# write output
#
            if write_output_csv or write_output_nc:
                mortality_copy = np.copy(mortality)
#
            if debug:
                print("Write output")
            if  write_output_csv:
                if debug:
                    print(np.sum(mortality))
                feed_csv_args.append((output_dir, runName, runYear, baseline_est[i], diseases[m], pop_names, mortality_copy, country_fractions))
            if write_output_nc:
                feed_nc_args.append((output_dir,runName,runYear,baseline_est[i],diseases[m],lat,lon,mortality_copy))
#
        # sys.exit()
#
# parallelize output
#
    if  write_output_csv:
        pool.starmap(write_output_country, feed_csv_args)
    if write_output_nc:
        pool.starmap(write_output_netcdf, feed_nc_args )
#
# make sure constant data is only read once
#
    initial_step = False        
# 
    pool.close()
    pool.join()

#cell 11
#
# end timer
#
t1 = dt.datetime.now()
print((t1-t0).total_seconds())

