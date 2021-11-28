# -*- coding: utf-8 -*-
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import os
import xarray as xr

####################################################################################################
#### CONVERTS THE .NPY FILES WITH COUNTRY LEVEL DATA TO A 0.5X0.5 DEGREE GLOBAL NETCDF FILE
#### THE ORDER OF THE COUNTRIES IN THE .NPY FILE IS FIXED AND CANNOT BE CHANGED
####################################################################################################

####################################################################################################
#### 05x05 COUNTRY FRACTIONS (PUT THE 6 FILES TOGETHER)
#### USER INPUT:
year = '2010' ## '2005' and '2016'
#### WHERE IS THE 0.1X0.1 DEGREE FILE WITH COUNTRY FRACTIONS?
base_path = 'F:\\Computer Programming\\Projects\\CMIP6\\data\\population\\'
#### WHAT IS THE NAME OF THE FILE WITH THE COUNTRY FRACTIONS?
#### IF 05X05
base_file = 'countryFractions_2010_0.5x0.5.nc'
#### IF 01X01
#base_file = 'CountryFractions_2010_0.1x0.1/countryFractions_2010_01x01_'
#### WHERE IS THE COUNTRY LEVEL .CSV FILE?
country_path = 'F:\\Computer Programming\\Projects\\CMIP6\\data\\population\\'
#### WHAT'S THE NAME OF THE COUNTRY LEVEL BASELINE FILE?
country_file = 'SSP1.nc'
#### WHERE ARE THE FILES WITH GRIDDED POPULATION? 
pop_path = 'D:\\CMIP6_data\\Pop\\'
#### Where to save the output
output_path = "D:\\CMIP6_data\\Pop_Result"
####################################################################################################

#### IMPORT COUNTRY FRACTIONS
#### IF 05X05
f1 = Dataset(base_path+base_file,'r')
fractionCountry = f1.variables['fractionCountry'][:,:,:] # countryIndex, latitude, longitude
latitude  = f1.variables['latitude'][:]
longitude = f1.variables['longitude'][:]
f1.close()

# #### IF 01x01 
# fractionCountry = np.zeros(shape = (193,1800,3600)) ## 193 is the number of countries 

#region
# f1              = Dataset(base_path+base_file+'a.nc', 'r')
# fractionCountry[0:25,:,:] = f1.variables['fractionCountry'][:,:,:]   
# latitude  = f1.variables['lat'][:]
# longitude = f1.variables['lon'][:]
# #num_countries   = len(f1.variables['countryIndex'][:])   ## country index starts from 1
# f1.close()
# f1              = Dataset(base_path+base_file+'b.nc', 'r')
# fractionCountry[25:50,:,:] = f1.variables['fractionCountry'][:,:,:]   
# #num_countries   = num_countries+len(f1.variables['countryIndex'][:])   ## country index stack on the previous one
# f1.close()
# f1              = Dataset(base_path+base_file+'c.nc', 'r')
# fractionCountry[50:75,:,:] = f1.variables['fractionCountry'][:,:,:] 
# #num_countries   = num_countries+len(f1.variables['countryIndex'][:])   
# f1.close()
# f1              = Dataset(base_path+base_file+'d.nc', 'r')
# fractionCountry[75:100,:,:] = f1.variables['fractionCountry'][:,:,:] 
# #num_countries   = num_countries+len(f1.variables['countryIndex'][:])   
# f1.close()
# f1              = Dataset(base_path+base_file+'e.nc', 'r')
# fractionCountry[100:125,:,:] = f1.variables['fractionCountry'][:,:,:] 
# #num_countries   = num_countries+len(f1.variables['countryIndex'][:])   
# f1.close()
# f1              = Dataset(base_path+base_file+'f.nc', 'r')
# fractionCountry[125:150,:,:] = f1.variables['fractionCountry'][:,:,:] 
# #num_countries   = num_countries+len(f1.variables['countryIndex'][:])   
# f1.close()
# f1              = Dataset(base_path+base_file+'g.nc', 'r')
# fractionCountry[150:175,:,:] = f1.variables['fractionCountry'][:,:,:]  
# #num_countries   = num_countries+len(f1.variables['countryIndex'][:])   
# f1.close()
# f1              = Dataset(base_path+base_file+'h.nc', 'r')
# fractionCountry[175:194,:,:] = f1.variables['fractionCountry'][:,:,:] 
# #num_countries   = num_countries+len(f1.variables['countryIndex'][:])   
# f1.close()
#endregion

fractionCountry[fractionCountry<0.0] = 0.0
fractionCountry[fractionCountry>1.0] = 0.0

# Change Longitude from -180 to 180 to 0 to 360 for ease of computation
fractionCountry = np.concatenate([fractionCountry[:,:,len(longitude) // 2:], fractionCountry[:,:, :len(longitude) // 2]], axis=2)

longitude=(longitude + 180) % 360 - 180

####################################################################################################    
#### IMPORT BASELINE NP ARRAYS
#### DIMENSION OF NP ARRAYS: [4, 193] FOR age_id, location_id
#### ALL VALUES ARE RATE PER 100K

ssps = ['ssp1']
age_groups = [1, 3, 6, 8] 
# 1: 0--19
# 2: 20--39
# 3: 40--64
# 4: 65+
years = np.arange(2010, 2020, 10)

data   = country_path+country_file
country_pop_ds = xr.open_dataset(data)
country_pop_all_age_values = country_pop_ds.data_vars['Population'].values
country_pop_ds.close()

# Select age groups
# Shape: Year, Age Group, Country ID
country_pop_values = country_pop_all_age_values[:,[1,3,6,8],:]

# Total Population of a Country: 
country_pop_sum = np.sum(country_pop_values, axis=1)

for ssp in ssps:
    for year in range(len(years)):
        # Import gridded population data
        grid_pop_file_name = f'{ssp}_tot_{str(years[year])}.nc'
        grid_pop_ds = xr.open_dataset(f'{pop_path}\\{ssp}\\{grid_pop_file_name}')
        grid_pop_values = grid_pop_ds.data_vars['population'].values 
        grid_pop_ds.close()

        pop_array = np.zeros(shape = (len(latitude), len(longitude), len(age_groups)))
        for age in range(len(age_groups)):

            for i in range(len(country_pop_values[year][age])):
                # print(i)

                ratio = country_pop_values[year, age, i] / country_pop_sum[year, i] if country_pop_sum[year, i] != 0 else 0

                pop_array[:,:,age] += fractionCountry[i,:,:] * grid_pop_values[:,:] * ratio
                
                ''' # Need to verify further
                if (i == 165):
                    for lat in range(360):
                        for lon in range(720):
                            if (fractionCountry[i][lat][lon] != 0):
                                print(
                                    -89.75 + 0.5 * lat, 
                                    0.25 + 0.5 * lon, 
                                    fractionCountry[i][lat][lon], grid_pop_values[lat][lon], 
                                    sep=' ')
                '''

        longitude = np.arange(0.25, 360, 0.5)
        
        da = xr.DataArray(
            data=pop_array,
            #dims=["lat", "lon", "age_group"],
            coords=dict(
                lat=latitude,
                lon=longitude,
                age_group=["0--19", "20--39", "40--64", "65+"],
            )
        )

        # Output

        os.makedirs(f'{output_path}\\{ssp}', exist_ok=True)
        
        da.to_netcdf(f'{output_path}\\{ssp}\\{str(year)}.nc')
        da.close()

# print(country_pop_values[24, 0, 35] / country_pop_sum[24, 35])


        



