from netCDF4 import Dataset
import numpy as np
import pandas as pd
import csv
import xarray as xr

from pandas.core.indexes import multi

####################################################################################################
#### 05x05 COUNTRY FRACTIONS (PUT THE 6 FILES TOGETHER)
#### USER INPUT:
#### WHERE IS THE COUNTRY LEVEL .CSV FILE?
country_path = 'F:\\Computer Programming\\Projects\\CMIP6\\data\\population\\'
#### WHAT'S THE NAME OF THE COUNTRY LEVEL BASELINE FILE?
country_file = 'wcde_data_SSP1.csv'
#### WHERE IS THE COUNTRY NAME .CSV FILE?
name_path = 'F:\\Computer Programming\\Projects\\CMIP6\\data\\population\\'
#### WHAT'S THE NAME OF THE COUNTRY NAME FILE?
name_file = 'countryvalue_blank.csv'
#### Where to save the output
output_path = 'F:\\Computer Programming\\Projects\\CMIP6\\data\\population\\'
####################################################################################################

# Note to self:
# 1. Should country_name be index of dataframe? 
# 2. Should country_id be 0-based or 1-based?
# 3. Order of sorting/columns

df = pd.read_csv(country_path + country_file, header=7)

# Add countries

to_be_added = [
    "Andorra",
    "Cook Islands",
    "Dominica",
    "Marshall Islands",
    "Monaco",
    "Nauru",
    "Niue",
    "Palau",
    "San Marino",
    "St.Kitts+Nevis",
    "Tuvalu",
]

years = df['Year'].drop_duplicates().tolist()

ages = df['Age'].drop_duplicates().tolist()

new_rows = [pd.Series([area, year, age, 0], index=df.columns) for area in to_be_added for year in years for age in ages]

df = df.append(new_rows)

# Combine Serbia and Montenegro

Serbia_index = [x for x in df.index.values if df['Area'].iloc[x] == 'Serbia']

Montenegro_index = [x for x in df.index.values if df['Area'].iloc[x] == 'Montenegro']

if (len(Serbia_index) != len(Montenegro_index)):
    raise Exception("Number of data points of Serbia is not equal to that of Montenegro!")

for i in range(0, len(Serbia_index)):
    df.at[Serbia_index[i], 'Population'] += df.at[Montenegro_index[i], 'Population']


# Drop Countries

to_be_dropped = [
    "Aruba",
    "Channel Islands",
    "Curaçao",
    "French Guiana",
    "French Polynesia",
    "Guadeloupe",
    "Guam",
    "Hong Kong Special Administrative Region of China",
    "Macao Special Administrative Region of China",
    "Martinique",
    "Mayotte",
    "Montenegro",
    "New Caledonia",
    "Occupied Palestinian Territory",
    "Reunion",
    "South Sudan",
    "Taiwan Province of China",
    "United States Virgin Islands",
    "Western Sahara",
    "World",
]


countries = df['Area'].drop_duplicates().tolist()

countries = [x for x in countries if x not in to_be_dropped]

df = df.set_index('Area')

df = df.loc[countries]

# Rename countries

to_be_renamed = {
    "Bahamas": "The Bahamas",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Brunei Darussalam": "Brunei",
    "Democratic People's Republic of Korea": "North Korea",
    "Gambia": "The Gambia",
    "Iran (Islamic Republic of)": "Iran",
    "Lao People's Democratic Republic": "Laos",
    "Libyan Arab Jamahiriya": "Libya",
    "Micronesia (Federated States of)": "Federated States of Micronesia",
    "Republic of Korea": "South Korea",
    "Republic of Moldova": "Moldova",
    "Russian Federation": "Russia",
    "Serbia": "Serbia+Montenegro",
    "Syrian Arab Republic": "Syria",
    "The former Yugoslav Republic of Macedonia": "Macedonia",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "United Republic of Tanzania": "Tanzania",
    "United States of America": "United States",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Viet Nam": "Vietnam",
}

labels = df.index.values

new_labels = list(map(lambda x: to_be_renamed[x] if x in to_be_renamed else x, labels))

df = df.set_axis(new_labels, axis='index')

df.index.name = "Area"

# Import country ids

countries_df = pd.read_csv(name_path + name_file, header=0)

country_names = list(countries_df['COUNTRY'])

country_ids = list(countries_df['COUNTRY_NO'] - 1)

country_dict = dict(zip(country_names, country_ids))

# Create country id column and Sort

new_country_ids = [country_dict[x] for x in df.index.values]

if (len(new_country_ids) != len(df)):
    raise Exception("Length of country IDs is not equal to length of dataframe!")

df.insert(loc=0, column="Country_ID", value=new_country_ids)

df.reset_index()

df = df.sort_values(by = ['Year', 'Age', 'Country_ID'])

multi_df = df.set_index(['Country_ID', 'Year', 'Age'])

# ds = multi_df.to_xarray()

multi_df.to_csv(output_path + 'SSP1.csv')
