from netCDF4 import Dataset
import numpy as np
import pandas as pd
import csv

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

# Drop countries

to_be_dropped = [
    "Aruba",
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

df = pd.read_csv(country_path + country_file, header=7)

# Combine Serbia and Montenegro

Serbia_index = [x for x in df.index.values if df['Area'].iloc[x] == 'Serbia']

Montenegro_index = [x for x in df.index.values if df['Area'].iloc[x] == 'Montenegro']

if (len(Serbia_index) != len(Montenegro_index)):
    raise Exception("Number of data points of Serbia is not equal to that of Montenegro!")

for i in range(0, len(Serbia_index)):
    df.at[Serbia_index[i], 'Population'] += df.at[Montenegro_index[i], 'Population']

# Drop Countries

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






