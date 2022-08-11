import pandas as pd
import itertools
import os

home_dir = "/home/ybenp"
countries_file = os.path.join(
    home_dir, "CMIP6_data", "population", "national_pop", "countryvalue_blank.csv"
)
countries_df = pd.read_csv(countries_file, usecols=["COUNTRY"])
country_names = [*countries_df["COUNTRY"].values, "World"]

# Classifications from https://www.healthdata.org/sites/default/files/files/Projects/GBD/GBDRegions_countries.pdf

# Super Region: Central Europe, Eastern Europe, and Central Asia
Central_Asia = [
    "Armenia",
    "Azerbaijan",
    "Georgia",
    "Kazakhstan",
    "Kyrgyzstan",
    "Mongolia",
    "Tajikistan",
    "Turkmenistan",
    "Uzbekistan",
]
Central_Europe = [
    "Albania",
    "Bosnia and Herzegovina",
    "Bulgaria",
    "Croatia",
    "Czech Republic",
    "Hungary",
    "Serbia+Montenegro",
    "Macedonia",
    "Poland",
    "Romania",
    "Slovakia",
    "Slovenia",
]
Eastern_Europe = [
    "Belarus",
    "Estonia",
    "Latvia",
    "Lithuania",
    "Moldova",
    "Russia",
    "Ukraine",
]

# Super Region: High-income
Australasia = ["Australia", "New Zealand"]
High_income_Asia_Pacific = ["Brunei", "Japan", "Singapore", "South Korea"]
High_income_North_America = ["Canada", "United States"]
Southern_Latin_America = ["Argentina", "Chile", "Uruguay"]
Western_Europe = [
    "Andorra",
    "Austria",
    "Belgium",
    "Cyprus",
    "Denmark",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Iceland",
    "Ireland",
    "Israel",
    "Italy",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Norway",
    "Portugal",
    "Spain",
    "Sweden",
    "Switzerland",
    "United Kingdom",
]

# Super Region: Latin America and Caribbean
Andean_Latin_America = ["Bolivia", "Ecuador", "Peru"]
Caribbean = [
    "Antigua and Barbuda",
    "The Bahamas",
    "Barbados",
    "Belize",
    "Cuba",
    "Dominica",
    "Dominican Republic",
    "Grenada",
    "Guyana",
    "Haiti",
    "Jamaica",
    "Puerto Rico",
    "Saint Lucia",
    "Saint Vincent and the Grenadines",
    "Suriname",
    "Trinidad and Tobago",
]
Central_Latin_America = [
    "Colombia",
    "Costa Rica",
    "El Salvador",
    "Guatemala",
    "Honduras",
    "Mexico",
    "Nicaragua",
    "Panama",
    "Venezuela",
]
Tropical_Latin_America = ["Brazil", "Paraguay"]

# Super Region: North Africa and Middle East
North_Africa_and_Middle_East = [
    "Afghanistan",
    "Algeria",
    "Bahrain",
    "Egypt",
    "Iran",
    "Iraq",
    "Jordan",
    "Kuwait",
    "Lebanon",
    "Libya",
    "Morocco",
    "Oman",
    "Qatar",
    "Saudi Arabia",
    "Syria",
    "Tunisia",
    "Turkey",
    "United Arab Emirates",
    "Yemen",
]

# Super Region: South Asia
South_Asia = ["Bangladesh", "Bhutan", "India", "Nepal", "Pakistan"]

# Super Region: Sub-Saharan Africa
Central_Sub_Saharan_Africa = [
    "Angola",
    "Central African Republic",
    "Congo",
    "Democratic Republic of the Congo",
    "Equatorial Guinea",
    "Gabon",
]
Eastern_Sub_Saharan_Africa = [
    "Burundi",
    "Comoros",
    "Djibouti",
    "Eritrea",
    "Ethiopia",
    "Kenya",
    "Madagascar",
    "Malawi",
    "Mozambique",
    "Rwanda",
    "Somalia",
    "Tanzania",
    "Uganda",
    "Zambia",
]
Southern_Sub_Saharan_Africa = [
    "Botswana",
    "Lesotho",
    "Namibia",
    "South Africa",
    "Swaziland",
    "Zimbabwe",
]
Western_Sub_Saharan_Africa = [
    "Benin",
    "Burkina Faso",
    "Cape Verde",
    "Cameroon",
    "Chad",
    "Cote d'Ivoire",
    "The Gambia",
    "Ghana",
    "Guinea",
    "Guinea-Bissau",
    "Liberia",
    "Mali",
    "Mauritania",
    "Niger",
    "Nigeria",
    "Sao Tome and Principe",
    "Senegal",
    "Sierra Leone",
    "Togo",
]

# Super Region: Southeast Asia, East Asia, and Oceania
East_Asia = ["China", "North Korea"]
Southeast_Asia = [
    "Cambodia",
    "Indonesia",
    "Laos",
    "Malaysia",
    "Maldives",
    "Mauritius",
    "Myanmar",
    "Philippines",
    "Seychelles",
    "Sri Lanka",
    "Thailand",
    "Timor-Leste",
    "Vietnam",
]

###############################################################

High_SDI = [
    "Switzerland",
    "Norway",
    "Monaco",
    "Germany",
    "Andorra",
    "Luxembourg",
    "Denmark",
    "San Marino",
    "Netherlands",
    "Canada",
    "South Korea",
    "Japan",
    "Iceland",
    "Singapore",
    # "Taiwan (province of China)",
    "Ireland",
    "United States",
    "Belgium",
    "Austria",
    "United Kingdom",
    "Cyprus",
    "Slovenia",
    "Australia",
    "New Zealand",
    "Lithuania",
    "France",
    "Estonia",
    "Czech Republic",
    "Brunei",
    "Latvia",
    # "Bermuda",
    "Slovakia",
    "Puerto Rico",
    "Italy",
    "Poland",
    "Greece",
    "Malta",
    "Croatia",
    "Sweden",
    "Finland",
]

High_middle_SDI = [
    # "Guam",
    "Israel",
    "Russia",
    "The Bahamas",
    "Hungary",
    "Saudi Arabia",
    # "Montenegro",
    "Oman",
    "Spain",
    "Serbia+Montenegro",
    # "Northern Mariana Islands",
    "Bulgaria",
    "Trinidad and Tobago",
    # "Greenland",
    "Romania",
    "Cook Islands",
    "Chile",
    "Barbados",
    "Macedonia",
    "Belarus",
    "Ukraine",
    "Portugal",
    "St.Kitts+Nevis",
    "Antigua and Barbuda",
    "Palau",
    "Turkey",
    "Malaysia",
    "Kazakhstan",
    "Libya",
    # "American Samoa",
    "Niue",
    "United Arab Emirates",
    "Kuwait",
    "Qatar",
    "Argentina",
    "Mauritius",
    "Lebanon",
    "Georgia",
    "Armenia",
    "Azerbaijan",
    "Panama",
    "Iran",
    "Turkmenistan",
    "Cuba",
]

Middle_SDI = [
    "Bahrain",
    "Dominica",
    "Jordan",
    "Bosnia and Herzegovina",
    "Seychelles",
    "Uruguay",
    "Moldova",
    "Jamaica",
    "Sri Lanka",
    "Thailand",
    "Albania",
    "South Africa",
    "Costa Rica",
    "China",
    "Saint Lucia",
    "Equatorial Guinea",
    "Tunisia",
    "Grenada",
    "Fiji",
    "Indonesia",
    "Mexico",
    "Peru",
    "Algeria",
    "Egypt",
    "Brazil",
    "Ecuador",
    "Suriname",
    "Paraguay",
    "Botswana",
    "Uzbekistan",
    "Colombia",
    "Venezuela",
    "Guyana",
    "Philippines",
    "Vietnam",
    "Mongolia",
    "Dominican Republic",
    "El Salvador",
    "Maldives",
]

Low_middle_SDI = [
    "Iraq",
    "Samoa",
    "Gabon",
    "Tonga",
    "Syria",
    "Namibia",
    "Belize",
    "Nauru",
    "Kyrgyzstan",
    "Tuvalu",
    "Federated States of Micronesia",
    "Swaziland",
    # "Palestine",
    "Bolivia",
    "North Korea",
    "Congo",
    "India",
    "Marshall Islands",
    "Ghana",
    "Tajikistan",
    "Morocco",
    "Guatemala",
    "Timor-Leste",
    "Cape Verde",
    "Nicaragua",
    "Myanmar",
    "Nigeria",
    "Lesotho",
    "Honduras",
    "Kenya",
    "Sudan",
    "Zambia",
    "Vanuatu",
    "Mauritania",
    "Laos",
    "Cameroon",
    "Zimbabwe",
    "Bangladesh",
    "Cambodia",
    "Bhutan",
    "Pakistan",
    "Nepal",
]

Low_SDI = [
    "Angola",
    "Comoros",
    "Djibouti",
    "Haiti",
    "Yemen",
    "Rwanda",
    "Tanzania",
    "Solomon Islands",
    "Togo",
    "Papua New Guinea",
    "Cote d'Ivoire",
    "Uganda",
    "The Gambia",
    "Madagascar",
    "Eritrea",
    "Senegal",
    "Malawi",
    "Liberia",
    # "South Sudan",
    "Democratic Republic of the Congo",
    "Guinea-Bissau",
    "Benin",
    "Sierra Leone",
    "Afghanistan",
    "Ethiopia",
    "Guinea",
    "Mozambique",
    "Burundi",
    "Central African Republic",
    "Mali",
    "Burkina Faso",
    "Chad",
    "Niger",
    "Somalia",
]

GBD_regions = [
    # Central Europe, Eastern Europe, and Central Asia
    Central_Asia,
    Central_Europe,
    Eastern_Europe,
    # High-income
    Australasia,
    High_income_Asia_Pacific,
    High_income_North_America,
    Southern_Latin_America,
    Western_Europe,
    # Latin America and Caribbean
    Andean_Latin_America,
    Caribbean,
    Central_Latin_America,
    Tropical_Latin_America,
    # North Africa and Middle East
    North_Africa_and_Middle_East,
    # South Asia
    South_Asia,
    # Sub-Saharan Africa
    Central_Sub_Saharan_Africa,
    Eastern_Sub_Saharan_Africa,
    Southern_Sub_Saharan_Africa,
    Western_Sub_Saharan_Africa,
    # Southeast Asia, East Asia, and Oceania
    East_Asia,
    Southeast_Asia,
]

SDI_regions = [High_SDI, High_middle_SDI, Middle_SDI, Low_middle_SDI, Low_SDI]

regions = SDI_regions

all_countries = list(itertools.chain.from_iterable(regions))

# Bijective verification

# None missing
# if all(country in country_names for country in all_countries):
#     print("Check complete: all region countries are in the official country list")
# else:
#     for country in all_countries:
#         if country not in country_names:
#             print(f"Country {country} not in official list!")

# Missing countries: Kiribati, Saint Vincent and the Grenadines, Sao Tome and Principe
# if all(country in all_countries for country in country_names):
#     print("Check complete: all countries in the official country list are in one of the regions")
# else:
#     for country in country_names:
#         if country not in all_countries:
#             print(f"Country {country} not in region lists")


    
