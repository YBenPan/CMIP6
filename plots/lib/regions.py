import pandas as pd
import os

home_dir = "/home/ybenp"
countries_file = os.path.join(home_dir, "CMIP6_data", "population", "national_pop", "countryvalue_blank.csv")
countries_df = pd.read_csv(countries_file, usecols=["COUNTRY"])
country_names = [*countries_df["COUNTRY"].values, "World"]

# Classifications from https://www.healthdata.org/sites/default/files/files/Projects/GBD/GBDRegions_countries.pdf

# Super Region: Central Europe, Eastern Europe, and Central Asia
Central_Asia = ["Armenia", "Azerbaijan", "Georgia", "Kazakhstan", "Kyrgyzstan", "Mongolia", "Tajikistan",
                "Turkmenistan", "Uzbekistan"]
Central_Europe = ["Albania", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Czech Republic", "Hungary",
                  "Serbia+Montenegro", "Macedonia", "Poland", "Romania", "Slovakia", "Slovenia"]
Eastern_Europe = ["Belarus", "Estonia", "Latvia", "Lithuania", "Moldova", "Russia", "Ukraine"]

# Super Region: High-income
Australasia = ["Australia", "New Zealand"]
High_income_Asia_Pacific = ["Brunei", "Japan", "Singapore", "South Korea"]
High_income_North_America = ["Canada", "United States"]
Southern_Latin_America = ["Argentina", "Chile", "Uruguay"]
Western_Europe = ["Andorra", "Austria", "Belgium", "Cyprus", "Denmark", "Finland", "France", "Germany", "Greece",
                  "Iceland", "Ireland", "Israel", "Italy", "Luxembourg", "Malta", "Netherlands", "Norway",
                  "Portugal", "Spain", "Sweden", "Switzerland", "United Kingdom"]

# Super Region: Latin America and Caribbean
Andean_Latin_America = ["Bolivia", "Ecuador", "Peru"]
Caribbean = ["Antigua and Barbuda", "The Bahamas", "Barbados", "Belize", "Cuba", "Dominica",
             "Dominican Republic", "Grenada", "Guyana", "Haiti", "Jamaica", "Puerto Rico", "Saint Lucia",
             "Saint Vincent and the Grenadines", "Suriname", "Trinidad and Tobago"]
Central_Latin_America = ["Colombia", "Costa Rica", "El Salvador", "Guatemala", "Honduras", "Mexico", "Nicaragua",
                         "Panama", "Venezuela"]
Tropical_Latin_America = ["Brazil", "Paraguay"]

# Super Region: North Africa and Middle East
North_Africa_and_Middle_East = ["Afghanistan", "Algeria", "Bahrain", "Egypt", "Iran", "Iraq", "Jordan", "Kuwait",
                                "Lebanon", "Libya", "Morocco", "Oman", "Qatar", "Saudi Arabia", "Syria",
                                "Tunisia", "Turkey", "United Arab Emirates", "Yemen"]

# Super Region: South Asia
South_Asia = ["Bangladesh", "Bhutan", "India", "Nepal", "Pakistan"]

# Super Region: Sub-Saharan Africa
Central_Sub_Saharan_Africa = ["Angola", "Central African Republic", "Congo", "Democratic Republic of the Congo",
                              "Equatorial Guinea", "Gabon"]
Eastern_Sub_Saharan_Africa = ["Burundi", "Comoros", "Djibouti", "Eritrea", "Ethiopia", "Kenya", "Madagascar", "Malawi",
                              "Mozambique", "Rwanda", "Somalia", "Tanzania", "Uganda", "Zambia"]
Southern_Sub_Saharan_Africa = ["Botswana", "Lesotho", "Namibia", "South Africa", "Swaziland", "Zimbabwe"]
Western_Sub_Saharan_Africa = ["Benin", "Burkina Faso", "Cape Verde", "Cameroon", "Chad", "Cote d'Ivoire", "The Gambia",
                              "Ghana", "Guinea", "Guinea-Bissau", "Liberia", "Mali", "Mauritania", "Niger", "Nigeria",
                              "Sao Tome and Principe", "Senegal", "Sierra Leone", "Togo"]

# Super Region: Southeast Asia, East Asia, and Oceania
East_Asia = ["China", "North Korea"]
Southeast_Asia = ["Cambodia", "Indonesia", "Laos", "Malaysia", "Maldives", "Mauritius", "Myanmar", "Philippines",
                  "Seychelles", "Sri Lanka", "Thailand", "Timor-Leste", "Vietnam"]

regions = [
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
    Southeast_Asia
]
if all(all(country in country_names for country in countries) for countries in regions):
    print("Check complete: all countries are in the official country list")
else:
    for countries in regions:
        for country in countries:
            if country not in country_names:
                print(f"Region {countries}, country {country} not in list!")
