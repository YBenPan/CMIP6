import pandas as pd
import os

home_dir = "/home/ybenp"
countries_file = os.path.join(home_dir, "CMIP6_data", "population", "national_pop", "countryvalue_blank.csv")
countries_df = pd.read_csv(countries_file, usecols=["COUNTRY"])
country_names = [*countries_df["COUNTRY"].values, "World"]

# Classifications from https://www.healthdata.org/sites/default/files/files/Projects/GBD/GBDRegions_countries.pdf

Central_Europe = ["Albania", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Czech Republic", "Hungary",
                  "Serbia+Montenegro", "Macedonia", "Poland", "Romania", "Slovakia", "Slovenia"]
Eastern_Europe = ["Belarus", "Estonia", "Latvia", "Lithuania", "Moldova", "Russia", "Ukraine"]
North_Africa_and_Middle_East = ["Afghanistan", "Algeria", "Bahrain", "Egypt", "Iran", "Iraq", "Jordan", "Kuwait",
                                "Lebanon", "Libya", "Morocco", "Oman", "Qatar", "Saudi Arabia", "Syria",
                                "Tunisia", "Turkey", "United Arab Emirates", "Yemen"]
South_Asia = ["Bangladesh", "Bhutan", "India", "Nepal", "Pakistan"]
High_income_North_America = ["Canada", "United States"]
Western_Sub_Saharan_Africa = ["Benin", "Burkina Faso", "Cape Verde", "Cameroon", "Chad", "Cote d'Ivoire", "The Gambia",
                              "Ghana", "Guinea", "Guinea-Bissau", "Liberia", "Mali", "Mauritania", "Niger", "Nigeria",
                              "Sao Tome and Principe", "Senegal", "Sierra Leone", "Togo"]

regions = [Central_Europe, Eastern_Europe, North_Africa_and_Middle_East, South_Asia, High_income_North_America,
           Western_Sub_Saharan_Africa]
if all(all(country in country_names for country in countries) for countries in regions):
    print("Check complete: all countries are in the official country list")
else:
    for countries in regions:
        for country in countries:
            if country not in country_names:
                print(f"Region {countries}, country {country} not in list!")
