import os
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lib.regions import *


def get_country_names():
    """Return a list with all 193 countries"""
    countries_file = os.path.join(
        home_dir, "CMIP6_data", "population", "national_pop", "countryvalue_blank.csv"
    )
    countries_df = pd.read_csv(countries_file, usecols=["COUNTRY"])
    country_names = [*countries_df["COUNTRY"].values, "World"]
    country_ids = [*countries_df.index.values, -1]
    country_dict = dict(zip(country_names, country_ids))
    return country_dict


def get_regions():
    """Return a list of regions with their country IDs and names"""
    country_dict = get_country_names()
    region_countries_dict = {
        # Custom region settings
        "W. Europe": Western_Europe,
        "Central Europe": Central_Europe,
        "E. Europe": Eastern_Europe,
        "Canada, US": High_income_North_America,
        "Australia, New Zealand": Australasia,
        "Caribbean": Caribbean,
        "Central America": Central_Latin_America,
        "Argentina, Chile, Uruguay": Southern_Latin_America,
        "Brazil, Paraguay": Tropical_Latin_America,
        "Bolivia, Ecuador, Peru": Andean_Latin_America,
        "Central Asia": Central_Asia,
        "South Asia": South_Asia,
        "East Asia": East_Asia,
        "Brunei, Japan, Singapore, S. Korea": High_income_Asia_Pacific,
        "S.E. Asia": Southeast_Asia,
        "N. Africa and Middle East": North_Africa_and_Middle_East,
        "Central Africa": Central_Sub_Saharan_Africa,
        "E. Africa": Eastern_Sub_Saharan_Africa,
        "S. Africa": Southern_Sub_Saharan_Africa,
        "W. Africa": Western_Sub_Saharan_Africa,
        "World": ["World"],
    }
    region_countries_names = list(region_countries_dict.values())
    regions = list(region_countries_dict.keys())
    region_countries = [
        [country_dict[country_name] for country_name in countries_names]
        for countries_names in region_countries_names
    ]
    assert len(regions) == len(region_countries) == len(region_countries_names)
    return regions, region_countries, region_countries_names