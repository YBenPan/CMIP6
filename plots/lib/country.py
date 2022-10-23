import os
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from lib.regions import *


def get_country_names():
    """Return a list with all 193 countries"""
    countries_file = os.path.join(
        home_dir, "CMIP6_data", "population", "national_pop", "countryvalue_blank.csv"
    )
    countries_df = pd.read_csv(countries_file, usecols=["COUNTRY"])
    country_names = [*countries_df["COUNTRY"].values, "World"]
    # warnings.warn("Manually check if world is correct!")
    country_ids = [*countries_df.index.values, -1]
    country_dict = dict(zip(country_names, country_ids))
    return country_dict


def get_regions(region_source):
    """Return a list of regions with their country IDs and names"""
    country_dict = get_country_names()
    GBD_region_countries_dict = {
        "Southern Latin America": Southern_Latin_America,
        "Australasia": Australasia,
        "High-income Asia Pacific": High_income_Asia_Pacific,
        "High-income North America": High_income_North_America,
        "W. Europe": Western_Europe,
        "Central Europe": Central_Europe,
        "E. Europe": Eastern_Europe,
        "Central Asia": Central_Asia,
        "Caribbean": Caribbean,
        "Central Latin America": Central_Latin_America,
        "Tropical Latin America": Tropical_Latin_America,
        "Andean Latin America": Andean_Latin_America,
        "South Asia": South_Asia,
        "East Asia": East_Asia,
        "S.E. Asia": Southeast_Asia,
        "N. Africa and Middle East": North_Africa_and_Middle_East,
        "Central Africa": Central_Sub_Saharan_Africa,
        "E. Africa": Eastern_Sub_Saharan_Africa,
        "S. Africa": Southern_Sub_Saharan_Africa,
        "W. Africa": Western_Sub_Saharan_Africa,
        "World": ["World"],
    }
    SDI_region_countries_dict = {
        "High SDI": High_SDI,
        "High-middle SDI": High_middle_SDI,
        "Middle SDI": Middle_SDI,
        "Low-middle SDI": Low_middle_SDI,
        "Low SDI": Low_SDI,
        "World": ["World"],
    }
    GBD_super_region_countries_dict = {
        "High-income": High_income,
        "Central Europe, Eastern Europe, Central Asia": Central_Europe_Eastern_Europe_Central_Asia,
        "Southeast Asia, East Asia": Southeast_Asia_East_Asia,
        "Latin America and Caribbean": Latin_America_and_Caribbean,
        "South Asia": South_Asia,
        "North Africa and Middle East": North_Africa_and_Middle_East,
        "Sub-Saharan Africa": Sub_Saharan_Africa,
        "World": ["World"],
    }

    if region_source == "GBD":
        region_countries_dict = GBD_region_countries_dict
    elif region_source == "SDI":
        region_countries_dict = SDI_region_countries_dict
    elif region_source == "GBD_super":
        region_countries_dict = GBD_super_region_countries_dict
    else:
        raise Exception(f"Invalid region source {region_source}")

    region_countries_names = list(region_countries_dict.values())
    regions = list(region_countries_dict.keys())
    region_countries = [
        [country_dict[country_name] for country_name in countries_names]
        for countries_names in region_countries_names
    ]
    assert len(regions) == len(region_countries) == len(region_countries_names)
    return regions, region_countries, region_countries_names
