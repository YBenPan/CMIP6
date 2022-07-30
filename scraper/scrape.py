from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import time
import shutil
import os
import pandas as pd

diseases = ["Allcause", "COPD", "Dementia", "IHD", "LRI", "LC", "NCD", "Stroke", "T2D"]
# How to get disease id:
# Important: do not type anything in the search bar of causes
# Select dropdown item to inspect
# Copy XPath of <li> element
disease_ids = [1, 126, 152, 111, 12, 85, 69, 112, 167]
download_path = "D:\CMIP6_data\Mortality\Age-specific Mortality Projections_2040_dl"
mort_path = "D:\CMIP6_data\Mortality\Age-specific Mortality Projections_2040"

ages = [
    "25-29 years",
    "30-34 years",
    "35-39 years",
    "40-44 years",
    "45-49 years",
    "50-54 years",
    "55-59 years",
    "60-64 years",
    "65-69 years",
    "70-74 years",
    "75-79 years",
    "80-84 years",
    "85-89 years",
    "90-94 years",
    "95+ years",
]

age_urls = [
    "http://ihmeuw.org/5tc4",  # 25-29
    "http://ihmeuw.org/5tc5",  # 30-34
    "http://ihmeuw.org/5tc6",  # 35-39
    "http://ihmeuw.org/5tc9",  # 40-44
    "http://ihmeuw.org/5tca",  # 45-49
    "http://ihmeuw.org/5tcb",  # 50-54
    "http://ihmeuw.org/5tcc",  # 55-59
    "http://ihmeuw.org/5t9m",  # 60-64
    "http://ihmeuw.org/5tcd",  # 65-69
    "http://ihmeuw.org/5tce",  # 70-74
    "http://ihmeuw.org/5tcf",  # 75-79
    "http://ihmeuw.org/5tcg",  # 80-84
    "http://ihmeuw.org/5tch",  # 85-89
    "http://ihmeuw.org/5tc8",  # 90-94
    "http://ihmeuw.org/5tci",  # 95+
]


def download():
    opts = Options()
    opts.headless = True

    # Set download path
    prefs = {"download.default_directory": download_path}
    opts.add_experimental_option("prefs", prefs)

    # Create browser object
    browser = Chrome(options=opts)

    year = 2040
    for age, age_url in zip(ages, age_urls):

        print(age, age_url)

        browser.get(age_url)  # Preload link with advanced settings
        browser.maximize_window()
        assert "GBD Foresight" in browser.title

        wait = WebDriverWait(browser, 10)
        # Click on map on the left sidebar
        wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, '//*[@id="control-option-buttonset-chart-map"]')
            )
        ).send_keys(Keys.SPACE)

        # "Use advanced settings"
        # wait.until(
        #     EC.element_to_be_clickable(
        #         (By.XPATH, '//*[@id="advanced-settings-button"]')
        #     )
        # ).click()

        for disease, disease_id in zip(diseases, disease_ids):

            # Cause dropdown
            browser.find_element("xpath", '//*[@id="s2id_autogen3"]').send_keys(
                Keys.SPACE
            )
            cause_elem = browser.find_element(
                "xpath", f'//*[@id="select2-results-3"]/li[{disease_id}]'
            )
            browser.execute_script("arguments[0].scrollIntoView();", cause_elem)
            cause_elem.click()
            browser.find_element("xpath", '//*[@id="s2id_autogen3"]').send_keys(
                Keys.SPACE
            )

            # FIXME: Select Age-specific
            # browser.find_element("xpath", '//*[@id="s2id_autogen18"]').send_keys(Keys.SPACE)
            # year_elem = browser.find_element("xpath",
            #     f'//*[@id="select2-results-18"]/li[9]'
            # )
            # browser.execute_script("arguments[0].scrollIntoView();", year_elem)
            # year_elem.click()

            # TODO: Scroll to age

            time.sleep(2)

            # Click on download button
            dl_elem = browser.find_element(
                "xpath", '//*[@id="header-actions-download"]'
            )
            dl_elem.click()

            # Click on "Chart data (csv)" button to download csv files
            csv_dl_elem = browser.find_element(
                "xpath", '//*[@id="vizhub-action-download-content"]/span[2]'
            )
            csv_dl_elem.click()

            time.sleep(2)
            # os.sync()

            download_file = os.path.join(download_path, "download.csv")
            df = pd.read_csv(download_file)
            print(df)

            download_file = os.path.join(download_path, "download.csv")
            output_path = os.path.join(download_path, str(year), disease)
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, f"{age}.csv")
            shutil.move(download_file, output_file)

            print(f"Done: {age}, {disease}")

    browser.quit()


def rename_helper(df):
    """Drop, rename, and add countries"""
    to_be_dropped = [
        "American Samoa",
        "Bermuda",
        "Greenland",
        "Guam",
        "Montenegro",
        "Northern Mariana Islands",
        "Palestine",
        "South Sudan",
        "Taiwan (Province of China)",
        "Virgin Islands, U.S.",
    ]

    to_be_renamed = {
        "The Bahamas": "Bahamas",
        "North Korea": "Democratic Republic of North Korea",
        "Federated States of Micronesia": "Micronesia",
        "The Gambia": "Gambia",
        "Macedonia": "Thb.Macedonia",
        "Moldova": "Ra.Moldova",
        "South Korea": "Qb.South Korea",
        "Puerto Rico": "Zz.Puerto Rico",
        "Russian Federation": "Russia",
        "Tanzania": "United L.Tanzania",
        "Cote d'Ivoire": "Côte d'Ivoire",
    }

    to_be_added = [
        "Cook Islands",
        "Monaco",
        "Nauru",
        "Niue",
        "Palau",
        "Saint Kitts and Nevis",
        "San Marino",
        "Tuvalu",
    ]

    # Add countries
    sample_row = df.iloc[1].copy()
    sample_row.at["Value"] = 0
    sample_row.at["Lower bound"] = 0
    sample_row.at["Upper bound"] = 0

    for x in to_be_added:
        sample_row.at["Location"] = x
        df = pd.concat([df, sample_row.to_frame().T], ignore_index=True)

    # Drop countries
    locations = df["Location"].drop_duplicates().tolist()
    locations = [x for x in locations if x not in to_be_dropped]
    df = df.set_index("Location")
    df = df.loc[locations]

    # Rename countries
    labels = df.index.values
    new_labels = list(
        map(lambda x: to_be_renamed[x] if x in to_be_renamed else x, labels)
    )

    df = df.set_axis(new_labels, axis="index")
    df.index.name = "Location"
    df = df.reset_index()
    df = df.sort_values(by="Location")
    df = df.reset_index()
    df = df.drop(columns=["index"])
    # print(df)

    # Verify countries match with those of original projection files
    # country_df = pd.read_csv("D:\\CMIP6_data\\countryvalue_blank.csv")
    # country_locations = country_df["COUNTRY"].values
    # locations = df["Location"].values
    # for location, country_location in zip(locations, country_locations):
    #     if location != country_location:
    #         print(f"{location} in age-specific vs {country_location} in all-age")
    # input()

    return df


def post_process():
    """Clean the downloaded data for it to be used by mortality scripts"""
    year = 2040
    for age in ages:

        for disease in diseases:

            download_file = os.path.join(
                download_path, str(year), disease, f"{age}.csv"
            )
            df = pd.read_csv(download_file)

            # Remove last three rows of metadata
            df = df.iloc[:-3, :]

            # Rename countries
            df = rename_helper(df)

            # Output post-processed version to folder
            mort_output_path = os.path.join(mort_path, str(year), disease)
            os.makedirs(mort_output_path, exist_ok=True)
            mort_file = os.path.join(mort_output_path, f"{age}.csv")
            df.to_csv(mort_file)

            print(f"Done: {age}, {disease}")


def main():
    # download()
    post_process()


if __name__ == "__main__":
    main()
