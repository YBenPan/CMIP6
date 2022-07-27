from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait, Select
import time
import shutil
import os

diseases = ["COPD", "IHD", "LRI", "LC", "Stroke", "T2D"]
# How to get disease id:
# Select dropdown item to inspect
# Copy XPath of <li> element
disease_ids = [126, 111, 12, 85, 112, 167]
default_path = "/Users/benpan/Documents/Mortality"

ages = [
    # "25-29",
    # "30-34",
    # "35-39",
    # "40-44",
    # "45-49",
    # "50-54",
    # "55-59",
    # "60-64",
    # "65-69",
    # "70-74",
    # "75-79",
    "80-84",
    "85-89",
    "90-94",
    "95+",
]

age_urls = [
    # "http://ihmeuw.org/5tc4",  # 25-29
    # "http://ihmeuw.org/5tc5",  # 30-34
    # "http://ihmeuw.org/5tc6",  # 35-39
    # "http://ihmeuw.org/5tc9",  # 40-44
    # "http://ihmeuw.org/5tca",  # 45-49
    # "http://ihmeuw.org/5tcb",  # 50-54
    # "http://ihmeuw.org/5tcc",  # 55-59
    # "http://ihmeuw.org/5t9m",  # 60-64
    # "http://ihmeuw.org/5tcd",  # 65-69
    # "http://ihmeuw.org/5tce",  # 70-74
    # "http://ihmeuw.org/5tcf",  # 75-79
    "http://ihmeuw.org/5tcg",  # 80-84
    "http://ihmeuw.org/5tch",  # 85-89
    "http://ihmeuw.org/5tc8",  # 90-94
    "http://ihmeuw.org/5tci",  # 95+
]


def main():
    opts = Options()
    opts.headless = True

    # Set download path
    prefs = {"download.default_directory": default_path}
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
            browser.find_element_by_xpath('//*[@id="s2id_autogen3"]').send_keys(
                Keys.SPACE
            )
            cause_elem = browser.find_element_by_xpath(
                f'//*[@id="select2-results-3"]/li[{disease_id}]'
            )
            browser.execute_script("arguments[0].scrollIntoView();", cause_elem)
            cause_elem.click()
            browser.find_element_by_xpath('//*[@id="s2id_autogen3"]').send_keys(
                Keys.SPACE
            )

            # FIXME: Select Age-specific
            # browser.find_element_by_xpath('//*[@id="s2id_autogen18"]').send_keys(Keys.SPACE)
            # year_elem = browser.find_element_by_xpath(
            #     f'//*[@id="select2-results-18"]/li[9]'
            # )
            # browser.execute_script("arguments[0].scrollIntoView();", year_elem)
            # year_elem.click()

            # TODO: Scroll to age

            # Click on download button
            dl_elem = browser.find_element_by_xpath(
                '//*[@id="header-actions-download"]'
            )
            dl_elem.click()

            # Click on "Chart data (csv)" button to download csv files
            csv_dl_elem = browser.find_element_by_xpath(
                '//*[@id="vizhub-action-download-content"]/span[2]'
            )
            csv_dl_elem.click()

            time.sleep(2)
            os.sync()

            # FIXME: Error that csv with 0s is copied to the output location

            # Open download.csv and change name
            default_file = os.path.join(default_path, "download.csv")
            output_path = os.path.join(default_path, str(year), disease)
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, f"{age}.csv")
            shutil.move(default_file, output_file)
            time.sleep(2)

            print(f"Done: {age}, {disease}")

    browser.quit()


if __name__ == "__main__":
    main()
