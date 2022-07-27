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


def main():
    opts = Options()
    opts.headless = True

    # Set download path
    prefs = {"download.default_directory": default_path}
    opts.add_experimental_option("prefs", prefs)

    # Create browser object
    browser = Chrome(options=opts)
    browser.get("http://ihmeuw.org/5tc3")  # Preload link with advanced settings
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
    wait.until(
        EC.element_to_be_clickable((By.XPATH, '//*[@id="advanced-settings-button"]'))
    ).click()

    year = 2040
    age = "Age-standardized"
    for disease, disease_id in zip(diseases, disease_ids):

        # Cause dropdown
        browser.find_element_by_xpath('//*[@id="s2id_autogen3"]').send_keys(Keys.SPACE)
        cause_elem = browser.find_element_by_xpath(
            f'//*[@id="select2-results-3"]/li[{disease_id}]'
        )
        browser.execute_script("arguments[0].scrollIntoView();", cause_elem)
        cause_elem.click()
        browser.find_element_by_xpath('//*[@id="s2id_autogen3"]').send_keys(Keys.SPACE)

        # FIXME: Select Age-specific
        browser.find_element_by_xpath('//*[@id="s2id_autogen18"]').send_keys(Keys.SPACE)
        year_elem = browser.find_element_by_xpath(
            f'//*[@id="select2-results-18"]/li[9]'
        )
        browser.execute_script("arguments[0].scrollIntoView();", year_elem)
        year_elem.click()

        # TODO: Scroll to age

        # Click on download button
        dl_elem = browser.find_element_by_xpath('//*[@id="header-actions-download"]')
        dl_elem.click()

        # Click on "Chart data (csv)" button to download csv files
        csv_dl_elem = browser.find_element_by_xpath(
            '//*[@id="vizhub-action-download-content"]/span[2]'
        )
        csv_dl_elem.click()

        time.sleep(2)

        # Open download.csv and change name
        default_file = os.path.join(default_path, "download.csv")
        output_path = os.path.join(default_path, str(year), disease)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{age}.csv")
        shutil.move(default_file, output_file)

        print(f"Done: {disease}")

    browser.quit()


if __name__ == "__main__":
    main()
