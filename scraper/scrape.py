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
    browser.get("https://vizhub.healthdata.org/gbd-foresight/")
    assert "GBD Foresight" in browser.title

    wait = WebDriverWait(browser, 10)
    # Click on map on the left sidebar
    wait.until(
        EC.element_to_be_clickable(
            (By.XPATH, '//*[@id="control-option-buttonset-chart-map"]')
        )
    ).send_keys(Keys.SPACE)

    # TODO: "Use advanced settings"
    wait.until(
        EC.element_to_be_clickable((By.XPATH, '//*[@id="advanced-settings-button"]'))
    ).send_keys(Keys.SPACE)

    year = 2040
    age = "Age-standardized"
    for disease, disease_id in zip(diseases, disease_ids):

        # TODO: Cause dropdown
        browser.find_element_by_xpath('//*[@id="s2id_autogen3"]').send_keys(Keys.SPACE)
        cause_elem = browser.find_element_by_xpath(f'//*[@id="select2-results-3"]/li[{disease_id}]')
        browser.execute_script("arguments[0].scrollIntoView();", cause_elem)
        cause_elem.click()
        
        # TODO: Scroll to year

        # TODO: Age-specific

        time.sleep(5)

        # Click on download button
        dl_elem = browser.find_element_by_xpath('//*[@id="header-actions-download"]')
        dl_elem.click()

        # Click on "Chart data (csv)" button to download csv files
        csv_dl_elem = browser.find_element_by_xpath(
            '//*[@id="vizhub-action-download-content"]/span[2]'
        )
        csv_dl_elem.click()

        time.sleep(5)

        # TODO: Open download.csv and change name
        default_file = os.path.join(default_path, "download.csv")
        output_path = os.path.join(default_path, str(year), disease)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{age}.csv")
        shutil.move(default_file, output_file)

        print(f"Done: {disease}")

    browser.close()


if __name__ == "__main__":
    main()
