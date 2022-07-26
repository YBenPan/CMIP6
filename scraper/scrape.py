from xml.dom.minidom import Element
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import time


def main():
    opts = Options()
    opts.headless = True

    # Set download path
    prefs = {"download.default_directory": "/Users/benpan/Documents/Mortality"}
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

    # TODO: Cause dropdown

    # TODO: Scroll to year

    # TODO: Age-specific

    # Click on download button
    dl_elem = browser.find_element_by_xpath('//*[@id="header-actions-download"]')
    dl_elem.click()
    time.sleep(5)

    # TODO: Set output path and file

    # Click on "Chart data (csv)" button to download csv files
    csv_dl_elem = browser.find_element_by_xpath(
        '//*[@id="vizhub-action-download-content"]/span[2]'
    )
    csv_dl_elem.click()
    time.sleep(5)

    browser.close()


if __name__ == "__main__":
    main()
