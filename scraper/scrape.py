from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
import time

def main():
    opts = Options()
    opts.headless = True

    # Set download path
    prefs = {"download.default_directory": "/Users/benpan/Documents/Mortality"}
    opts.add_experimental_option("prefs", prefs);

    # Create browser object
    browser = Chrome(options=opts)
    browser.get("https://vizhub.healthdata.org/gbd-foresight/")
    assert "GBD Foresight" in browser.title

    # Click on download button 
    map_elem = browser.find_element_by_xpath('//*[@id="header-actions-download"]')
    map_elem.click()
    time.sleep(5)

    # Click on "Chart data (csv)" button to download csv files  
    map_csv_download_elem = browser.find_element_by_xpath('//*[@id="vizhub-action-download-content"]/span[2]')
    map_csv_download_elem.click()
    time.sleep(5)

    browser.close()


if __name__ == "__main__":
    main()