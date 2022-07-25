from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
opts = Options()
opts.headless = True
browser = Chrome(options=opts)
browser.get("https://google.com")
