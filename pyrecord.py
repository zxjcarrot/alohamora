from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

d = DesiredCapabilities.CHROME.copy()

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.set_capability("acceptInsecureCerts", True)

d['goog:loggingPrefs'] = { 'browser':'ALL' }

driver = webdriver.Chrome(executable_path='/home/zxjcarrot/ungoogled-chromium_75.0.3770.80-1.1_linux/chromedriver',
                          chrome_options=chrome_options,
                          desired_capabilities=d,
                          service_args=["--verbose", "--log-path=chrome.log"])
driver.get('https://www.adobe.com')

p = driver.page_source


# print messages
for entry in driver.get_log('browser'):
    print(entry)

driver.quit()