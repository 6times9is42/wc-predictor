from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import urllib.parse

options = Options()
options.add_argument('--headless')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

driver.get('https://html.duckduckgo.com/html/?q=France+national+football+team+squad+site:transfermarkt.com')
time.sleep(2)
links = driver.find_elements(By.CSS_SELECTOR, 'a.result__snippet')
for link in links:
    href = link.get_attribute('href')
    print("RAW HREF:", href)
    if href and 'transfermarkt' in href and ('/verein/' in href or '%2Fverein%2F' in href):
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        print("PARSED:", parsed)
driver.quit()
