from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import re
import time

options = Options()
options.add_argument('--headless')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

team_name = "United States"
driver.get(f"https://www.transfermarkt.us/schnellsuche/ergebnis/schnellsuche?query={team_name.replace(' ', '+')}")
time.sleep(3)

soup = BeautifulSoup(driver.page_source, 'html.parser')

# Find all links that have the title matching team_name
links = soup.find_all('a', title=re.compile(f'^{team_name}$', re.IGNORECASE))
tm_url = None
for link in links:
    href = link.get('href')
    if href and '/verein/' in href:
        tm_url = "https://www.transfermarkt.us" + href.replace('/startseite/', '/kader/')
        break

print("Found URL:", tm_url)
if tm_url:
    driver.get(tm_url)
    time.sleep(3)
    squad_soup = BeautifulSoup(driver.page_source, 'html.parser')
    table = squad_soup.select_one('table.items')
    if table:
        print("Found items table. Rows:", len(table.select('tbody tr.odd, tbody tr.even')))
driver.quit()
