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

driver.get('https://www.transfermarkt.com/frankreich/kader/verein/3377')
time.sleep(3)
soup = BeautifulSoup(driver.page_source, 'html.parser')
table = soup.select_one('table.items')
if table:
    rows = table.select('tbody tr.odd, tbody tr.even')
    for row in rows[:3]:
        inline_table = row.select_one('table.inline-table')
        if inline_table:
            tds = inline_table.select('td')
            player_name = tds[1].get_text(strip=True) if len(tds) > 1 else tds[0].get_text(strip=True)
            
            pos_str = row.select('td')[4].get_text(strip=True) if len(row.select('td')) > 4 else "Unknown"
            age_str = row.select('td')[5].get_text(strip=True) if len(row.select('td')) > 5 else "0"
            age_match = re.search(r'\((\d+)\)', age_str)
            age = int(age_match.group(1)) if age_match else 0
            mv_str = row.select('td.rechts.hauptlink')
            mv = mv_str[0].get_text(strip=True) if mv_str else "0"
            
            print(player_name, pos_str, age, mv)
driver.quit()
