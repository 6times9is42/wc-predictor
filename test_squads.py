import sys
sys.path.append('.')
from src.scraper.squad_data import get_all_squads
import pandas as pd

df = get_all_squads(["France", "United States"])
print(df.head())
print("Total rows:", len(df))
