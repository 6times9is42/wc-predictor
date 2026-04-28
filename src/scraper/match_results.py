import os
import yaml
import requests
import pandas as pd
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def fetch_football_data_org(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetches match results from football-data.org API.
    Note: Free tier has limited access to historical data and specific competitions.
    """
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    if not api_key or api_key == "your_key_here":
        logger.warning("No valid FOOTBALL_DATA_API_KEY found. Skipping football-data.org.")
        return pd.DataFrame()

    headers = {"X-Auth-Token": api_key}
    # Competition 2000 is World Cup, 2018 is Euro, etc.
    url = f"https://api.football-data.org/v4/competitions/2000/matches?dateFrom={start_year}-01-01&dateTo={end_year}-12-31"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        matches = data.get("matches", [])
        
        processed = []
        for match in matches:
            if match["status"] != "FINISHED":
                continue
            
            home_team = match["homeTeam"]["name"]
            away_team = match["awayTeam"]["name"]
            home_goals = match["score"]["fullTime"]["home"]
            away_goals = match["score"]["fullTime"]["away"]
            
            if home_goals > away_goals:
                outcome = "H"
            elif home_goals < away_goals:
                outcome = "A"
            else:
                outcome = "D"
                
            processed.append({
                "date": pd.to_datetime(match["utcDate"]).date(),
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "outcome": outcome,
                "match_type": "WC",
                "tournament_name": "FIFA World Cup",
                "neutral_venue": True
            })
        return pd.DataFrame(processed)
    except Exception as e:
        logger.error(f"Error fetching from football-data.org: {e}")
        return pd.DataFrame()


def get_public_international_matches(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetches a well-maintained public dataset of international football results
    (Mart Jürisoo's dataset hosted on GitHub) as the primary reliable source.
    """
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    logger.info(f"Downloading historical match data from {url}")
    try:
        df = pd.read_csv(url)
        df["date"] = pd.to_datetime(df["date"])
        
        # Filter by year
        df = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)].copy()
        
        # Rename columns to match requirements
        df = df.rename(columns={
            "home_score": "home_goals",
            "away_score": "away_goals",
            "tournament": "tournament_name",
            "neutral": "neutral_venue"
        })
        
        # Standardize team names to match our qualified_teams.json
        name_mapping = {
            "United States": "USA",
            "Turkey": "Türkiye",
            "Czech Republic": "Czechia"
        }
        df["home_team"] = df["home_team"].replace(name_mapping)
        df["away_team"] = df["away_team"].replace(name_mapping)
        
        # Calculate outcome
        def get_outcome(row):
            if row["home_goals"] > row["away_goals"]:
                return "H"
            elif row["home_goals"] < row["away_goals"]:
                return "A"
            return "D"
            
        df["outcome"] = df.apply(get_outcome, axis=1)
        
        # Map match types
        def map_match_type(t):
            t = str(t).lower()
            if "friendly" in t: return "Friendly"
            if "world cup qualification" in t: return "WCQ"
            if "world cup" in t: return "WC"
            if "euro" in t and "qualification" in t: return "UCQ"
            if "euro" in t: return "EC"
            if "copa américa" in t: return "CopaA"
            if "african cup of nations" in t: return "AFCON"
            if "gold cup" in t: return "Gold Cup"
            if "nations league" in t: return "Nations League"
            return "Tournament"
            
        df["match_type"] = df["tournament_name"].apply(map_match_type)
        
        return df
    except Exception as e:
        logger.error(f"Failed to fetch public dataset: {e}")
        return pd.DataFrame()

def calculate_dynamic_elo(df: pd.DataFrame, base_elo: int, k_factor: int, config: dict) -> pd.DataFrame:
    """
    Calculates ELO ratings dynamically chronologically.
    """
    logger.info("Calculating dynamic ELO ratings...")
    elo_dict = {}
    
    home_elos = []
    away_elos = []
    
    # Sort chronologically just in case
    df = df.sort_values("date").reset_index(drop=True)
    
    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        
        # Initialize if not exists
        if home not in elo_dict: elo_dict[home] = base_elo
        if away not in elo_dict: elo_dict[away] = base_elo
        
        # Record before-match ELO
        home_elo_before = elo_dict[home]
        away_elo_before = elo_dict[away]
        
        home_elos.append(home_elo_before)
        away_elos.append(away_elo_before)
        
        # Expected outcomes
        dr = home_elo_before - away_elo_before + 100 # Add 100 points for home advantage if not neutral
        if row["neutral_venue"]:
            dr = home_elo_before - away_elo_before
            
        e_home = 1 / (10 ** (-dr / 400) + 1)
        e_away = 1 - e_home
        
        # Actual outcomes
        if row["outcome"] == "H":
            s_home, s_away = 1, 0
        elif row["outcome"] == "A":
            s_home, s_away = 0, 1
        else:
            s_home, s_away = 0.5, 0.5
            
        # Match weight
        m_type = row["match_type"]
        weight = 1.0
        if m_type == "Friendly":
            weight = config["data"].get("friendly_weight", 0.4)
        elif m_type in ["WCQ", "UCQ"]:
            weight = config["data"].get("qualifying_weight", 0.8)
        elif m_type in ["WC", "EC", "CopaA", "AFCON"]:
            weight = config["data"].get("tournament_weight", 1.0)
            
        # Update ELO
        # Goal difference multiplier
        gd = abs(row["home_goals"] - row["away_goals"])
        g_mult = 1.0
        if gd == 2: g_mult = 1.5
        elif gd == 3: g_mult = 1.75
        elif gd > 3: g_mult = 1.75 + (gd - 3) / 8.0
            
        elo_change = k_factor * weight * g_mult * (s_home - e_home)
        
        elo_dict[home] += elo_change
        elo_dict[away] -= elo_change
        
    df["home_elo_before"] = home_elos
    df["away_elo_before"] = away_elos
    
    return df

def scrape_international_matches(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Main entry point to scrape and process matches.
    """
    logger.info(f"Starting match results scraping from {start_year} to {end_year}")
    config = load_config()
    
    # 1. Try public github dataset first for comprehensive data
    df_public = get_public_international_matches(start_year, end_year)
    
    # 2. Try API (optional)
    df_api = fetch_football_data_org(start_year, end_year)
    
    # Merge and deduplicate
    if not df_public.empty and not df_api.empty:
        df = pd.concat([df_public, df_api]).drop_duplicates(subset=["date", "home_team", "away_team"])
    elif not df_public.empty:
        df = df_public
    else:
        df = df_api
        
    if df.empty:
        logger.error("No match data could be retrieved from any source.")
        return pd.DataFrame()
        
    # Calculate ELO
    base_elo = config["features"].get("elo_base", 1500)
    k_factor = config["features"].get("elo_k_factor", 32)
    df = calculate_dynamic_elo(df, base_elo, k_factor, config)
    
    # Keep required columns
    required_cols = ["date", "home_team", "away_team", "home_goals", "away_goals", "outcome", 
                     "match_type", "tournament_name", "neutral_venue", "home_elo_before", "away_elo_before"]
    
    df = df[required_cols]
    
    # Save
    out_dir = config["paths"]["raw_data"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "international_matches.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Successfully saved {len(df)} matches to {out_path}")
    
    return df

if __name__ == "__main__":
    config = load_config()
    current_year = datetime.now().year
    start_year = current_year - config["data"]["match_history_years"]
    scrape_international_matches(start_year, current_year)
