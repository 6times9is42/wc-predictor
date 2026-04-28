import os
import yaml
import pandas as pd
from loguru import logger
from datetime import datetime
from src.features.recency_weights import compute_recency_weight

CONFEDERATION_BY_TEAM = {
    "Mexico": "CONCACAF", "South Africa": "CAF", "South Korea": "AFC", "Czechia": "UEFA",
    "Canada": "CONCACAF", "Switzerland": "UEFA", "Qatar": "AFC", "Bosnia and Herzegovina": "UEFA",
    "Brazil": "CONMEBOL", "Morocco": "CAF", "Haiti": "CONCACAF", "Scotland": "UEFA",
    "USA": "CONCACAF", "Paraguay": "CONMEBOL", "Australia": "AFC", "Türkiye": "UEFA",
    "Germany": "UEFA", "Curaçao": "CONCACAF", "Ivory Coast": "CAF", "Ecuador": "CONMEBOL",
    "Netherlands": "UEFA", "Japan": "AFC", "Tunisia": "CAF", "Sweden": "UEFA",
    "Belgium": "UEFA", "Egypt": "CAF", "Iran": "AFC", "New Zealand": "OFC",
    "Spain": "UEFA", "Cape Verde": "CAF", "Saudi Arabia": "AFC", "Uruguay": "CONMEBOL",
    "France": "UEFA", "Senegal": "CAF", "Norway": "UEFA", "Iraq": "AFC",
    "Argentina": "CONMEBOL", "Algeria": "CAF", "Austria": "UEFA", "Jordan": "AFC",
    "Portugal": "UEFA", "Jamaica": "CONCACAF", "Uzbekistan": "AFC", "Colombia": "CONMEBOL",
    "England": "UEFA", "Croatia": "UEFA", "Ghana": "CAF", "Panama": "CONCACAF",
}

WORLD_CUP_APPEARANCES = {
    "Mexico": 17, "South Africa": 3, "South Korea": 11, "Czechia": 9,
    "Canada": 2, "Switzerland": 12, "Qatar": 1, "Bosnia and Herzegovina": 1,
    "Brazil": 22, "Morocco": 6, "Haiti": 1, "Scotland": 8,
    "USA": 11, "Paraguay": 8, "Australia": 6, "Türkiye": 2,
    "Germany": 20, "Curaçao": 0, "Ivory Coast": 3, "Ecuador": 4,
    "Netherlands": 11, "Japan": 7, "Tunisia": 6, "Sweden": 12,
    "Belgium": 14, "Egypt": 3, "Iran": 6, "New Zealand": 2,
    "Spain": 16, "Cape Verde": 0, "Saudi Arabia": 6, "Uruguay": 14,
    "France": 16, "Senegal": 3, "Norway": 3, "Iraq": 1,
    "Argentina": 18, "Algeria": 4, "Austria": 7, "Jordan": 0,
    "Portugal": 8, "Jamaica": 1, "Uzbekistan": 0, "Colombia": 6,
    "England": 16, "Croatia": 6, "Ghana": 4, "Panama": 1,
}

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def _team_confederation(team: str, team_squad: pd.DataFrame) -> str:
    valid_confederations = {"UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC"}
    if "team_confederation" in team_squad.columns:
        values = team_squad["team_confederation"].dropna().astype(str)
        values = values[values.str.len() > 0]
        if not values.empty and values.iloc[0] in valid_confederations:
            return values.iloc[0]
    return CONFEDERATION_BY_TEAM.get(team, "UNKNOWN")

def _team_fifa_ranking(team_squad: pd.DataFrame) -> int:
    if "team_fifa_ranking" not in team_squad.columns:
        return 0
    values = pd.to_numeric(team_squad["team_fifa_ranking"], errors="coerce").dropna()
    values = values[values > 0]
    return int(values.iloc[0]) if not values.empty else 0

def compute_team_features(team: str, reference_date: pd.Timestamp, matches_df: pd.DataFrame,
                          squads_df: pd.DataFrame, ratings_df: pd.DataFrame) -> dict:
    """
    Computes a flat feature dictionary for a single team.
    """
    config = load_config()
    halflife = config["data"]["min_matches_for_team"] # this should be recency_halflife_days
    halflife = config["data"].get("recency_halflife_days", 365)
    
    # 1. Filter matches
    team_matches = matches_df[(matches_df["home_team"] == team) | (matches_df["away_team"] == team)].copy()
    team_matches["date"] = pd.to_datetime(team_matches["date"])
    team_matches = team_matches[team_matches["date"] < reference_date].sort_values("date")
    
    if len(team_matches) < config["data"].get("min_matches_for_team", 10):
        logger.warning(f"Team {team} has fewer than {config['data'].get('min_matches_for_team', 10)} matches before {reference_date}.")
    
    features = {"team_name": team, "reference_date": reference_date}
    
    # Base ELO (get last available ELO before reference date)
    if not team_matches.empty:
        last_match = team_matches.iloc[-1]
        features["team_elo"] = last_match["home_elo_before"] if last_match["home_team"] == team else last_match["away_elo_before"]
    else:
        features["team_elo"] = config["features"].get("elo_base", 1500)
        
    # Form & Results Features
    weighted_wins = 0.0
    weighted_draws = 0.0
    weighted_losses = 0.0
    weighted_goals_scored = 0.0
    weighted_goals_conceded = 0.0
    weighted_clean_sheets = 0.0
    total_weight = 0.0
    
    tournament_matches = 0
    tournament_wins = 0
    wc_matches = 0
    wc_wins = 0
    
    recent_points_10 = 0
    recent_points_5 = 0
    competitive_matches = []
    
    for _, match in team_matches.iterrows():
        weight = compute_recency_weight(match["date"], reference_date, halflife)
        
        is_home = match["home_team"] == team
        goals_for = match["home_goals"] if is_home else match["away_goals"]
        goals_against = match["away_goals"] if is_home else match["home_goals"]
        outcome_char = match["outcome"]
        
        if (is_home and outcome_char == "H") or (not is_home and outcome_char == "A"):
            result = "W"
            weighted_wins += weight
        elif outcome_char == "D":
            result = "D"
            weighted_draws += weight
        else:
            result = "L"
            weighted_losses += weight
            
        weighted_goals_scored += goals_for * weight
        weighted_goals_conceded += goals_against * weight
        if goals_against == 0:
            weighted_clean_sheets += weight
            
        total_weight += weight
        
        # Points
        pts = 3 if result == "W" else (1 if result == "D" else 0)
        if match["match_type"] != "Friendly":
            competitive_matches.append(pts)
            
        # Tournaments
        if match["match_type"] in ["WC", "EC", "CopaA", "AFCON", "Gold Cup", "Tournament"]:
            tournament_matches += 1
            if result == "W": tournament_wins += 1
            if match["match_type"] == "WC":
                wc_matches += 1
                if result == "W": wc_wins += 1
                
    if total_weight > 0:
        features["weighted_win_rate"] = weighted_wins / total_weight
        features["weighted_draw_rate"] = weighted_draws / total_weight
        features["weighted_loss_rate"] = weighted_losses / total_weight
        features["weighted_goals_scored_per_game"] = weighted_goals_scored / total_weight
        features["weighted_goals_conceded_per_game"] = weighted_goals_conceded / total_weight
        features["weighted_clean_sheet_rate"] = weighted_clean_sheets / total_weight
        features["goal_difference_per_game"] = features["weighted_goals_scored_per_game"] - features["weighted_goals_conceded_per_game"]
    else:
        features["weighted_win_rate"] = 0.0
        features["weighted_draw_rate"] = 0.0
        features["weighted_loss_rate"] = 0.0
        features["weighted_goals_scored_per_game"] = 0.0
        features["weighted_goals_conceded_per_game"] = 0.0
        features["weighted_clean_sheet_rate"] = 0.0
        features["goal_difference_per_game"] = 0.0
        
    features["last_5_points"] = sum(competitive_matches[-5:]) if competitive_matches else 0
    features["last_10_points"] = sum(competitive_matches[-10:]) if competitive_matches else 0
    features["tournament_win_rate"] = tournament_wins / tournament_matches if tournament_matches > 0 else 0.0
    features["wc_win_rate"] = wc_wins / wc_matches if wc_matches > 0 else 0.0
    
    # Squad Quality Features
    team_squad = squads_df[squads_df["team_name"] == team]
    team_ratings = ratings_df[ratings_df["team_name"] == team]
    
    if not team_squad.empty and not team_ratings.empty:
        rating_cols = [col for col in ["player_name", "team_name", "rating_composite"] if col in team_ratings.columns]
        merge_keys = ["player_name", "team_name"] if "team_name" in rating_cols else ["player_name"]
        ratings_for_merge = team_ratings[rating_cols].drop_duplicates(merge_keys)
        merged_squad = pd.merge(team_squad, ratings_for_merge, on=merge_keys, how="left")
        merged_squad["rating_composite"] = merged_squad["rating_composite"].fillna(5.0) # Impute avg rating
        
        # Sort by rating to get top players
        merged_squad = merged_squad.sort_values("rating_composite", ascending=False)
        top_23 = merged_squad.head(23)
        
        features["squad_avg_rating"] = top_23["rating_composite"].mean()
        
        # Top 11 by position
        gks = merged_squad[merged_squad["position"] == "GK"].head(1)
        defs = merged_squad[merged_squad["position"] == "DEF"].head(4)
        mids = merged_squad[merged_squad["position"] == "MID"].head(4)
        fwds = merged_squad[merged_squad["position"] == "FWD"].head(3) # Flexible
        top_11 = pd.concat([gks, defs, mids, fwds]).head(11)
        
        features["squad_avg_rating_starters"] = top_11["rating_composite"].mean() if not top_11.empty else features["squad_avg_rating"]
        features["gk_rating"] = gks["rating_composite"].mean() if not gks.empty else 5.0
        features["def_avg_rating"] = defs["rating_composite"].mean() if not defs.empty else 5.0
        features["mid_avg_rating"] = mids["rating_composite"].mean() if not mids.empty else 5.0
        features["fwd_avg_rating"] = fwds["rating_composite"].mean() if not fwds.empty else 5.0
        
        features["squad_depth_score"] = (features["squad_avg_rating"] - features["squad_avg_rating_starters"]) * -1 + 10
        features["squad_avg_age"] = top_23["age"].mean()
        
        peak_age_count = top_23[(top_23["age"] >= 24) & (top_23["age"] <= 29)].shape[0]
        features["squad_peak_age_ratio"] = peak_age_count / len(top_23) if len(top_23) > 0 else 0.0
        
        features["squad_total_caps"] = top_23["caps"].sum()
        features["squad_avg_caps"] = top_23["caps"].mean()
        
        top_leagues = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]
        top_league_count = top_23[top_23["club_league"].isin(top_leagues)].shape[0]
        features["squad_top_league_ratio"] = top_league_count / len(top_23) if len(top_23) > 0 else 0.0
        
        injured_starters = top_11[top_11["injured"] == True].shape[0]
        features["injured_key_players"] = injured_starters
        
        available = top_23[(top_23["injured"] == False) & (top_23["suspended"] == False)].shape[0]
        features["available_pct"] = available / len(top_23) if len(top_23) > 0 else 1.0
        
        features["market_value_total_eur"] = top_23["market_value_eur"].sum()
        features["market_value_top11_eur"] = top_11["market_value_eur"].sum() if not top_11.empty else 0.0
        features["fifa_ranking"] = _team_fifa_ranking(team_squad)
    else:
        # Fallback values
        features.update({
            "squad_avg_rating": 5.0, "squad_avg_rating_starters": 5.0,
            "gk_rating": 5.0, "def_avg_rating": 5.0, "mid_avg_rating": 5.0, "fwd_avg_rating": 5.0,
            "squad_depth_score": 5.0, "squad_avg_age": 27.0, "squad_peak_age_ratio": 0.5,
            "squad_total_caps": 300, "squad_avg_caps": 15.0, "squad_top_league_ratio": 0.1,
            "injured_key_players": 0, "available_pct": 1.0,
            "market_value_total_eur": 1e7, "market_value_top11_eur": 5e6,
            "fifa_ranking": _team_fifa_ranking(team_squad)
        })
        
    # Contextual Features
    # Manager tenure is not available on Transfermarkt squad pages; keep a neutral deterministic fallback.
    features["manager_tenure_days"] = 365
    
    confs = ["UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC"]
    conf = _team_confederation(team, team_squad)
    for c in confs:
        features[f"confederation_{c}"] = 1 if c == conf else 0
    features["confederation_name"] = conf # Store for later use
        
    features["is_host_nation"] = 1 if team in ["USA", "Canada", "Mexico"] else 0
    features["wc_appearances_total"] = WORLD_CUP_APPEARANCES.get(team, 0)
    features["wc_best_result_encoded"] = 0
    
    return features

def build_all_team_features():
    logger.info("Building team features for all qualified teams...")
    try:
        matches_df = pd.read_csv("data/raw/international_matches.csv")
        squads_df = pd.read_csv("data/raw/squads.csv")
        ratings_df = pd.read_csv("data/raw/player_ratings.csv")
        
        import json
        with open("data/external/qualified_teams.json", "r") as f:
            teams = json.load(f)
            
        reference_date = pd.Timestamp(datetime.now())
        
        all_features = []
        for team in teams:
            feat = compute_team_features(team, reference_date, matches_df, squads_df, ratings_df)
            all_features.append(feat)
            
        df = pd.DataFrame(all_features)
        
        # Drop temporary string columns if needed, but we might need confederation_name later
        
        out_path = "data/processed/team_features.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved team features to {out_path}")
    except Exception as e:
        logger.error(f"Error building team features: {e}")

if __name__ == "__main__":
    build_all_team_features()
