import os
import joblib
import pandas as pd
import numpy as np
from loguru import logger
from src.features.matchup_features import compute_matchup_features

from functools import lru_cache

@lru_cache(maxsize=1)
def load_resources():
    calibrator_path = "models/calibrator.pkl"
    if not os.path.exists(calibrator_path):
        model_path = "models/match_outcome_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found. Please train first.")
        model = joblib.load(model_path)
    else:
        model = joblib.load(calibrator_path)
        
    try:
        team_features = pd.read_csv("data/processed/team_features.csv")
    except FileNotFoundError:
        logger.error("team_features.csv not found")
        team_features = pd.DataFrame()
        
    try:
        matches_df = pd.read_csv("data/raw/international_matches.csv")
    except FileNotFoundError:
        matches_df = None
        
    try:
        xg_home = joblib.load("models/xg_home_model.pkl")
        xg_away = joblib.load("models/xg_away_model.pkl")
    except FileNotFoundError:
        xg_home = None
        xg_away = None
        
    return model, team_features, matches_df, xg_home, xg_away

@lru_cache(maxsize=None)
def predict_match(home_team: str, away_team: str, neutral: bool = True) -> dict:
    """
    Predicts the outcome probabilities of a single match.
    """
    model, team_features, matches_df, _, _ = load_resources()
    
    if team_features.empty:
        return {}
        
    feat_dict = compute_matchup_features(home_team, away_team, team_features, matches_df)
    feat_vector = {k: v for k, v in feat_dict.items() if k not in ["home_team", "away_team"]}
    
    if neutral:
        feat_vector["home_is_host"] = 0
        feat_vector["away_is_host"] = 0
        feat_vector["neutral_venue"] = 1
        
    df_X = pd.DataFrame([feat_vector])
    
    probs = model.predict_proba(df_X)[0]
    
    home_xg, away_xg = predict_expected_goals(home_team, away_team, df_X)
    
    confidence = "medium"
    home_feat = team_features[team_features["team_name"] == home_team]
    if not home_feat.empty and home_feat.iloc[0]["wc_appearances_total"] > 10:
        confidence = "high"
        
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": float(probs[2]),
        "draw_prob": float(probs[1]),
        "away_win_prob": float(probs[0]),
        "expected_home_goals": float(home_xg),
        "expected_away_goals": float(away_xg),
        "confidence": confidence
    }

def predict_expected_goals(home_team: str, away_team: str, df_X: pd.DataFrame = None) -> tuple[float, float]:
    """
    Predicts expected goals for each team.
    """
    _, team_features, matches_df, home_model, away_model = load_resources()
    
    if home_model is None or away_model is None:
        return 1.1, 1.1 # Default fallback
        
    try:
        if df_X is None:
            feat_dict = compute_matchup_features(home_team, away_team, team_features, matches_df)
            feat_vector = {k: v for k, v in feat_dict.items() if k not in ["home_team", "away_team"]}
            df_X = pd.DataFrame([feat_vector])
            
        home_xg = max(0.0, float(home_model.predict(df_X)[0]))
        away_xg = max(0.0, float(away_model.predict(df_X)[0]))
        return home_xg, away_xg
    except Exception:
        return 1.1, 1.1 # Default fallback
        
if __name__ == "__main__":
    res = predict_match("Brazil", "France")
    print(res)
