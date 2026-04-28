import os
import pandas as pd
import numpy as np
from loguru import logger

H2H_PRIOR_MATCHES = 4
H2H_GOAL_DIFF_CAP = 3.0


def compute_h2h(home: str, away: str, matches_df: pd.DataFrame, as_of_date=None) -> dict:
    """
    Computes smoothed head-to-head features from completed matches only.
    """
    if matches_df is None or matches_df.empty:
        return {"home_win_rate": 0.5, "avg_goal_diff": 0.0, "match_count": 0}

    completed = matches_df.dropna(subset=["home_goals", "away_goals"]).copy()
    if as_of_date is not None and "date" in completed.columns:
        match_dates = pd.to_datetime(completed["date"], errors="coerce")
        completed = completed[match_dates < pd.Timestamp(as_of_date)]

    h2h_matches = completed[
        ((completed["home_team"] == home) & (completed["away_team"] == away)) |
        ((completed["home_team"] == away) & (completed["away_team"] == home))
    ]

    if h2h_matches.empty:
        return {"home_win_rate": 0.5, "avg_goal_diff": 0.0, "match_count": 0}

    home_wins = 0
    goal_diffs = []

    for _, match in h2h_matches.iterrows():
        is_home = match["home_team"] == home
        h_goals = match["home_goals"] if is_home else match["away_goals"]
        a_goals = match["away_goals"] if is_home else match["home_goals"]

        if h_goals > a_goals:
            home_wins += 1
        goal_diffs.append(h_goals - a_goals)

    match_count = len(h2h_matches)
    raw_goal_diff = float(np.mean(goal_diffs))
    capped_goal_diff = np.clip(raw_goal_diff, -H2H_GOAL_DIFF_CAP, H2H_GOAL_DIFF_CAP)
    shrinkage = match_count / (match_count + H2H_PRIOR_MATCHES)
    smoothed_win_rate = (home_wins + 0.5 * H2H_PRIOR_MATCHES) / (match_count + H2H_PRIOR_MATCHES)

    return {
        "home_win_rate": smoothed_win_rate,
        "avg_goal_diff": capped_goal_diff * shrinkage,
        "match_count": match_count,
    }

def compute_matchup_features(home_team: str, away_team: str, 
                             team_features_df: pd.DataFrame, matches_df: pd.DataFrame,
                             h2h_as_of_date=None) -> dict:
    """
    Creates input vector for match prediction model using differential features.
    """
    home_feat = team_features_df[team_features_df["team_name"] == home_team]
    away_feat = team_features_df[team_features_df["team_name"] == away_team]
    
    if home_feat.empty or away_feat.empty:
        raise ValueError(f"Feature vector not found for {home_team} or {away_team}")
        
    home_feat = home_feat.iloc[0]
    away_feat = away_feat.iloc[0]
    
    matchup = {
        # Identifiers
        "home_team": home_team,
        "away_team": away_team,
        
        # ELO differential
        "elo_diff": home_feat["team_elo"] - away_feat["team_elo"],
        "elo_home": home_feat["team_elo"],
        "elo_away": away_feat["team_elo"],
        
        # Form differentials
        "win_rate_diff": home_feat["weighted_win_rate"] - away_feat["weighted_win_rate"],
        "goals_scored_diff": home_feat["weighted_goals_scored_per_game"] - away_feat["weighted_goals_scored_per_game"],
        "goals_conceded_diff": home_feat["weighted_goals_conceded_per_game"] - away_feat["weighted_goals_conceded_per_game"],
        "last_10_points_diff": home_feat["last_10_points"] - away_feat["last_10_points"],
        "tournament_win_rate_diff": home_feat["tournament_win_rate"] - away_feat["tournament_win_rate"],
        
        # Squad quality differentials
        "squad_rating_diff": home_feat["squad_avg_rating"] - away_feat["squad_avg_rating"],
        "gk_rating_diff": home_feat["gk_rating"] - away_feat["gk_rating"],
        "def_rating_diff": home_feat["def_avg_rating"] - away_feat["def_avg_rating"],
        "mid_rating_diff": home_feat["mid_avg_rating"] - away_feat["mid_avg_rating"],
        "fwd_rating_diff": home_feat["fwd_avg_rating"] - away_feat["fwd_avg_rating"],
        "squad_depth_diff": home_feat["squad_depth_score"] - away_feat["squad_depth_score"],
        "market_value_diff_log": np.log(home_feat["market_value_top11_eur"] + 1) - np.log(away_feat["market_value_top11_eur"] + 1),
        "injured_diff": home_feat["injured_key_players"] - away_feat["injured_key_players"],
        
        # Head-to-head
        "neutral_venue": 1,  # WC matches are always neutral except openers (handled outside if needed)
        "home_is_host": home_feat["is_host_nation"],
        "away_is_host": away_feat["is_host_nation"],
        "same_confederation": int(home_feat.get("confederation_name", "") == away_feat.get("confederation_name", "")),
        "combined_wc_experience": home_feat["wc_appearances_total"] + away_feat["wc_appearances_total"],
    }
    
    if matches_df is not None:
        h2h = compute_h2h(home_team, away_team, matches_df, as_of_date=h2h_as_of_date)
        matchup["h2h_home_win_rate"] = h2h["home_win_rate"]
        matchup["h2h_recent_goal_diff"] = h2h["avg_goal_diff"]
        matchup["h2h_match_count"] = h2h["match_count"]
    else:
        matchup["h2h_home_win_rate"] = 0.5
        matchup["h2h_recent_goal_diff"] = 0.0
        matchup["h2h_match_count"] = 0
        
    return matchup

def build_all_matchup_features():
    logger.info("Building all possible matchup features...")
    try:
        team_features = pd.read_csv("data/processed/team_features.csv")
        matches_df = pd.read_csv("data/raw/international_matches.csv")
        
        teams = team_features["team_name"].tolist()
        matchups = []
        
        for i, home in enumerate(teams):
            for j, away in enumerate(teams):
                if i != j:
                    feat = compute_matchup_features(home, away, team_features, matches_df)
                    matchups.append(feat)
                    
        df = pd.DataFrame(matchups)
        out_path = "data/processed/matchup_features.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(df)} matchup features to {out_path}")
    except Exception as e:
        logger.error(f"Error building matchup features: {e}")

if __name__ == "__main__":
    build_all_matchup_features()
