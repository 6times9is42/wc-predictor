import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.features.recency_weights import compute_recency_weight
from src.features.matchup_features import compute_h2h, compute_matchup_features

def test_compute_recency_weight():
    ref_date = pd.Timestamp("2026-06-01")
    
    # Today
    w1 = compute_recency_weight(ref_date, ref_date, 365)
    assert w1 == 1.0
    
    # Exactly halflife
    match_date = ref_date - timedelta(days=365)
    w2 = compute_recency_weight(match_date, ref_date, 365)
    assert w2 == 0.5
    
    # Future matches
    match_date = ref_date + timedelta(days=10)
    w3 = compute_recency_weight(match_date, ref_date, 365)
    assert w3 == 0.0

def test_compute_matchup_features_no_nans():
    # Mock data
    team_features = pd.DataFrame([{
        "team_name": "Brazil",
        "team_elo": 2000,
        "weighted_win_rate": 0.8,
        "weighted_goals_scored_per_game": 2.5,
        "weighted_goals_conceded_per_game": 0.5,
        "last_10_points": 24,
        "tournament_win_rate": 0.75,
        "squad_avg_rating": 8.5,
        "gk_rating": 8.0,
        "def_avg_rating": 8.2,
        "mid_avg_rating": 8.6,
        "fwd_avg_rating": 8.8,
        "squad_depth_score": 8.0,
        "market_value_top11_eur": 800000000,
        "injured_key_players": 0,
        "is_host_nation": 0,
        "confederation_name": "CONMEBOL",
        "wc_appearances_total": 22
    }, {
        "team_name": "France",
        "team_elo": 1980,
        "weighted_win_rate": 0.75,
        "weighted_goals_scored_per_game": 2.2,
        "weighted_goals_conceded_per_game": 0.6,
        "last_10_points": 22,
        "tournament_win_rate": 0.70,
        "squad_avg_rating": 8.4,
        "gk_rating": 8.1,
        "def_avg_rating": 8.0,
        "mid_avg_rating": 8.5,
        "fwd_avg_rating": 8.7,
        "squad_depth_score": 8.2,
        "market_value_top11_eur": 750000000,
        "injured_key_players": 1,
        "is_host_nation": 0,
        "confederation_name": "UEFA",
        "wc_appearances_total": 16
    }])
    
    matches_df = pd.DataFrame([{
        "date": "2022-12-18",
        "home_team": "Argentina",
        "away_team": "France",
        "home_goals": 3,
        "away_goals": 3,
        "outcome": "D"
    }])
    
    feat = compute_matchup_features("Brazil", "France", team_features, matches_df)
    
    # Check no NaNs
    for k, v in feat.items():
        if isinstance(v, float):
            assert not pd.isna(v)
            
    # Check deterministic
    feat2 = compute_matchup_features("Brazil", "France", team_features, matches_df)
    assert feat == feat2


def test_compute_h2h_ignores_unplayed_matches_and_smooths_small_samples():
    matches_df = pd.DataFrame([
        {
            "date": "2022-12-01",
            "home_team": "Japan",
            "away_team": "Spain",
            "home_goals": 2,
            "away_goals": 1,
        },
        {
            "date": "2026-06-15",
            "home_team": "Spain",
            "away_team": "Cape Verde",
            "home_goals": None,
            "away_goals": None,
        },
        {
            "date": "2026-06-25",
            "home_team": "Japan",
            "away_team": "Spain",
            "home_goals": None,
            "away_goals": None,
        },
    ])

    no_history = compute_h2h("Spain", "Cape Verde", matches_df)
    assert no_history == {"home_win_rate": 0.5, "avg_goal_diff": 0.0, "match_count": 0}

    one_match = compute_h2h("Spain", "Japan", matches_df)
    assert one_match["match_count"] == 1
    assert one_match["home_win_rate"] == pytest.approx(0.4)
    assert one_match["avg_goal_diff"] == pytest.approx(-0.2)


def test_compute_h2h_is_time_aware_for_training_rows():
    matches_df = pd.DataFrame([
        {
            "date": "2020-01-01",
            "home_team": "Team A",
            "away_team": "Team B",
            "home_goals": 2,
            "away_goals": 0,
        },
        {
            "date": "2024-01-01",
            "home_team": "Team B",
            "away_team": "Team A",
            "home_goals": 3,
            "away_goals": 0,
        },
    ])

    h2h = compute_h2h("Team A", "Team B", matches_df, as_of_date="2022-01-01")

    assert h2h["match_count"] == 1
    assert h2h["home_win_rate"] == pytest.approx(0.6)
    assert h2h["avg_goal_diff"] == pytest.approx(0.4)
