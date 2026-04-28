import json
import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from fastapi import APIRouter, HTTPException
from src.api.schemas import MatchPredictionResponse, SimulationResult, GroupSimulationResponse, ExplainResponse
from src.features.matchup_features import compute_matchup_features
from src.model.predict import predict_match
from src.simulation.bracket import BracketSimulator
from src.simulation.monte_carlo import build_prediction_lookup

router = APIRouter()

MATCH_PREDICTIONS_PATH = Path("data/processed/match_predictions.csv")


def _load_prediction_lookup(path: Path = MATCH_PREDICTIONS_PATH) -> dict:
    if not path.exists():
        return build_prediction_lookup(save_path="")

    predictions = pd.read_csv(path)
    required_cols = {
        "home_team",
        "away_team",
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "expected_home_goals",
        "expected_away_goals",
    }
    missing = required_cols - set(predictions.columns)
    if missing:
        raise ValueError(f"match_predictions.csv is missing columns: {sorted(missing)}")

    lookup = {}
    for row in predictions.itertuples(index=False):
        lookup[(row.home_team, row.away_team)] = {
            "home_team": row.home_team,
            "away_team": row.away_team,
            "home_win_prob": float(row.home_win_prob),
            "draw_prob": float(row.draw_prob),
            "away_win_prob": float(row.away_win_prob),
            "expected_home_goals": float(row.expected_home_goals),
            "expected_away_goals": float(row.expected_away_goals),
            "confidence": getattr(row, "confidence", "precomputed"),
        }
    return lookup


def _sorted_probabilities(counts: dict, n_sims: int) -> dict:
    probs = {team: count / n_sims for team, count in counts.items()}
    return dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))


def _home_win_class_index(model) -> int:
    classes = list(getattr(model, "classes_", []))
    return classes.index(2) if 2 in classes else 2


def _extract_home_win_shap_values(model, df_X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_X)
    class_index = _home_win_class_index(model)

    if isinstance(shap_values, list):
        if class_index >= len(shap_values):
            class_index = len(shap_values) - 1
        return np.asarray(shap_values[class_index])[0]

    values = np.asarray(shap_values)
    if values.ndim == 3:
        # SHAP versions vary between (samples, features, classes) and
        # (classes, samples, features). Support both.
        if values.shape[0] == len(df_X):
            class_index = min(class_index, values.shape[2] - 1)
            return values[0, :, class_index]
        class_index = min(class_index, values.shape[0] - 1)
        return values[class_index, 0, :]

    if values.ndim == 2:
        return values[0]

    raise ValueError(f"Unsupported SHAP output shape: {values.shape}")


def _build_explanation_features(
    team_name: str,
    opponent_name: str,
    team_features: pd.DataFrame,
    matches_df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    feat_dict = compute_matchup_features(team_name, opponent_name, team_features, matches_df)
    feat_vector = {k: v for k, v in feat_dict.items() if k not in ["home_team", "away_team"]}
    feat_vector["neutral_venue"] = 1
    feat_vector["home_is_host"] = 0
    feat_vector["away_is_host"] = 0

    feature_names = list(getattr(model, "feature_names_in_", feat_vector.keys()))
    return pd.DataFrame([feat_vector]).reindex(columns=feature_names, fill_value=0)

@router.get("/match", response_model=MatchPredictionResponse)
def get_match_prediction(home: str, away: str):
    try:
        if home == away:
            raise HTTPException(status_code=400, detail="Choose two different teams for a matchup.")

        res = predict_match(home, away, neutral=True)
        if not res:
            raise HTTPException(status_code=500, detail="Prediction failed")
            
        # Add key factors
        home_win_prob = res["home_win_prob"]
        away_win_prob = res["away_win_prob"]
        
        factors = []
        if home_win_prob > away_win_prob + 0.1:
            factors.append(f"{home} is heavily favored based on current form and squad quality.")
        elif away_win_prob > home_win_prob + 0.1:
            factors.append(f"{away} is heavily favored based on current form and squad quality.")
        else:
            factors.append("This is an evenly matched game. Small details will decide the outcome.")
            
        res["key_factors"] = factors
        return res
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tournament")
def get_tournament_predictions():
    try:
        df = pd.read_csv("data/processed/simulation_results.csv")
        return df.to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Simulation results not found. Run simulation first.")

@router.get("/group/{group_letter}", response_model=GroupSimulationResponse)
def simulate_group(group_letter: str):
    group_name = f"Group {group_letter.upper()}"
    try:
        team_features = pd.read_csv("data/processed/team_features.csv")
    except FileNotFoundError:
        team_features = None
        
    try:
        prediction_lookup = _load_prediction_lookup()
    except Exception:
        prediction_lookup = {}

    sim = BracketSimulator(team_features, prediction_lookup=prediction_lookup)
    if group_name not in sim.groups:
        raise HTTPException(status_code=404, detail=f"Group {group_name} not found")
        
    n_sims = 5000 if prediction_lookup else 1000
    group_teams = sim.groups[group_name]
    advance_counts = {t: 0 for t in group_teams}
    top_two_counts = {t: 0 for t in group_teams}
    third_place_counts = {t: 0 for t in group_teams}
    points = {t: 0 for t in group_teams}
    rank_total = {t: 0 for t in group_teams}

    seed_offset = ord(group_letter.upper()[0]) - ord("A")
    np.random.seed(2026 + seed_offset)
    
    for _ in range(n_sims):
        standings = sim.simulate_group_stage()
        third_qualifiers = {record["team"] for record in sim.get_third_place_qualifier_records(standings)}
        group_res = standings[group_name]
        
        for rank, (team, stats) in enumerate(group_res, start=1):
            rank_total[team] += rank
            points[team] += stats["pts"]

            if rank <= 2:
                top_two_counts[team] += 1
                advance_counts[team] += 1
            elif rank == 3:
                third_place_counts[team] += 1
                if team in third_qualifiers:
                    advance_counts[team] += 1
            
    expected_points = {t: p / n_sims for t, p in points.items()}
    average_rank = {t: rank_total[t] / n_sims for t in group_teams}
    expected_standings = sorted(
        group_teams,
        key=lambda t: (average_rank[t], -expected_points[t], -top_two_counts[t]),
    )
    qualification_probs = _sorted_probabilities(advance_counts, n_sims)
    
    return {
        "group_name": group_name,
        "expected_standings": expected_standings,
        "qualification_probs": qualification_probs,
        "top_two_probs": _sorted_probabilities(top_two_counts, n_sims),
        "third_place_probs": _sorted_probabilities(third_place_counts, n_sims),
        "average_points": {
            team: round(expected_points[team], 2)
            for team in expected_standings
        },
    }

@router.get("/explain/{team_name}", response_model=ExplainResponse)
def explain_team_prediction(team_name: str):
    try:
        model = joblib.load("models/match_outcome_model.pkl")
        team_features = pd.read_csv("data/processed/team_features.csv")
        
        # We need a sample to explain. We'll use the team vs an average opponent
        if team_features.empty:
             raise HTTPException(status_code=500, detail="Data not found")
             
        team_row = team_features[team_features["team_name"] == team_name]
        if team_row.empty:
             raise HTTPException(status_code=404, detail="Team not found")
             
        # Calculate actual SHAP values against a median opponent
        opponent_pool = team_features[team_features["team_name"] != team_name].copy()
        median_elo = opponent_pool["team_elo"].median()
        opponent_idx = (opponent_pool["team_elo"] - median_elo).abs().idxmin()
        opponent_row = opponent_pool.loc[[opponent_idx]]
        opponent_name = opponent_row["team_name"].values[0]
        
        matches_df = pd.read_csv("data/raw/international_matches.csv")
        df_X = _build_explanation_features(team_name, opponent_name, team_features, matches_df, model)
        sv_home = _extract_home_win_shap_values(model, df_X)
        
        feature_names = df_X.columns
        shap_dict = dict(zip(feature_names, sv_home))
        
        sorted_shap = [
            (k, float(v))
            for k, v in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            if np.isfinite(v) and abs(v) > 1e-6
        ][:10]
        
        pos_features = {}
        neg_features = {}
        
        for k, v in sorted_shap:
            if v > 0:
                pos_features[k] = v
            elif v < 0:
                neg_features[k] = v
        
        return {
            "team_name": team_name,
            "opponent_name": opponent_name,
            "positive_features": pos_features,
            "negative_features": neg_features
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
