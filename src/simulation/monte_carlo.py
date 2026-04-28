import os
import yaml
import multiprocessing
import pandas as pd
from tqdm import tqdm
from loguru import logger
from src.model.predict import load_resources
from src.simulation.bracket import BracketSimulator
from src.simulation.results_aggregator import aggregate_results

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

# Global var for multiprocessing cache
_GLOBAL_TEAM_FEATURES = None
_GLOBAL_PREDICTION_LOOKUP = None

def build_prediction_lookup(save_path: str = "data/processed/match_predictions.csv") -> dict:
    logger.info("Precomputing match predictions for tournament teams...")
    model, _, _, xg_home, xg_away = load_resources()
    matchups = pd.read_csv("data/processed/matchup_features.csv")
    feature_df = matchups.drop(columns=["home_team", "away_team"])
    feature_df["neutral_venue"] = 1
    feature_df["home_is_host"] = 0
    feature_df["away_is_host"] = 0

    probabilities = model.predict_proba(feature_df)
    if xg_home is not None and xg_away is not None:
        home_xg = xg_home.predict(feature_df)
        away_xg = xg_away.predict(feature_df)
    else:
        home_xg = [1.1] * len(matchups)
        away_xg = [1.1] * len(matchups)

    prediction_rows = []
    prediction_lookup = {}
    for idx, row in matchups[["home_team", "away_team"]].iterrows():
        pred = {
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_win_prob": float(probabilities[idx][2]),
            "draw_prob": float(probabilities[idx][1]),
            "away_win_prob": float(probabilities[idx][0]),
            "expected_home_goals": max(0.0, float(home_xg[idx])),
            "expected_away_goals": max(0.0, float(away_xg[idx])),
            "confidence": "precomputed",
        }
        prediction_lookup[(row["home_team"], row["away_team"])] = pred
        prediction_rows.append(pred)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pd.DataFrame(prediction_rows).to_csv(save_path, index=False)
        logger.info(f"Saved precomputed match predictions to {save_path}")

    return prediction_lookup


def _init_worker(team_features: pd.DataFrame, prediction_lookup: dict):
    global _GLOBAL_TEAM_FEATURES, _GLOBAL_PREDICTION_LOOKUP
    _GLOBAL_TEAM_FEATURES = team_features
    _GLOBAL_PREDICTION_LOOKUP = prediction_lookup

def get_team_features():
    global _GLOBAL_TEAM_FEATURES
    if _GLOBAL_TEAM_FEATURES is None:
        try:
            _GLOBAL_TEAM_FEATURES = pd.read_csv("data/processed/team_features.csv")
        except FileNotFoundError:
            _GLOBAL_TEAM_FEATURES = pd.DataFrame()
    return _GLOBAL_TEAM_FEATURES

def get_prediction_lookup():
    global _GLOBAL_PREDICTION_LOOKUP
    if _GLOBAL_PREDICTION_LOOKUP is None:
        _GLOBAL_PREDICTION_LOOKUP = build_prediction_lookup(save_path="")
    return _GLOBAL_PREDICTION_LOOKUP

def run_single_simulation(seed: int) -> dict:
    import numpy as np
    np.random.seed(seed)
    
    team_features = get_team_features()
        
    sim = BracketSimulator(team_features, prediction_lookup=get_prediction_lookup())
    return sim.simulate_full_tournament()

def run_monte_carlo(n_simulations: int = None, seed: int = None) -> pd.DataFrame:
    config = load_config()
    
    if n_simulations is None:
        n_simulations = config["simulation"].get("n_simulations", 100000)
    if seed is None:
        seed = config["simulation"].get("random_seed", 2026)
        
    logger.info(f"Starting Monte Carlo simulation with {n_simulations} runs...")
    
    team_features = get_team_features()
    prediction_lookup = build_prediction_lookup()
    
    # We use a Pool
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Using {num_cores} cores.")
    
    seeds = [seed + i for i in range(n_simulations)]
    
    results = []
    # Adjust chunksize for better progress bar updates
    chunksize = max(1, n_simulations // (num_cores * 10))
    
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(num_cores, initializer=_init_worker, initargs=(team_features, prediction_lookup)) as pool:
        for res in tqdm(pool.imap_unordered(run_single_simulation, seeds, chunksize=chunksize), total=n_simulations):
            results.append(res)
            
    df = aggregate_results(results, n_simulations)
    
    out_path = "data/processed/simulation_results.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Simulation results saved to {out_path}")
    
    return df

if __name__ == "__main__":
    # Run full volume production simulation
    run_monte_carlo(n_simulations=100000)
