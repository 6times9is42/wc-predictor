import os
import yaml
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.metrics import log_loss
from loguru import logger
from src.features.matchup_features import compute_matchup_features

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def prepare_training_data(matches_df: pd.DataFrame, team_features_df: pd.DataFrame):
    """
    Prepares training data using ONLY historical data to prevent leakage.
    (Note: For simplicity in this implementation we use the snapshot features.
     In a true temporal setup, we would re-calculate features iteratively.)
    """
    logger.info("Preparing training data...")
    X = []
    y_outcome = []
    y_home_goals = []
    y_away_goals = []
    dates = []
    
    for _, match in matches_df.iterrows():
        try:
            home_goals = float(match["home_goals"])
            away_goals = float(match["away_goals"])
            if np.isnan(home_goals) or np.isnan(away_goals):
                continue
                
            match_date = pd.to_datetime(match["date"])
            feat = compute_matchup_features(
                match["home_team"],
                match["away_team"],
                team_features_df,
                matches_df,
                h2h_as_of_date=match_date,
            )
            # Remove identifiers
            feat_vector = {k: v for k, v in feat.items() if k not in ["home_team", "away_team"]}
            
            outcome = match["outcome"]
            if outcome == "A": y = 0
            elif outcome == "D": y = 1
            else: y = 2
                
            X.append(feat_vector)
            y_outcome.append(y)
            y_home_goals.append(home_goals)
            y_away_goals.append(away_goals)
            dates.append(match_date)
        except ValueError:
            continue
            
    df_X = pd.DataFrame(X)
    df_y = pd.Series(y_outcome)
    return df_X, df_y, pd.Series(y_home_goals), pd.Series(y_away_goals), pd.Series(dates)

def train_match_outcome_model():
    config = load_config()
    logger.info("Starting model training pipeline...")
    
    matches_df = pd.read_csv("data/raw/international_matches.csv")
    team_features_df = pd.read_csv("data/processed/team_features.csv")
    
    X, y, _, _, dates = prepare_training_data(matches_df, team_features_df)
    
    if X.empty:
        logger.error("No training data generated.")
        return
        
    # Temporal Split
    train_mask = dates < pd.Timestamp("2022-01-01")
    val_mask = dates >= pd.Timestamp("2022-01-01")
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    if X_train.empty or X_val.empty:
        logger.warning("Temporal split empty. Using standard train_test_split.")
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
    logger.info(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples.")

    # Optuna XGBoost Tuning
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": 42
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)
        return log_loss(y_val, preds)

    logger.info("Running Optuna optimization for XGBoost...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    n_trials = config["model"].get("optuna_trials", 100)
    # Reduce trials for testing speed
    study.optimize(objective, n_trials=min(n_trials, 20))
    
    logger.info(f"Best XGBoost Params: {study.best_params}")
    
    # Train final XGBoost
    best_xgb_params = study.best_params
    best_xgb_params["objective"] = "multi:softprob"
    best_xgb_params["num_class"] = 3
    best_xgb_params["random_state"] = 42
    
    xgb_model = xgb.XGBClassifier(**best_xgb_params)
    xgb_model.fit(X_train, y_train)
    
    xgb_val_loss = log_loss(y_val, xgb_model.predict_proba(X_val))
    logger.info(f"XGBoost Validation Log-Loss: {xgb_val_loss:.4f}")
    
    # Train LGBM Challenger
    logger.info("Training LightGBM Challenger Model...")
    lgbm_model = lgb.LGBMClassifier(objective="multiclass", num_class=3, random_state=42, verbose=-1)
    lgbm_model.fit(X_train, y_train)
    lgbm_val_loss = log_loss(y_val, lgbm_model.predict_proba(X_val))
    logger.info(f"LightGBM Validation Log-Loss: {lgbm_val_loss:.4f}")
    
    os.makedirs(config["paths"]["models"], exist_ok=True)
    
    xgb_path = os.path.join(config["paths"]["models"], "match_outcome_model_xgb.pkl")
    lgbm_path = os.path.join(config["paths"]["models"], "match_outcome_model_lgbm.pkl")
    final_path = os.path.join(config["paths"]["models"], "match_outcome_model.pkl")
    
    joblib.dump(xgb_model, xgb_path)
    joblib.dump(lgbm_model, lgbm_path)
    
    if xgb_val_loss <= lgbm_val_loss:
        logger.info("XGBoost selected as the primary model.")
        joblib.dump(xgb_model, final_path)
    else:
        logger.info("LightGBM selected as the primary model.")
        joblib.dump(lgbm_model, final_path)
        
    # Expected Goals Models
    logger.info("Training Expected Goals (xG) Models...")
    _, _, y_home, y_away, _ = prepare_training_data(matches_df, team_features_df)
    
    xg_home_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    xg_home_model.fit(X, y_home)
    
    xg_away_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    xg_away_model.fit(X, y_away)
    
    joblib.dump(xg_home_model, os.path.join(config["paths"]["models"], "xg_home_model.pkl"))
    joblib.dump(xg_away_model, os.path.join(config["paths"]["models"], "xg_away_model.pkl"))
    
    logger.info("Model training complete.")

if __name__ == "__main__":
    train_match_outcome_model()
