import os
import joblib
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from loguru import logger

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Computes ECE"""
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            prob_in_bin = np.mean(mask)
            ece += np.abs(acc - conf) * prob_in_bin
            
    return ece

def calibrate_model():
    logger.info("Calibrating model probabilities...")
    
    model_path = "models/match_outcome_model.pkl"
    if not os.path.exists(model_path):
        logger.error("Base model not found. Run train.py first.")
        return
        
    base_model = joblib.load(model_path)
    
    # In a real setup, we should use a held-out calibration set.
    # We will reuse the validation setup logic from train for demonstration.
    from src.model.train import prepare_training_data
    matches_df = pd.read_csv("data/raw/international_matches.csv")
    team_features_df = pd.read_csv("data/processed/team_features.csv")
    
    X, y, _, _, dates = prepare_training_data(matches_df, team_features_df)
    
    val_mask = dates >= pd.Timestamp("2022-01-01")
    X_cal, y_cal = X[val_mask], y[val_mask]
    
    if X_cal.empty:
        X_cal, y_cal = X, y # Fallback
        
    # Multi-class ECE before
    preds = base_model.predict_proba(X_cal)
    ece_before = sum([expected_calibration_error((y_cal == c).astype(int), preds[:, c]) for c in range(3)]) / 3
    
    logger.info(f"Pre-calibration ECE: {ece_before:.4f}")
    
    # Isotonic Calibration
    calibrated = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=2)
    calibrated.fit(X_cal, y_cal)
    
    preds_cal = calibrated.predict_proba(X_cal)
    ece_after = sum([expected_calibration_error((y_cal == c).astype(int), preds_cal[:, c]) for c in range(3)]) / 3
    
    logger.info(f"Post-calibration ECE: {ece_after:.4f}")
    
    if ece_after <= ece_before:
        joblib.dump(calibrated, "models/calibrator.pkl")
        logger.info("Calibrator saved to models/calibrator.pkl")
    else:
        joblib.dump(base_model, "models/calibrator.pkl")
        logger.warning(
            "Calibration worsened ECE; saved the uncalibrated model to models/calibrator.pkl "
            "so prediction uses the better probability estimates."
        )

if __name__ == "__main__":
    calibrate_model()
