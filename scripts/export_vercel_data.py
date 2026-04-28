import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.features.matchup_features import compute_matchup_features


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "web" / "public" / "data"


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def dataframe_records(df: pd.DataFrame) -> list[dict]:
    return [
        {column: clean_value(value) for column, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def write_json(name: str, payload) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {path.relative_to(ROOT)}")


def home_win_class_index(model) -> int:
    classes = list(getattr(model, "classes_", []))
    return classes.index(2) if 2 in classes else 2


def extract_home_win_shap_values(model, shap_values, sample_count: int) -> np.ndarray:
    class_index = home_win_class_index(model)

    if isinstance(shap_values, list):
        class_index = min(class_index, len(shap_values) - 1)
        return np.asarray(shap_values[class_index])[0]

    values = np.asarray(shap_values)
    if values.ndim == 3:
        if values.shape[0] == sample_count:
            class_index = min(class_index, values.shape[2] - 1)
            return values[0, :, class_index]
        class_index = min(class_index, values.shape[0] - 1)
        return values[class_index, 0, :]

    if values.ndim == 2:
        return values[0]

    raise ValueError(f"Unsupported SHAP output shape: {values.shape}")


def build_explanations(team_features: pd.DataFrame) -> dict:
    model = joblib.load(ROOT / "models" / "match_outcome_model.pkl")
    matches_df = pd.read_csv(ROOT / "data" / "raw" / "international_matches.csv")
    explainer = shap.TreeExplainer(model)
    feature_names = list(getattr(model, "feature_names_in_", []))
    explanations = {}

    for team_name in sorted(team_features["team_name"].tolist()):
        opponent_pool = team_features[team_features["team_name"] != team_name].copy()
        median_elo = opponent_pool["team_elo"].median()
        opponent_idx = (opponent_pool["team_elo"] - median_elo).abs().idxmin()
        opponent_name = opponent_pool.loc[opponent_idx, "team_name"]

        feat_dict = compute_matchup_features(team_name, opponent_name, team_features, matches_df)
        feat_vector = {k: v for k, v in feat_dict.items() if k not in {"home_team", "away_team"}}
        feat_vector["neutral_venue"] = 1
        feat_vector["home_is_host"] = 0
        feat_vector["away_is_host"] = 0

        df_X = pd.DataFrame([feat_vector]).reindex(columns=feature_names, fill_value=0)
        shap_values = extract_home_win_shap_values(model, explainer.shap_values(df_X), len(df_X))
        shap_items = [
            (feature, float(value))
            for feature, value in zip(df_X.columns, shap_values)
            if np.isfinite(value) and abs(value) > 1e-6
        ]
        shap_items = sorted(shap_items, key=lambda item: abs(item[1]), reverse=True)[:10]

        explanations[team_name] = {
            "team_name": team_name,
            "opponent_name": opponent_name,
            "positive_features": {k: v for k, v in shap_items if v > 0},
            "negative_features": {k: v for k, v in shap_items if v < 0},
        }

    return explanations


def build_feature_importance() -> list[dict]:
    model = joblib.load(ROOT / "models" / "match_outcome_model.pkl")
    features = list(getattr(model, "feature_names_in_", []))
    importances = list(getattr(model, "feature_importances_", []))
    records = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(features, importances)
    ]
    return sorted(records, key=lambda item: item["importance"], reverse=True)


def main() -> None:
    team_features = pd.read_csv(ROOT / "data" / "processed" / "team_features.csv")
    simulation_results = pd.read_csv(ROOT / "data" / "processed" / "simulation_results.csv")
    match_predictions = pd.read_csv(ROOT / "data" / "processed" / "match_predictions.csv")

    simulation_results = simulation_results.sort_values("win_probability", ascending=False).reset_index(drop=True)
    simulation_results.insert(0, "rank", simulation_results.index + 1)

    with (ROOT / "data" / "external" / "wc2026_groups.json").open("r", encoding="utf-8") as f:
        groups = json.load(f)

    write_json("simulation_results.json", dataframe_records(simulation_results))
    write_json("teams.json", dataframe_records(team_features))
    write_json("match_predictions.json", dataframe_records(match_predictions))
    write_json("groups.json", groups)
    write_json("feature_importance.json", build_feature_importance())
    write_json("explanations.json", build_explanations(team_features))
    write_json(
        "meta.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "teams": int(team_features["team_name"].nunique()),
            "matchups": int(len(match_predictions)),
            "simulations": 100000,
            "app": "vercel-static",
        },
    )


if __name__ == "__main__":
    main()
