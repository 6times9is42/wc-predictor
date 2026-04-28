import os

import numpy as np
import pandas as pd
import yaml
from loguru import logger


TOP_LEAGUES = {
    "Premier League",
    "LaLiga",
    "La Liga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
}

RATING_COLUMNS = [
    "player_name",
    "team_name",
    "league",
    "npxg_per90",
    "xa_per90",
    "progressive_passes_per90",
    "pressures_per90",
    "tackles_won_pct",
    "save_pct",
    "psxg_difference_per90",
    "sweeper_actions_per90",
    "minutes_played",
    "games_played",
    "market_value_eur",
    "age",
    "caps",
    "position",
    "club_name",
    "club_league_tier",
    "rating_source",
    "rating_composite",
]


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_value = values.min()
    max_value = values.max()
    if np.isclose(max_value, min_value):
        return pd.Series(0.5, index=series.index)
    return (values - min_value) / (max_value - min_value)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _age_score(age: float, position: str) -> float:
    if pd.isna(age) or age <= 0:
        return 0.5

    if position == "GK":
        prime_low, prime_high = 26, 34
    else:
        prime_low, prime_high = 24, 29

    if prime_low <= age <= prime_high:
        return 1.0
    if age < prime_low:
        return max(0.35, 1.0 - (prime_low - age) / 8.0)
    return max(0.30, 1.0 - (age - prime_high) / 8.0)


def _league_score(league: str, tier: int) -> float:
    if league in TOP_LEAGUES:
        return 1.0
    if tier == 1:
        return 0.82
    if tier == 2:
        return 0.55
    if tier == 3:
        return 0.35
    if tier >= 4:
        return 0.20
    return 0.45


def _column_or_default(df: pd.DataFrame, column: str, default):
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def build_transfermarkt_ratings(squads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds deterministic player quality scores from Transfermarkt fields.

    Transfermarkt does not publish an all-in player rating, so this creates the
    pipeline's rating_composite from scraped market value, age, caps, club league
    tier, and availability. No random generation is used.
    """
    if squads_df.empty:
        return pd.DataFrame(columns=RATING_COLUMNS)

    df = squads_df.copy()
    df["market_value_eur"] = pd.to_numeric(df.get("market_value_eur", 0), errors="coerce").fillna(0.0)
    df["age"] = pd.to_numeric(df.get("age", 0), errors="coerce").fillna(0.0)
    df["caps"] = pd.to_numeric(df.get("caps", 0), errors="coerce").fillna(0.0)
    df["club_league_tier"] = pd.to_numeric(df.get("club_league_tier", 0), errors="coerce").fillna(0).astype(int)
    df["club_league"] = _column_or_default(df, "club_league", "Unknown").fillna("Unknown")
    df["position"] = _column_or_default(df, "position", "MID").fillna("MID")

    market_score = normalize(np.log1p(df["market_value_eur"]))
    caps_score = normalize(np.log1p(df["caps"]))
    age_score = pd.Series(
        [_age_score(age, position) for age, position in zip(df["age"], df["position"])],
        index=df.index,
    )
    league_score = pd.Series(
        [_league_score(league, tier) for league, tier in zip(df["club_league"], df["club_league_tier"])],
        index=df.index,
    )

    rating = (
        0.65 * market_score
        + 0.15 * league_score
        + 0.10 * age_score
        + 0.10 * caps_score
    ) * 10

    unavailable = (
        _column_or_default(df, "injured", False).map(_as_bool)
        | _column_or_default(df, "suspended", False).map(_as_bool)
    )
    rating = rating.where(~unavailable, rating * 0.85)
    rating = rating.clip(lower=0.5, upper=10.0)

    ratings = pd.DataFrame(
        {
            "player_name": df["player_name"],
            "team_name": df["team_name"],
            "league": df["club_league"],
            "npxg_per90": 0.0,
            "xa_per90": 0.0,
            "progressive_passes_per90": 0.0,
            "pressures_per90": 0.0,
            "tackles_won_pct": 0.0,
            "save_pct": 0.0,
            "psxg_difference_per90": 0.0,
            "sweeper_actions_per90": 0.0,
            "minutes_played": 0,
            "games_played": 0,
            "market_value_eur": df["market_value_eur"].astype(int),
            "age": df["age"].astype(int),
            "caps": df["caps"].astype(int),
            "position": df["position"],
            "club_name": _column_or_default(df, "club_name", "Unknown Club"),
            "club_league_tier": df["club_league_tier"],
            "rating_source": "transfermarkt_market_value",
            "rating_composite": rating.round(4),
        }
    )

    return ratings[RATING_COLUMNS]


def get_player_ratings(season: str = "2025-2026") -> pd.DataFrame:
    """
    Builds player ratings from scraped Transfermarkt squad data.
    """
    logger.info(f"Building Transfermarkt-derived player ratings for season {season}")

    squads_path = "data/raw/squads.csv"
    if not os.path.exists(squads_path):
        logger.error(f"Squads data not found at {squads_path}. Run squad_data.py first.")
        return pd.DataFrame(columns=RATING_COLUMNS)

    squads_df = pd.read_csv(squads_path)
    ratings_df = build_transfermarkt_ratings(squads_df)

    config = load_config()
    out_dir = config["paths"]["raw_data"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "player_ratings.csv")
    ratings_df.to_csv(out_path, index=False)
    logger.info(f"Saved player ratings to {out_path}")

    return ratings_df


if __name__ == "__main__":
    get_player_ratings()
