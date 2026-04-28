import json
import pandas as pd
from fastapi import APIRouter, HTTPException
from src.api.schemas import TeamBase, TeamDetail

router = APIRouter()

def get_team_features():
    try:
        return pd.read_csv("data/processed/team_features.csv")
    except FileNotFoundError:
        return pd.DataFrame()

@router.get("", response_model=list[TeamBase])
def get_all_teams():
    df = get_team_features()
    if df.empty:
        # Fallback to qualified_teams.json
        try:
            with open("data/external/qualified_teams.json", "r") as f:
                teams = json.load(f)
            return [{"team_name": t, "team_elo": 1500.0, "squad_avg_rating": 5.0} for t in teams]
        except:
            raise HTTPException(status_code=500, detail="Data not available")
            
    teams = []
    for _, row in df.iterrows():
        teams.append({
            "team_name": row["team_name"],
            "team_elo": row.get("team_elo", 1500.0),
            "squad_avg_rating": row.get("squad_avg_rating", 5.0)
        })
    return teams

@router.get("/{team_name}", response_model=TeamDetail)
def get_team(team_name: str):
    df = get_team_features()
    if df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
        
    team = df[df["team_name"] == team_name]
    if team.empty:
        raise HTTPException(status_code=404, detail=f"Team {team_name} not found")
        
    row = team.iloc[0]
    return {
        "team_name": row["team_name"],
        "team_elo": row.get("team_elo", 1500.0),
        "squad_avg_rating": row.get("squad_avg_rating", 5.0),
        "weighted_win_rate": row.get("weighted_win_rate", 0.0),
        "squad_depth_score": row.get("squad_depth_score", 5.0),
        "manager_tenure_days": int(row.get("manager_tenure_days", 0)),
        "wc_appearances_total": int(row.get("wc_appearances_total", 0)),
        "is_host_nation": int(row.get("is_host_nation", 0))
    }
