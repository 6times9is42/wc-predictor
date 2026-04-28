from pydantic import BaseModel
from typing import List, Optional

class TeamBase(BaseModel):
    team_name: str
    team_elo: float
    squad_avg_rating: float
    
class TeamDetail(TeamBase):
    weighted_win_rate: float
    squad_depth_score: float
    manager_tenure_days: int
    wc_appearances_total: int
    is_host_nation: int
    
class MatchPredictionResponse(BaseModel):
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    key_factors: List[str]
    confidence: str

class SimulationResult(BaseModel):
    rank: int
    team: str
    win_probability: float
    runner_up_probability: float
    semifinal_probability: float
    quarterfinal_probability: float
    r16_probability: float
    group_stage_exit_probability: float
    avg_goals_scored: float
    avg_simulated_points: float

class SimulateRequest(BaseModel):
    n_simulations: Optional[int] = 100000
    exclude_injured: Optional[bool] = False

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int

class GroupSimulationResponse(BaseModel):
    group_name: str
    expected_standings: List[str]
    qualification_probs: dict
    top_two_probs: Optional[dict] = None
    third_place_probs: Optional[dict] = None
    average_points: Optional[dict] = None

class ExplainResponse(BaseModel):
    team_name: str
    opponent_name: Optional[str] = None
    positive_features: dict
    negative_features: dict
