import pandas as pd
import numpy as np

def aggregate_results(simulation_results: list, n_simulations: int) -> pd.DataFrame:
    """
    Aggregates the raw simulation logs into probability metrics per team.
    """
    teams = set()
    for res in simulation_results:
        teams.update(res["team_stats"].keys())
        
    agg = {team: {
        "wins": 0,
        "runner_ups": 0,
        "sfs": 0,
        "qfs": 0,
        "r16s": 0,
        "r32s": 0,
        "total_goals": 0,
        "total_points": 0,
        "paths": {}
    } for team in teams}
    
    for res in simulation_results:
        winner = res["final"]
        if winner in agg:
            agg[winner]["wins"] += 1
            
        runner_up = res["runner_up"]
        if runner_up in agg:
            agg[runner_up]["runner_ups"] += 1
            
        for t in res["sf"]:
            if t in agg: agg[t]["sfs"] += 1
        for t in res["qf"]:
            if t in agg: agg[t]["qfs"] += 1
        for t in res["r16"]:
            if t in agg: agg[t]["r16s"] += 1
        for t in res["r32"]:
            if t in agg: agg[t]["r32s"] += 1
            
        for t, stats in res["team_stats"].items():
            if t in agg:
                agg[t]["total_points"] += stats["pts"]
                agg[t]["total_goals"] += stats["gf"]
                
    # Build dataframe
    records = []
    for team, data in agg.items():
        records.append({
            "team": team,
            "win_probability": data["wins"] / n_simulations,
            "runner_up_probability": data["runner_ups"] / n_simulations,
            "semifinal_probability": data["sfs"] / n_simulations,
            "quarterfinal_probability": data["qfs"] / n_simulations,
            "r16_probability": data["r16s"] / n_simulations,
            "r32_probability": data["r32s"] / n_simulations,
            "group_stage_exit_probability": 1.0 - (data["r32s"] / n_simulations),
            "avg_goals_scored": data["total_goals"] / n_simulations,
            "avg_simulated_points": data["total_points"] / n_simulations,
            "most_common_path": "Group -> R32" # Simplification for now
        })
        
    df = pd.DataFrame(records)
    # Sort by win prob
    df = df.sort_values("win_probability", ascending=False).reset_index(drop=True)
    df.index += 1 # Rank 1-indexed
    df.index.name = "Rank"
    
    return df
