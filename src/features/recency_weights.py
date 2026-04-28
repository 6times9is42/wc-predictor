import pandas as pd

def compute_recency_weight(match_date: pd.Timestamp,
                            reference_date: pd.Timestamp,
                            halflife_days: int) -> float:
    """
    Exponential decay weight. A match played `halflife_days` ago gets weight 0.5.
    A match played today gets weight 1.0.
    """
    if pd.isna(match_date) or pd.isna(reference_date):
        return 0.0
    
    # Ensure dates are compatible
    if isinstance(match_date, str):
        match_date = pd.to_datetime(match_date)
    if isinstance(reference_date, str):
        reference_date = pd.to_datetime(reference_date)
        
    days_ago = (reference_date - match_date).days
    if days_ago < 0:
        return 0.0 # Future matches have no weight
        
    return 0.5 ** (days_ago / halflife_days)
