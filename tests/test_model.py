import pytest
import os
import joblib
import pandas as pd
import numpy as np


class FixedProbabilityModel:
    def predict_proba(self, X):
        probs = np.zeros((len(X), 3))
        probs[:, 0] = 0.2
        probs[:, 1] = 0.3
        probs[:, 2] = 0.5
        return probs

@pytest.fixture
def mock_model(tmp_path):
    model_path = tmp_path / "match_outcome_model.pkl"
    joblib.dump(FixedProbabilityModel(), model_path)
    return str(model_path)

def test_model_probabilities(mock_model):
    model = joblib.load(mock_model)
    X = np.random.rand(1, 5)
    probs = model.predict_proba(X)[0]
    
    assert len(probs) == 3
    assert np.all((probs >= 0) & (probs <= 1))
    assert np.isclose(sum(probs), 1.0)
    assert probs[2] == 0.5 # Home win prob
    
def test_predict_match_structure(monkeypatch, tmp_path):
    # Mock files to test logic
    import src.model.predict as pred

    pred.load_resources.cache_clear()
    pred.predict_match.cache_clear()
    
    # Mocking out the file reading
    monkeypatch.setattr("os.path.exists", lambda x: True)
    
    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.2, 0.3, 0.5]])
    
    monkeypatch.setattr("joblib.load", lambda x: MockModel())
    monkeypatch.setattr("pandas.read_csv", lambda x: pd.DataFrame([{
        "team_name": "TeamA", "team_elo": 1500, "wc_appearances_total": 5
    }, {
        "team_name": "TeamB", "team_elo": 1500, "wc_appearances_total": 2
    }]))
    
    monkeypatch.setattr(pred, "compute_matchup_features", lambda *args: {"feat1": 1.0})
    monkeypatch.setattr(pred, "predict_expected_goals", lambda *args: (1.5, 1.2))
    
    res = pred.predict_match("TeamA", "TeamB")
    
    assert "home_win_prob" in res
    assert "draw_prob" in res
    assert "away_win_prob" in res
    assert "expected_home_goals" in res
    assert "expected_away_goals" in res
    assert "confidence" in res

    pred.load_resources.cache_clear()
    pred.predict_match.cache_clear()
