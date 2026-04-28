import streamlit as st
import pandas as pd
import requests

API_URL = "http://localhost:8000"


def _feature_label(feature_name: str) -> str:
    return feature_name.replace("_", " ").title()


def _render_shap_list(features: dict, positive: bool) -> None:
    if not features:
        st.caption("No material positive SHAP drivers." if positive else "No material negative SHAP drivers.")
        return

    for feature, value in features.items():
        text = f"{_feature_label(feature)}: {value:+.3f}"
        if positive:
            st.success(text)
        else:
            st.error(text)

def render_team_deep_dive():
    st.header("Team Deep Dive")
    
    try:
        df = pd.read_csv("data/processed/team_features.csv")
        teams = df["team_name"].tolist()
        teams.sort()
    except FileNotFoundError:
        st.error("Team data not found.")
        return
        
    team = st.selectbox("Select Team", teams)
    
    if team:
        # Fetch team info from API
        try:
            res = requests.get(f"{API_URL}/teams/{team}")
            if res.status_code == 200:
                data = res.json()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("ELO Rating", round(data["team_elo"]))
                col2.metric("Squad Avg Rating", round(data["squad_avg_rating"], 2))
                col3.metric("Recent Win Rate", f"{data['weighted_win_rate']:.1%}")
                
                st.subheader("Why does this team have this probability?")
                
                # SHAP Explanation placeholder
                explain_res = requests.get(f"{API_URL}/predictions/explain/{team}")
                if explain_res.status_code == 200:
                    exp_data = explain_res.json()
                    opponent = exp_data.get("opponent_name", "an average team")
                    st.write(f"**Key SHAP drivers vs {opponent}:**")
                    _render_shap_list(exp_data.get("positive_features", {}), positive=True)
                    _render_shap_list(exp_data.get("negative_features", {}), positive=False)
                else:
                    detail = explain_res.json().get("detail", "Unable to load SHAP explanation.")
                    st.warning(detail)
            else:
                detail = res.json().get("detail", "Unable to load team details.")
                st.error(detail)
                        
        except Exception as e:
            st.error(f"API Error: {e}")
