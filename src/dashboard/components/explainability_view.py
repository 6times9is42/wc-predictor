import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import requests

API_URL = "http://localhost:8000"


def _feature_label(feature_name: str) -> str:
    return feature_name.replace("_", " ").title()


def _render_shap_features(features: dict, positive: bool) -> None:
    if not features:
        st.caption("No material positive SHAP drivers." if positive else "No material negative SHAP drivers.")
        return

    for feature, value in features.items():
        text = f"{_feature_label(feature)}: {value:+.3f}"
        if positive:
            st.success(text)
        else:
            st.error(text)

def render_explainability_view():
    st.header("Model Explainability")
    st.markdown("Understand what features are driving the predictions.")
    
    model_path = "models/match_outcome_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model not found. Run pipeline first.")
        return
        
    model = joblib.load(model_path)
    
    st.subheader("Global Feature Importance (XGBoost)")
    importances = model.feature_importances_
    features = model.feature_names_in_
    
    df_imp = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(15)
    
    # Make names readable
    df_imp["Feature"] = df_imp["Feature"].str.replace('_', ' ').str.title()
    
    fig = px.bar(
        df_imp, 
        x="Importance", 
        y="Feature", 
        orientation='h',
        title="Top 15 Most Important Features",
        color="Importance",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Team-Specific SHAP Analysis")
    try:
        df_features = pd.read_csv("data/processed/team_features.csv")
        teams = df_features["team_name"].tolist()
        teams.sort()
        
        team = st.selectbox("Select Team for Local Explanation", teams)
        
        if team:
            try:
                res = requests.get(f"{API_URL}/predictions/explain/{team}")
                if res.status_code == 200:
                    exp_data = res.json()
                    
                    opponent = exp_data.get("opponent_name", "an average team")
                    st.write(f"**Top drivers increasing {team}'s win probability vs {opponent}:**")
                    _render_shap_features(exp_data.get("positive_features", {}), positive=True)
                        
                    st.write(f"**Top factors decreasing {team}'s win probability vs {opponent}:**")
                    _render_shap_features(exp_data.get("negative_features", {}), positive=False)
                else:
                    detail = res.json().get("detail", "Unable to load SHAP explanation.")
                    st.warning(detail)
                        
            except Exception as e:
                st.error(f"API Error: {e}")
    except FileNotFoundError:
        st.error("Team features not found.")
