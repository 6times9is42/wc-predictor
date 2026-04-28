import streamlit as st
from src.dashboard.components.probability_chart import render_probability_chart
from src.dashboard.components.bracket_view import render_h2h_matchup, render_group_simulator
from src.dashboard.components.team_card import render_team_deep_dive
from src.dashboard.components.explainability_view import render_explainability_view

st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 FIFA 2026 World Cup Win Probability Predictor")

pages = {
    "Tournament Overview": render_probability_chart,
    "Head-to-Head Matchup": render_h2h_matchup,
    "Group Simulator": render_group_simulator,
    "Team Deep Dive": render_team_deep_dive,
    "Model Explainability": render_explainability_view
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

page = pages[selection]
page()
