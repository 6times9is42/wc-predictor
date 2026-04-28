import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

API_URL = "http://localhost:8000"

def render_h2h_matchup():
    st.header("Head-to-Head Matchup Tool")
    
    try:
        df = pd.read_csv("data/processed/team_features.csv")
        teams = df["team_name"].tolist()
        teams.sort()
    except FileNotFoundError:
        st.error("Team data not found.")
        return

    if len(teams) < 2:
        st.error("At least two teams are required for a matchup.")
        return
        
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A", teams, index=0, key="h2h_team_a")

    team_b_options = [team for team in teams if team != team_a]
    if st.session_state.get("h2h_team_b") == team_a:
        st.session_state["h2h_team_b"] = team_b_options[0]

    with col2:
        team_b = st.selectbox("Select Team B", team_b_options, index=0, key="h2h_team_b")
        
    if team_a and team_b:
        try:
            res = requests.get(
                f"{API_URL}/predictions/match",
                params={"home": team_a, "away": team_b},
            )
            if res.status_code == 200:
                data = res.json()
                
                # Donut Chart
                labels = [f"{team_a} Win", "Draw", f"{team_b} Win"]
                values = [data["home_win_prob"], data["draw_prob"], data["away_win_prob"]]
                
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
                fig.update_layout(title="Match Probability")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Key Factors")
                for factor in data["key_factors"]:
                    st.info(factor)
                    
                st.subheader("Expected Goals")
                st.metric(f"{team_a} xG", f"{data['expected_home_goals']:.2f}")
                st.metric(f"{team_b} xG", f"{data['expected_away_goals']:.2f}")
            else:
                detail = res.json().get("detail", "Unable to load matchup prediction.")
                st.error(detail)
                
        except Exception as e:
            st.error(f"API Error: {e}")

def render_group_simulator():
    st.header("Group Stage Simulator")
    
    group_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    group = st.selectbox("Select Group", group_letters)
    
    if st.button("Simulate Group"):
        try:
            res = requests.get(f"{API_URL}/predictions/group/{group}")
            if res.status_code == 200:
                data = res.json()
                
                st.subheader(f"Expected Standings for Group {group}")
                avg_points = data.get("average_points", {})
                for i, team in enumerate(data["expected_standings"]):
                    points_text = ""
                    if team in avg_points:
                        points_text = f" ({avg_points[team]:.2f} pts)"
                    st.write(f"{i+1}. {team}{points_text}")
                    
                st.subheader("Round of 32 Qualification Probabilities")
                probs = data["qualification_probs"]
                for team, prob in sorted(probs.items(), key=lambda item: item[1], reverse=True):
                    clipped_prob = min(max(prob, 0.0), 1.0)
                    st.progress(clipped_prob, text=f"{team}: {prob:.1%}")

                top_two_probs = data.get("top_two_probs", {})
                if top_two_probs:
                    with st.expander("Top-two finish probabilities"):
                        for team, prob in sorted(top_two_probs.items(), key=lambda item: item[1], reverse=True):
                            clipped_prob = min(max(prob, 0.0), 1.0)
                            st.progress(clipped_prob, text=f"{team}: {prob:.1%}")
            else:
                detail = res.json().get("detail", "Unable to simulate group.")
                st.error(detail)
        except Exception as e:
            st.error(f"API Error: {e}")
