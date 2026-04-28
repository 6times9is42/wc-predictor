import streamlit as st
import plotly.express as px
import pandas as pd

def render_probability_chart():
    st.header("Tournament Overview")
    try:
        df = pd.read_csv("data/processed/simulation_results.csv")
    except FileNotFoundError:
        st.error("Simulation results not found. Run pipeline first.")
        return

    df = df.sort_values("win_probability", ascending=False).reset_index(drop=True)
    df["win_prob_pct"] = df["win_probability"] * 100

    fig = px.bar(
        df,
        x="win_prob_pct",
        y="team",
        orientation="h",
        title="Teams by Win Probability",
        labels={"win_prob_pct": "Win Probability (%)", "team": "Team"},
        text_auto=".1f"
    )
    fig.update_traces(
        marker_color="#2563eb",
        hovertemplate="<b>%{y}</b><br>Win Probability: %{x:.2f}%<extra></extra>",
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=df["team"].tolist(),
        autorange="reversed",
    )
    fig.update_layout(
        height=max(650, 24 * len(df)),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Ranked Table")
    df["final_probability"] = df["win_probability"] + df["runner_up_probability"]

    display_cols = ["team", "win_probability", "final_probability", "semifinal_probability", "quarterfinal_probability", "r16_probability"]

    st.dataframe(df[display_cols].style.format({
        "win_probability": "{:.1%}",
        "final_probability": "{:.1%}",
        "semifinal_probability": "{:.1%}",
        "quarterfinal_probability": "{:.1%}",
        "r16_probability": "{:.1%}"
    }))
