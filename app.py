import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load ML model
with open("ipl_win_predictor.pkl", "rb") as f:
    pipe = joblib.load(f)

st.title("IPL Win Predictor")

teams = [
    'Royal Challengers Bangalore', 'Mumbai Indians', 'Kolkata Knight Riders',
    'Chennai Super Kings', 'Delhi Capitals', 'Kings XI Punjab',
    'Lucknow Super Giants', 'Rajasthan Royals'
]

cities = [
    'Bangalore', 'Chandigarh', 'Delhi', 'Mumbai', 'Jaipur', 'Chennai',
    'Kolkata', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
    'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
    'Ahmedabad', 'Dharamsala', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Cuttack', 'Visakhapatnam', 'Bengaluru', 'Indore',
    'Hyderabad', 'Dubai', 'Sharjah', 'Navi Mumbai', 'Lucknow',
    'Guwahati'
]

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select batting team", sorted(teams), key="batting")

with col2:
    bowling_team = st.selectbox("Select bowling team", sorted(teams), key="bowling")

selected_city = st.selectbox("Select city", sorted(cities), key="city")

target = st.number_input("Target", min_value=1, value=150, key="target")

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input("Current Score", min_value=0, value=50, key="score")

with col4:
    overs = st.number_input("Overs completed", min_value=0.0, max_value=20.0, value=5.0, key="overs")

with col5:
    wickets_out = st.number_input("Wickets fallen", min_value=0, max_value=10, value=2, key="wickets")

if st.button("Predict Probability"):

    if overs == 0:
        st.error("Overs must be greater than 0 to compute CRR.")
        st.stop()

    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets_remaining = 10 - wickets_out

    if balls_left <= 0:
        st.error("Invalid input: Balls left cannot be zero or negative.")
        st.stop()
    if wickets_remaining < 0:
        st.error("Invalid input: Wickets remaining is negative.")
        st.stop()

    crr = score / overs
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        "batting_team": [batting_team],
        "bowling_team": [bowling_team],
        "city": [selected_city],
        "runs_left": [runs_left],
        "balls_left": [balls_left],
        "wickets": [wickets_remaining],
        "total_runs_x": [target],
        "crr": [crr],
        "rrr": [rrr]
    })

    input_df = input_df.replace([np.inf, -np.inf], np.nan)
    input_df = input_df.fillna(0)

    try:
        result = pipe.predict_proba(input_df)[0]
        loss, win = result[0], result[1]

        st.subheader(f"🏏 {batting_team} Win Probability: **{round(win * 100)}%**")
        st.subheader(f"⚠️ {bowling_team} Win Probability: **{round(loss * 100)}%**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
