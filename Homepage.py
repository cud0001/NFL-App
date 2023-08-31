import streamlit as st
import pandas as pd
import nfl_data_py as nfl

st.set_page_config(layout='wide', page_title="Homepage🏈")

col1, col2=st.columns(2)
with col1:
    st.title('NFL Analytics Web Application')

    st.markdown("""
    This web application features 3 different webpages that can be used to analyze NFL play by play data
    * Built with Python 3.8
    * **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, plotly, seaborn, nfl_data_py, scikit-learn 1.1.3
    * **Data source:** Nfl-data-py, NFL Next Gen Stats powered by AWS
    * Web application built by Carmen Desiderio 
    """)
    st.markdown('#')
    st.subheader('Page Breakdown')
    st.markdown("""
    **Play By Play**
    * Features play by play data from every nfl game from 1999-present day (updated weekly)
    * **Data source:** Nfl-data-py
    * This page highlights the EPA and WPA for each play in a given game, including separate tables highlighting the top 10 plays for EPA and WPA
    * Full play descriptions are given in the play by play table
    * The chart plots the offensive and defensive EPA for each team in the year selected
    """)
    st.markdown("""
    **Player Evaluation**
    * Features statistics for offensive players in the selected week/year
    * **Data source:** Nfl-data-py and NFL Next Gen Stats
    * This page highlights advanced metrics for Quarterbacks, Receivers, and Runningbacks
    """)
    st.markdown("""
    **Run/Pass Predictor**
    * A tool powered my machine learning to predict the play and play location based on pre snap conditions
    * **Data source:** Nfl-data-py
    * Uses the Gradient boosting classifier from the Scikit-Learn python library
    * The model is trained on pre snap conditions over the last 5 year of play by play data with the target being a run or pass play, and the location of the field that the play is directed towards based on the tendencies of that team over time
    **Model accuracy by down**
    * 1st - 62.41%
    * 2nd - 67.32%
    * 3rd - 82.61%
    * 4th - 80.24%
    """)
    st.sidebar.markdown("Select a page above!")
with col2:
    st.subheader('NFL Schedule')
    selected_year=st.selectbox('Year', list(reversed(range(1999, 2024))))
    schedule=nfl.import_schedules([selected_year])
    selected_week=st.selectbox('Select Week', list(range(1, 23)))
    scheduledf=schedule [schedule ["week"] == selected_week]
    weeklyschedule=scheduledf [
        ["week", "weekday", "gameday", "gametime", "away_team", "away_score", "home_team", "home_score"]]
    weeklyschedule [['away_score']]=weeklyschedule [['away_score']].astype('Int32')
    weeklyschedule [['home_score']]=weeklyschedule [['home_score']].astype('Int32')
    weeklyschedule=weeklyschedule.rename(
        columns={'week': 'Week', 'weekday': 'Weekday', 'gameday': 'Gameday', 'gametime': 'Gametime', 'away_team': 'Away Team',
                 'away_score': 'Away Score', 'home_team': 'Home Team', 'home_score': 'Home Score'}
    )
    st.write(weeklyschedule.reset_index(drop=True))
    st.subheader('Pick Team Schedule')
    selectedteam=st.selectbox('Team', schedule ['home_team'].unique())
    teamscheduledf=schedule [schedule ["week"] == selected_week]
    teamscheduledf=schedule [(schedule ["home_team"] == selectedteam) | (schedule ["away_team"] == selectedteam)]
    displayteamsched=teamscheduledf [
        ["week", "weekday", "gameday", "gametime", "away_team", "away_score", "home_team", "home_score"]]
    displayteamsched [['away_score']]=displayteamsched [['away_score']].astype('Int32')
    displayteamsched [['home_score']]=displayteamsched [['home_score']].astype('Int32')
    displayteamsched=displayteamsched.rename(
        columns={'week': 'Week', 'weekday': 'Weekday', 'gameday': 'Gameday', 'gametime': 'Gametime', 'away_team': 'Away Team',
                 'away_score': 'Away Score', 'home_team': 'Home Team', 'home_score': 'Home Score'}
    )
    st.write(displayteamsched.reset_index(drop=True))
st.subheader('Glossary')
st.markdown("""
**Expected Points Added:**
* Measures how well a team performs relative to expectation by assigning a point value to each play
**Win Probability Added:**
* Win probability added after the play for the team with posession
**QB EPA per Dropback:**
* Expected points added for the individual QB divided by their total dropbacks within a game
**Average Cushion:**
* The distance (in yards) measured between a WR/TE and the defender they’re lined up against at the time of snap on all targets
**Average Separation:**
* The distance (in yards) measured between a WR/TE and the nearest defender at the time of catch or incompletion
**Avg YAC Above Expectation:**
* Average yards after catch above expectation
**8+ Defenders in the Box:**
* On every play, Next Gen Stats calculates how many defenders are stacked in the box at snap. Using that logic, DIB% calculates how often does a rusher see 8 or more defenders in the box against them
""")
