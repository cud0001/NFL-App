import streamlit as st
import pandas as pd
import numpy as np
import time as tm
import seaborn as sns
from pickle import load
import nfl_data_py as nfl
from datetime import time, timedelta
import plotly.express as px
from PIL import Image

from sklearn.compose import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import *
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *


def convert_yd_line_vars(posteam, ydline):
    """
    Convert yardline feature from 'Team 25' to numerical yardline based on (100- yards from endzone)
    """
    if type(ydline) == str:
        newydline=ydline.split()
        if ydline == '50':
            return float(ydline)
        elif posteam == newydline [0]:
            return float(newydline [1])
        else:
            return 100 - float(newydline [1])
    else:
        return np.nan


# Filter for only pre-play features
def filter_data(X):
    Xdf_c=X.copy()
    pre_play_features=[
        'posteam',
        'defteam',
        'quarter_seconds_remaining',
        'half_seconds_remaining',
        'game_seconds_remaining',
        'game_half',
        'qtr',
        'goal_to_go',
        'yrdln',
        'ydstogo',
        'posteam_timeouts_remaining',
        'defteam_timeouts_remaining',
        'score_differential',
        'season'
    ]
    Xdf_c=Xdf_c [pre_play_features]
    Xdf_c ['ydstogo']=Xdf_c ['ydstogo'].astype(float)
    Xdf_c ['score_differential']=pd.cut(Xdf_c ['score_differential'],
                                        bins=[-100, -17, -12, -9, -4, 0, 4, 9, 12, 17, 100])
    Xdf_c ['yrdln']=Xdf_c.apply(lambda x: convert_yd_line_vars(x ['posteam'], x ['yrdln']), axis=1)
    return Xdf_c


pickle_in1=open('/Users/Carmen/Desktop/NFL App/pages/classifier1downrp.pkl', 'rb')
classifier1=load(pickle_in1)
pickle_in2=open('/Users/Carmen/Desktop/NFL App/pages/classifier2downrp.pkl', 'rb')
classifier2=load(pickle_in2)
pickle_in3=open('/Users/Carmen/Desktop/NFL App/pages/classifier3downrp.pkl', 'rb')
classifier3=load(pickle_in3)
pickle_in4=open('/Users/Carmen/Desktop/NFL APP/pages/classifier4downrp.pkl', 'rb')
classifier4=load(pickle_in4)


@st.cache
def prediction(user_prediction_data):
    if down == 4:
        return classifier4.predict_proba(user_prediction_data)
    elif down == 3:
        return classifier3.predict_proba(user_prediction_data)
    elif down == 2:
        return classifier2.predict_proba(user_prediction_data)
    elif down == 1:
        return classifier1.predict_proba(user_prediction_data)


st.set_page_config(layout='centered', initial_sidebar_state="expanded", page_title="Run or Pass Prediction🏈")
st.title('NFL Run or Pass Play Prediction')
intro_text="1) Use the sidebar to customize the game situation\n2) Keep track of your changes on the scoreboard ➡️\n3) Predict your play!"
st.sidebar.title('Directions:')
st.sidebar.text(intro_text)

columns=[
    'posteam',
    'defteam',
    'quarter_seconds_remaining',
    'half_seconds_remaining',
    'game_seconds_remaining',
    'game_half',
    'qtr',
    'goal_to_go',
    'yrdln',
    'ydstogo',
    'posteam_timeouts_remaining',
    'defteam_timeouts_remaining',
    'score_differential',
    'season'
]

col01, col02=st.columns(2)
col1, col2, col3=st.columns([1, 1, 2])
col11, col22=st.columns(2)
teamsdf=pd.read_csv('https://gist.githubusercontent.com/cnizzardini/' +
                    '13d0a072adb35a0d5817/raw/dbda01dcd8c86101e68cbc9fbe05e0aa6ca0305b/nfl_teams.csv')
teams=sorted(list(teamsdf.Name))
# Define User Prediction Data
st.sidebar.subheader("Pick Teams")
posteam=st.sidebar.selectbox('Team on Offense', teams, index=15)
negteam=st.sidebar.selectbox('Team on Defense', teams, index=29)
posteam_abb=teamsdf [teamsdf.Name == posteam].iloc [0, 2]
negteam_abb=teamsdf [teamsdf.Name == negteam].iloc [0, 2]
st.sidebar.subheader("What's the Score?")
posteam_score=st.sidebar.number_input('Team Points', min_value=0, max_value=50, value=0, step=1)
defteam_score=st.sidebar.number_input('Opp Team Points', min_value=0, max_value=50, value=0, step=1)
st.sidebar.subheader("Where's the Ball?")
sideoffield=st.sidebar.selectbox("Side Of Field", ['OPP', "OWN"])
ydline=st.sidebar.slider('Yard Line', min_value=1, max_value=50, value=35)
down=st.sidebar.slider('Select Down', 1, 4)
ydstogo=st.sidebar.slider('Yards To Go', min_value=1, max_value=30, value=1)
if sideoffield == 'OWN':
    side=posteam_abb
else:
    side=negteam_abb
if sideoffield == 'OPP' and ydline < 10:
    goal_to_go=1
else:
    goal_to_go=0
st.sidebar.subheader("How much Time is Left?")
quarter=st.sidebar.selectbox("Quarter", [1, 2, 3, 4])
if quarter > 2:
    half='Half2'
    halfval=2.0
else:
    half="Half1"
    halfval=1.0
time_left=st.sidebar.slider("Time Left in Quarter:", value=(time(0, 2, 0)), max_value=time(0, 15, 0),
                            step=timedelta(seconds=1), format='mm:ss')
sec_left_in_quarter=time_left.minute * 60.0 + time_left.second
sec_left_in_half=((halfval * 2) - quarter) * 15.0 * 60.0 + sec_left_in_quarter
sec_left_in_game=(2 - halfval) * 30 * 60 + sec_left_in_half

posteam_timeouts_remaining=st.sidebar.selectbox("Timeouts Left", [0, 1, 2, 3], index=3)
defteam_timeouts_remaining=st.sidebar.selectbox("Opp. Timeouts Left", [0, 1, 2, 3], index=3)
season=2022
arr=[[posteam_abb,
      negteam_abb,
      sec_left_in_quarter,
      sec_left_in_half,
      sec_left_in_game,
      half,
      quarter,
      goal_to_go,
      side + " " + str(ydline),
      ydstogo,
      posteam_timeouts_remaining * 1.0,
      defteam_timeouts_remaining * 1.0,
      int(posteam_score - defteam_score),
      season]]

teamsdf ['Name2']=teamsdf ['Name'].str.replace('NY', 'New York').str.lower()
team_str=teamsdf [teamsdf.Name == posteam].iloc [0, -1].replace(' ', '-')
oppteam_str=teamsdf [teamsdf.Name == negteam].iloc [0, -1].replace(' ', '-')
col01.image(f'images/{team_str}.png', use_column_width='always')
col02.image(f'images/{oppteam_str}.png', use_column_width='always')
user_prediction_data=pd.DataFrame(arr, columns=columns)
directions_html=f""" 
<div style="border-style: solid;
border-radius: 5px;
background-color: #000000;
border-width: 2px;
border-color: #f0f2f6;margin: 0 auto; text-align: center; width: 60%;font-size:4vw; font-weight:bold">{posteam_score}    -   {defteam_score}</div>"""
directions_html2=f""" 
<div style="padding:12px;float: left;font-size:2vw; margin-left:10px;font-weight:bold">{down} & {ydstogo}</div>
<div style="padding:12px;float: right;font-size:2vw; margin-right:10px; font-weight:bold">{sideoffield} {ydline}</div>
<div style="border-style: solid;
border-radius: 5px;
background-color: #000000;
border-width: 2px;
border-color: #f0f2f6; text-align: center;padding:10px;font-size:2vw; font-weight:bold">Q{quarter}  &nbsp;&nbsp;&nbsp;&nbsp; {time_left.minute:02d}:{time_left.second:02d} </div>"""
st.markdown(directions_html, unsafe_allow_html=True)
st.markdown(directions_html2, unsafe_allow_html=True)
st.markdown('#')

play_class=pd.read_pickle('/Users/Carmen/Desktop/NFL App/pages/team_play_freq2.pkl')
st.sidebar.markdown('###')
if st.sidebar.button("Predict Your Play!"):

    result=prediction(user_prediction_data) [0]

    resultlist=sorted(list(zip(["Pass", "Run"], result)), key=lambda x: x [1]) [::-1]
    progress_bar=st.progress(0)
    status_text=st.empty()

    for i in range(100):
        # Update progress bar.
        progress_bar.progress(i + 1)

        # Update status text.
        if i < 25:
            status_text.text('Loading Play By Play Data')
        new_rows=np.random.rand(1, 4)

        # Update status text.
        if i >= 25 and i < 50:
            status_text.text(f'{posteam_abb} offense vs {negteam_abb} defense? Hmmm...')
        if i >= 50 and i < 75:
            if ydstogo > 5:
                status_text.text(f"{down} and {ydstogo}? That's tough...")
            else:
                status_text.text(f"{down} and {ydstogo}? Seems within reach...")

        if i >= 75:
            if sideoffield == 'OWN':
                status_text.text("Not even past midfield? Risky... ")
            elif ydline < 51 and ydline > 38:
                status_text.text("Just out of field goal range...better go for it")
            else:
                status_text.text("I think we need more than a field goal here...")

        tm.sleep(0.065)
    status_text.text('')

    for play_type, prob, in resultlist:
        if prob == result.max():
            if play_type in ['Pass', 'Run']:
                df=play_class [
                    (play_class.posteam == posteam_abb) & (play_class.play_type == play_type.lower()) & (
                        play_class.play_class.str.contains(play_type.lower())) & (
                        play_class ['yd_bucket'].apply(lambda x: ydstogo in x))]
                play=df [df.play_id == df.play_id.max()] ['play_class'].iloc [0].title()
                st.title(f'{play}')
            else:
                st.title(f'{play_type}')

            st.title(f'{play_type} (in {prob*100:.1f}% of similar situations)')
        else:
            st.markdown(f'{play_type} ({prob*100:.1f}%)')


st.sidebar.markdown('##')
