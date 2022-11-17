import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import os
import urllib.request
import seaborn as sns
import numpy as np
import nfl_data_py as nfl
from st_aggrid import AgGrid
from IPython.core.display import Image, HTML
import requests
import time
from io import BytesIO

st.set_page_config(layout='wide', page_title="PBP Breakdown🏈")
st.title('WPA and EPA Breakdown')

st.markdown("""
Select the game using the sidebar
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, nfl_data_py
* **Data source:** Nfl-data-py.
""")

st.sidebar.header('Filter Games')
selected_year=st.sidebar.selectbox('Year', list(reversed(range(1999, 2023))))


@st.cache(allow_output_mutation=True)
def get_pbp():
    data=nfl.import_pbp_data([selected_year], downcast=True)

    return data


my_bar=st.progress(0)

for percent_complete in range(100):
    time.sleep(0.035)
    my_bar.progress(percent_complete + 1)

df=get_pbp()

selected_week=st.sidebar.slider('Select Week', 1, 22)
# Sidebar - game selection

df_week=df [df ["week"] == selected_week]
sorted_unique_game=sorted(df_week.game_id.unique())

selected_game=st.sidebar.selectbox('Game', sorted_unique_game)

# Filtering data
df_selected_game=df [df ["game_id"] == selected_game]

game_pbp=df_selected_game [["qtr", "time", "desc", "down", "ydstogo", "posteam", "epa", "wpa", "wp"]]
game_pbp=game_pbp [~game_pbp ['down'].isnull()]
game_pbp=game_pbp [~game_pbp ['ydstogo'].isnull()]
game_pbp [['down']]=game_pbp [['down']].astype(int)
game_pbp [['ydstogo']]=game_pbp [['ydstogo']].astype(int)
wpa_table=df_selected_game [["game_id", "qtr", "time", "desc", "posteam", "wpa"]]
nfl.clean_nfl_data(game_pbp)

game_pbp ['WP After Play']=game_pbp.wp + game_pbp.wpa

game_pbp2=game_pbp.rename(
    columns={'qtr': 'Quarter', 'time': 'Time', 'desc': 'Play Description', 'posteam': 'Team with Possession',
             'epa': 'Expected Points Added', 'wp': 'Win Probability', 'wpa': 'Win Probability Added', 'down': 'Down',
             'ydstogo': 'Yards to go'})

format_dict={
    'Quarter': int,
    'Expected Points Added': '{:,.1f}',
    'Win Probability Added': '{:,.3f}',
    'Win Probability': '{:,.2%}',
    'WP After Play': '{:,.2%}'
}

game_pbp3=game_pbp2.style.format(format_dict)

wpa_table=df_selected_game [["game_id", "qtr", "time", "desc", "posteam", "wpa"]]

top_ten_wpa=wpa_table.nlargest(10, 'wpa')

top_ten_wpa=top_ten_wpa.rename(
    columns={'qtr': 'Quarter', 'time': 'Time', 'desc': 'Play Description', 'posteam': 'Team with Possession',
             'wpa': 'Win Probability Added', })

epa_table=df_selected_game [["game_id", "qtr", "time", "desc", "posteam", "epa"]]

top_ten_epa=epa_table.nlargest(10, 'epa')

top_ten_epa=top_ten_epa.rename(
    columns={'qtr': 'Quarter', 'time': 'Time', 'desc': 'Play Description', 'posteam': 'Team with Possession',
             'epa': 'Expected Points Added', })

home_score_df=df_selected_game [["game_id", "home_team", "home_score"]]
away_score_df=df_selected_game [["game_id", "away_team", "away_score"]]

teamsdf=pd.read_csv('https://gist.githubusercontent.com/cnizzardini/' +
                    '13d0a072adb35a0d5817/raw/dbda01dcd8c86101e68cbc9fbe05e0aa6ca0305b/nfl_teams.csv')
teams=sorted(list(teamsdf.Name))

st.header('Play by Play')
st.write(game_pbp3)
st.header('Top 10 Plays by Win Probability Added')
st.write(top_ten_wpa.style.format(format_dict))
st.header('Top 10 Plays by Expected Points Added')
st.write(top_ten_epa.style.format(format_dict))

epa_df=pd.DataFrame({
    'offense_epa': df.groupby('posteam') ['epa'].sum(),
    'offense_plays': df ['posteam'].value_counts(),
    'offense_yards': df.groupby('posteam') ['yards_gained'].sum(),
})

epa_df ['offense_epa/play']=epa_df ['offense_epa'] / epa_df ['offense_plays']
epa_df ['defense_epa']=df.groupby('defteam') ['epa'].sum()
epa_df ['defense_plays']=df ['defteam'].value_counts()
epa_df ['defense_epa/play']=epa_df ['defense_epa'] / epa_df ['defense_plays']
epa_df ['defense_yards_given_up']=df.groupby('defteam') ['yards_gained'].sum()

plt.style.use('ggplot')
x=epa_df ['offense_epa/play'].values
y=epa_df ['defense_epa/play'].values

fig, ax=plt.subplots(figsize=(20, 15))

ax.grid(alpha=0.5)
# plot a vertical and horixontal line to create separate quadrants
ax.vlines(0, -0.3, 0.3, color='#fcc331', alpha=0.7, lw=4, linestyles='dashed')
ax.hlines(0, -0.3, 0.3, color='#fcc331', alpha=0.7, lw=4, linestyles='dashed')
ax.set_ylim(-0.3, 0.3)
ax.set_xlim(-0.3, 0.3)
ax.set_xlabel('Offense EPA/play', fontsize=20)
ax.set_ylabel('Defense EPA/play', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

annot_styles={
    'bbox': {'boxstyle': 'round,pad=0.5', 'facecolor': 'none', 'edgecolor': '#fcc331'},
    'fontsize': 15,
    'color': '#202f52'
}

# annotate the quadrants
ax.annotate('Good offense, good defense', xy=(x.max() - 0.06, y.max() - 0.3), **annot_styles)
ax.annotate('Bad offense, good defense', xy=(x.min(), y.max() - 0.3), **annot_styles)
ax.annotate('Good offense, bad defense', xy=(x.max() - 0.06, y.max() + 0.02), **annot_styles)
ax.annotate('Bad offense, bad defense', xy=(x.min(), y.max() + 0.02), **annot_styles)

team_colors=pd.read_csv('https://raw.githubusercontent.com/guga31bb/nflfastR-data/master/teams_colors_logos.csv')
# annotate the points with team logos
for idx, row in epa_df.iterrows():
    offense_epa=row ['offense_epa/play']
    defense_epa=row ['defense_epa/play']
    logo_src=team_colors [team_colors ['team_abbr'] == idx] ['team_logo_espn'].values [0]
    res=requests.get(logo_src)
    img=plt.imread(BytesIO(res.content))
    ax.imshow(img, extent=[row ['offense_epa/play'] - 0.0085, row ['offense_epa/play'] + 0.0085,
                           row ['defense_epa/play'] - 0.00725, row ['defense_epa/play'] + 0.00725], aspect='equal',
              zorder=1000)

ax.set_title(f'Offense EPA and Defense EPA for {selected_year}', fontsize=20);
st.pyplot(fig)

wpcols=['home_wp', 'home_team', 'away_wp', 'away_team', 'game_seconds_remaining', 'home_score', 'away_score']
winpbdf=df_selected_game [wpcols].dropna()
COLORS = {'ARI':'#97233F','ATL':'#A71930','BAL':'#241773','BUF':'#00338D',
          'CAR':'#0085CA','CHI':'#00143F','CIN':'#FB4F14','CLE':'#FB4F14',
          'DAL':'#7F9695','DEN':'#FB4F14','DET':'#046EB4','GB':'#2D5039',
          'HOU':'#C9243F','IND':'#003D79','JAX':'#136677','KC':'#CA2430',
          'LA':'#FFA300','LAC':'#2072BA','LV':'#343434','MIA':'#0091A0',
          'MIN':'#4F2E84','NE':'#0A2342','NO':'#A08A58','NYG':'#192E6C',
          'NYJ':'#203731','PHI':'#014A53','PIT':'#FFC20E','SEA':'#7AC142',
          'SF':'#C9243F','TB':'#D40909','TEN':'#4095D1','WAS':'#FFC20F'}

# Set style
plt.style.use('dark_background')

# Create a figure
fig2, ax=plt.subplots(figsize=(16, 8))

# Generate lineplots
awayteam=winpbdf['away_team'].iat[0]
hometeam=winpbdf['home_team'].iat[0]
sns.lineplot('game_seconds_remaining', 'away_wp',
             data=winpbdf, color=COLORS.get(awayteam), linewidth=2)

sns.lineplot('game_seconds_remaining', 'home_wp',
             data=winpbdf, color=COLORS.get(hometeam), linewidth=2)

# Generate fills for the favored team at any given time
ax.fill_between(winpbdf ['game_seconds_remaining'], 0.5, winpbdf ['away_wp'],
                where=winpbdf ['away_wp'] > .5, color=COLORS.get(awayteam), alpha=0.3)

ax.fill_between(winpbdf ['game_seconds_remaining'], 0.5, winpbdf ['home_wp'],
                where=winpbdf ['home_wp'] > .5, color=COLORS.get(hometeam), alpha=0.3)

# Labels
plt.ylabel('Win Probability %', fontsize=16)
plt.xlabel('', fontsize=16)

# Divider lines for aesthetics
plt.axvline(x=900, color='white', alpha=0.7)
plt.axvline(x=1800, color='white', alpha=0.7)
plt.axvline(x=2700, color='white', alpha=0.7)
plt.axhline(y=.50, color='white', alpha=0.7)

# Format and rename xticks
ax.set_xticks(np.arange(0, 3601, 900))

plt.gca().invert_xaxis()
x_ticks_labels=['End', 'End Q3', 'Half', 'End Q1', 'Kickoff']
ax.set_xticklabels(x_ticks_labels, fontsize=12)
hometeam=winpbdf ['home_team'].iat [0]
awayteam=winpbdf ['away_team'].iat [0]
homescore=winpbdf['home_score'].iat[0]
awayscore=winpbdf['away_score'].iat[0]
plt.suptitle(f'{awayteam}@{hometeam}',
             fontsize=20, style='italic', weight='bold')

plt.title(f'{awayteam} {awayscore}-{hometeam} {homescore} Week {selected_week}, {selected_year}', fontsize=16,
          style='italic', weight='semibold')


# Citations
plt.figtext(0.131, 0.137, 'Data: @nflfastR')

st.pyplot(fig2)
# Titles
with st.sidebar:
    st.header('Final Score')
    col1, col2=st.columns(2)
    col3, col4=st.columns(2)
    col1.metric(home_score_df ['home_team'].iat [0], home_score_df ['home_score'].iat [0])
    col2.metric(away_score_df ['away_team'].iat [0], away_score_df ['away_score'].iat [0])
    team_str=home_score_df ['home_team'].iat [0]
    oppteam_str=away_score_df ['away_team'].iat [0]
    col3.image(f'images2/{team_str}.png', use_column_width='always')
    col4.image(f'images2/{oppteam_str}.png', use_column_width='always')
