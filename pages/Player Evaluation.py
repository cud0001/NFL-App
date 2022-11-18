import streamlit as st
import time
import pandas as pd
import base64
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import urllib.request
import seaborn as sns
import numpy as np
import nfl_data_py as nfl
from st_aggrid import AgGrid
from IPython.core.display import Image, HTML

st.set_page_config(layout='wide', page_title="Player Evaluation🏈")
st.title('Player Evaluation')

st.sidebar.header('Filter Week')
selected_year=st.sidebar.selectbox('Year', list(reversed(range(1999, 2023))))


@st.cache(allow_output_mutation=True)
def get_pbp():
    data=nfl.import_pbp_data([selected_year], downcast=True)

    return data


@st.cache
def get_ngs_rec():
    receivingdata=nfl.import_ngs_data('receiving')

    return receivingdata


@st.cache
def get_ngs_rus():
    rushingdata=nfl.import_ngs_data('rushing')

    return rushingdata


my_bar=st.progress(0)
colors=nfl.import_team_desc()
for percent_complete in range(100):
    time.sleep(0.035)
    my_bar.progress(percent_complete + 1)

df=get_pbp()

selected_week=st.sidebar.slider('Select Week', 1, 20)
# Sidebar - game selection

df_week=df [df ["week"] == selected_week]

qbdf=df_week [df_week ["qb_dropback"] == True]

qb_epa_df=pd.DataFrame({
    'qb_epa': qbdf.groupby('passer_player_name') ['qb_epa'].sum(),
    'qb_dropbacks': qbdf.groupby('passer_player_name') ['qb_dropback'].sum(),
    'qb_yards': qbdf.groupby('passer_player_name') ['passing_yards'].sum(),
})
qb_epa_df ['qb_epa/dropback']=qb_epa_df ['qb_epa'] / qb_epa_df ['qb_dropbacks']
qb_epa_df1=qb_epa_df [qb_epa_df.qb_dropbacks > 5]

qb_epa_df1=qb_epa_df1.rename(
    columns={'qb_epa': 'QB EPA', 'qb_dropbacks': 'QB Dropbacks', 'qb_yards': 'QB Yards',
             'qb_epa/dropback': 'QB EPA per Dropback'})

format_dict={
    'QB EPA': '{:,.2f}',
    'QB EPA per Dropback': '{:,.2f}',
    'QB Dropbacks': int,
    'QB Yards': int

}
qb_epa_df1.reset_index(inplace=True)
qb_epa_df1=qb_epa_df1.rename(columns={'passer_player_name': 'Quarterback'})
qb_epa_df2=qb_epa_df1.merge
# open figure + axis
fig1, ax=plt.subplots()
ax.set_facecolor(black)
# plot
ax.scatter(x=qb_epa_df1 ['QB Dropbacks'], y=qb_epa_df1 ['QB EPA per Dropback'], c='DarkBlue')
# set labels
ax.set_xlabel('QB Dropbacks')
ax.set_ylabel('QB EPA per Dropback')

# annotate points in axis
for idx, row in qb_epa_df1.iterrows():
    ax.annotate(row ['Quarterback'], (row ['QB Dropbacks'], row ['QB EPA per Dropback']))

st.header('Quaterback EPA Evaluation')
col1, col2=st.columns(2)
with col1:
    st.pyplot(fig1)
with col2:
    st.write(qb_epa_df1.sort_values(by='QB EPA per Dropback', ascending=False).style.format(format_dict))

st.header('Receiver Evaluation')

receiving_stat_df=get_ngs_rec()
receiving_stat_df=receiving_stat_df.merge(colors [['team_abbr', 'team_color']])
receiving_stat_df=receiving_stat_df [receiving_stat_df ['week'] == selected_week]
receiving_stat_df=receiving_stat_df [receiving_stat_df ['season'] == selected_year]
sorted_unique_team=sorted(receiving_stat_df.team_abbr.unique())
selected_team=st.multiselect('Teams', sorted_unique_team, sorted_unique_team)
receiving_stat_df=receiving_stat_df [receiving_stat_df.team_abbr.isin(selected_team)]
receiving_stat_df=receiving_stat_df.reset_index()

fig2, ax=plt.subplots()
# plot
ax.scatter(x=receiving_stat_df ['avg_cushion'], y=receiving_stat_df ['avg_separation'],
           c=receiving_stat_df ['team_color'])
# set labels
ax.set_xlabel('Average Cushion')
ax.set_ylabel('Average Separation')

for idx, row in receiving_stat_df.iterrows():
    ax.annotate(row ['player_display_name'], (row ['avg_cushion'], row ['avg_separation']))
st.pyplot(fig2)
receivingdf=receiving_stat_df [
    ["season", "week", "player_display_name", "player_position", "team_abbr", "receptions", "targets", "yards",
     "avg_yac_above_expectation", "avg_cushion", "avg_separation"]]
receivingdf=receivingdf.rename(
    columns={'season': 'Season', 'week': 'Week', 'player_display_name': 'Player', 'player_position': 'Position',
             'team_abbr': 'Team', 'receptions': 'Receptions', 'targets': 'Targets', 'yards': 'Yards',
             'avg_yac_above_expectation': 'Avg YAC Above Expectation', 'avg_cushion': 'Avg Cushion',
             'avg_separation': 'Avg Separation',
             })

format_dict3={
    'Avg YAC Above Expectation': '{:,.2f}',
    'Avg Cushion': '{:,.2f}',
    'Avg Separation': '{:,.2f}',
    'Receptions': int,
    'Targets': int,
    'Yards': int

}

st.write(receivingdf.style.format(format_dict3))

st.header('Rushing Evaluation')

rushing_stat_df=get_ngs_rus()
rushing_stat_df=rushing_stat_df.merge(colors [['team_abbr', 'team_color']])
rushing_stat_df=rushing_stat_df [rushing_stat_df ['week'] == selected_week]
rushing_stat_df=rushing_stat_df [rushing_stat_df ['season'] == selected_year]
rushing_stat_df=rushing_stat_df [rushing_stat_df ['rush_attempts'] > 5]
sorted_unique_team2=sorted(rushing_stat_df.team_abbr.unique())
selected_team2=st.multiselect('Teams', sorted_unique_team2, sorted_unique_team2)
rushing_stat_df=rushing_stat_df [rushing_stat_df.team_abbr.isin(selected_team2)]
rushing_stat_df=rushing_stat_df.reset_index()

fig3, ax=plt.subplots()
# plot
ax.scatter(x=rushing_stat_df ['percent_attempts_gte_eight_defenders'],
           y=rushing_stat_df ['rush_yards_over_expected_per_att'], c=rushing_stat_df ['team_color'])
# set labels
ax.set_xlabel('% of Attempts With 8 Defenders in the Box')
ax.set_ylabel('Rush Yards Over Expected Per Att')

for idx, row in rushing_stat_df.iterrows():
    ax.annotate(row ['player_display_name'],
                (row ['percent_attempts_gte_eight_defenders'], row ['rush_yards_over_expected_per_att']))
st.pyplot(fig3)
rushingdf=rushing_stat_df [["season", "week", "player_display_name", "player_position", "team_abbr",
                            "percent_attempts_gte_eight_defenders", "rush_yards_over_expected_per_att",
                            "rush_attempts", "rush_yards"]]
rushingdf=rushingdf.rename(
    columns={'season': 'Season', 'week': 'Week', 'player_display_name': 'Player', 'team_abbr': 'Team',
             'percent_attempts_gte_eight_defenders': '% Attempts with 8 Defenders in the Box',
             'rush_yards_over_expected_per_att': 'Rush Yards Over Expected Per Attempt', 'rush_attempts': 'Attempts',
             'rush_yards': 'Yards'
             })
format_dict2={
    '% Attempts with 8 Defenders in the Box': '{:,.2f}',
    'Rush Yards Over Expected Per Attempt': '{:,.2f}'

}
st.write(rushingdf.style.format(format_dict2))
