#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
import os.path
import json
import requests
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[2]:


# In[17]:


#ata_frames = [data_ffr, data_inf, data_coreinf, data_un, data_oil, data_indpro, data_m3, data_wages]


# In[18]:


#df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Time'], how='outer'), data_frames)


# In[19]:



# In[20]:

df_merged = pd.read_csv('inflation_forecasting/raw_data/data_us.csv')

config = {'displayModeBar': False}

fig = make_subplots(
    rows=2, cols=4,
    subplot_titles=("Fed funds rate", "CPI", "Core CPI", "Unemmployment",
                   "Oil price", "Index industrial production", "Money supply", "Wage growth"))

fig.add_trace(go.Scatter(x = df_merged["Time"],
                         y = df_merged["DFF"],
                         name = "DFF"),
              row=1, col=1)

fig.add_trace(go.Scatter(x = df_merged["Time"],
                         y = df_merged["CPIAUCSL"],
                         name = "CPIAUCSL"),
              row=1, col=2)

fig.add_trace(go.Scatter(x = df_merged["Time"],
                         y = df_merged["CPILFESL"],
                         name = "CPILFESL"),
              row=1, col=3)

fig.add_trace(go.Scatter(x = df_merged["Time"],
                         y = df_merged["UNRATE"],
                         name = "UNRATE"),
              row=1, col=4)

fig.add_trace(go.Scatter(x = df_merged["Time"],
                         y = df_merged["WTISPLC"],
                         name = "WTISPLC"),
              row=2, col=1)

fig.add_trace(go.Scatter(x = df_merged["Time"],
                         y = df_merged["INDPRO"],
                         name = "INDPRO"),
              row=2, col=2)

fig.add_trace(go.Scatter(x = df_merged["Time"],
                         y = df_merged["MABMM301USM189S"],
                         name = "MABMM301USM189S"),
              row=2, col=3)

fig.add_trace(go.Scatter(x = df_merged["Time"],
                         y = df_merged["A576RC1"],
                         name = "A576RC1"),
              row=2, col=4)

fig.update_layout(
    title = "Variables",
    autosize=False,
    width=890,
    height=600,
  legend = dict(
        xanchor = "center",
        yanchor = "top",
        y = -0.2,
        x = 0.5,
        orientation = 'h'
  )
)

fig.show(config=config)


# # Target variable: CPILFESL

# In[90]:


# X = df_merged.drop(['Time'], axis=1)

# fig = make_subplots(
#     rows=2, cols=2,
#     subplot_titles=("Plot 1", "Plot 2"))

# for column in X:
#         fig.add_trace(go.Scatter(x = df_merged["Time"],
#                              y = X[column]))

# fig.show()


# In[24]:
