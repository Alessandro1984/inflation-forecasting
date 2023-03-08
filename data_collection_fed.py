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


with open("secrets_file.json", "r") as f:
    secrets = json.load(f)

api_key = secrets["api_key"]


# In[3]:


class FredPy:

    def __init__(self, token=None):
        self.token = token
        self.url = "https://api.stlouisfed.org/fred/series/observations" + \
                   "?series_id={seriesID}&api_key={key}&file_type=json" + \
                   "&observation_start={start}&observation_end={end}&frequency={frequency}&units={units}"

    def set_token(self, token):
        self.token = token

    def get_series(self, seriesID, start, end, frequency, units):
        url_formatted = self.url.format(seriesID = seriesID,
                                        start = start,
                                        end = end,
                                        frequency=frequency,
                                        units = units,
                                        key = self.token)

        response = requests.get(url_formatted)

        if (self.token):
            if (response.status_code == 200):
                df = pd.DataFrame(response.json()['observations'])[['date', 'value']]\
                                .assign(date = lambda cols: pd.to_datetime(cols["date"]))\
                                .assign(value = lambda cols: cols["value"].astype(float))\
                                .rename(columns = {"value": seriesID,
                                                   "date": "Time"})
                return df
            else:
                raise Exception("Bad response from API, status code = {}".format(response.status_code))
        else:
            raise Exception("You did not specify an API key.")


# # Initialize class

# In[7]:


fredpy = FredPy()

fredpy.set_token(api_key)


# # Federal funds rate

# In[9]:


data_ffr = fredpy.get_series(
    seriesID = "DFF",
    start = "1960-01-01",
    end = "2023-01-01",
    frequency = "m",
    units = "lin")


# # Consumer price index

# In[10]:


data_inf = fredpy.get_series(
    seriesID = "CPIAUCSL",
    start = "1960-01-01",
    end = "2023-01-01",
    frequency = "m",
    units = "pc1")


# # Core consumer price index

# In[11]:


data_coreinf = fredpy.get_series(
    seriesID = "CPILFESL",
    start = "1960-01-01",
    end = "2023-01-01",
    frequency = "m",
    units = "pc1")


# # Unemployment rate

# In[12]:


data_un = fredpy.get_series(
    seriesID = "UNRATE",
    start = "1960-01-01",
    end = "2023-01-01",
    frequency = "m",
    units = "lin")


# # Oil price

# In[13]:


data_oil = fredpy.get_series(
    seriesID = "WTISPLC",
    start = "1960-01-01",
    end = "2023-01-01",
    frequency = "m",
    units = "lin")


# # Index of industrial production

# In[14]:


data_indpro = fredpy.get_series(
    seriesID = "INDPRO",
    start = "1960-01-01",
    end = "2023-01-01",
    frequency = "m",
    units = "pc1")


# # Money supply M3

# In[15]:


# Only from 1961 until December 2022
data_m3 = fredpy.get_series(
    seriesID = "MABMM301USM189S",
    start = "1961-01-01",
    end = "2022-12-01",
    frequency = "m",
    units = "pc1")


# # Wage growth

# In[16]:


data_wages = fredpy.get_series(
    seriesID = "A576RC1",
    start = "1960-01-01",
    end = "2023-01-01",
    frequency = "m",
    units = "pc1")


# In[17]:


data_frames = [data_ffr, data_inf, data_coreinf, data_un, data_oil, data_indpro, data_m3, data_wages]


# In[18]:


df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Time'], how='outer'), data_frames)


# In[19]:


df_merged


# In[20]:


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


#df_merged.to_csv(os.path.join('data','data.csv'))
