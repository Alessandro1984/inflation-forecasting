import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

path = os.path.dirname(__file__)

file_path = os.path.join(path, "df_HCPI.csv")

df = pd.read_csv(file_path)

print(df)

countries  = df['Country'].unique()
df_us = df[df['Country'] == "US"]
df_de = df[df['Country'] == "DE"]
df_nl = df[df['Country'] == "NL"]
df_it = df[df['Country'] == "IT"]
df_fr = df[df['Country'] == "FR"]
df_es = df[df['Country'] == "ES"]

config = {'displayModeBar': False}

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=("United States", "Germany", "Netherlands", "Italy", "France", "Spain"))

fig.add_trace(go.Scatter(x = df_us["Time"],
                         y = df_us["HCPI"],
                         name = "United States"),
              row=1, col=1)

fig.add_trace(go.Scatter(x = df_de["Time"],
                         y = df_de["HCPI"],
                         name = "Germany"),
              row=1, col=2)

fig.add_trace(go.Scatter(x = df_nl["Time"],
                         y = df_nl["HCPI"],
                         name = "Netherlands"),
              row=1, col=3)

fig.add_trace(go.Scatter(x = df_it["Time"],
                         y = df_it["HCPI"],
                         name = "Italy"),
              row=2, col=1)

fig.add_trace(go.Scatter(x = df_fr["Time"],
                         y = df_fr["HCPI"],
                         name = "France"),
              row=2, col=2)

fig.add_trace(go.Scatter(x = df_es["Time"],
                         y = df_es["HCPI"],
                         name = "Spain"),
              row=2, col=3)

fig.update_layout(
    title = "HCPI",
    autosize = False,
    width = 890,
    height = 600,
  legend = dict(
        xanchor = "center",
        yanchor = "top",
        y = -0.2,
        x = 0.5,
        orientation = 'h'
  )
)

fig.show(config=config)
