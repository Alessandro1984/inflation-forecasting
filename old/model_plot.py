#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# # Load data

# In[2]:


os.getcwd()

csv_path = os.path.join('..', 'inflation-forecasting', 'raw_data')

df = pd.read_csv(os.path.join(csv_path,'data_final.csv'), index_col=0)


# In[3]:


df.columns


# In[4]:


countries  = df['country_id'].unique()

df_us = df[df['country_id'] == "USA"]
df_de = df[df['country_id'] == "DE"]
df_nl = df[df['country_id'] == "NL"]
df_it = df[df['country_id'] == "IT"]
df_fr = df[df['country_id'] == "FR"]
df_es = df[df['country_id'] == "ES"]


# # Inflation (CPI)

# In[5]:


config = {'displayModeBar': False}

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=("United States", "Germany", "Netherlands", "Italy", "France", "Spain"))

fig.add_trace(go.Scatter(x = df_us["year"],
                         y = df_us["cpi"],
                         name = "United States"),
              row=1, col=1)

fig.add_trace(go.Scatter(x = df_de["year"],
                         y = df_de["cpi"],
                         name = "Germany"),
              row=1, col=2)

fig.add_trace(go.Scatter(x = df_nl["year"],
                         y = df_nl["cpi"],
                         name = "Netherlands"),
              row=1, col=3)

fig.add_trace(go.Scatter(x = df_it["year"],
                         y = df_it["cpi"],
                         name = "Italy"),
              row=2, col=1)

fig.add_trace(go.Scatter(x = df_fr["year"],
                         y = df_fr["cpi"],
                         name = "France"),
              row=2, col=2)

fig.add_trace(go.Scatter(x = df_es["year"],
                         y = df_es["cpi"],
                         name = "Spain"),
              row=2, col=3)

fig.update_layout(
    title = "CPI",
    autosize=False,
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


# # Core inflation (CCPI)

# In[6]:


config = {'displayModeBar': False}

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=("United States", "Germany", "Netherlands", "Italy", "France", "Spain"))

fig.add_trace(go.Scatter(x = df_us["year"],
                         y = df_us["ccpi"],
                         name = "United States"),
              row=1, col=1)

fig.add_trace(go.Scatter(x = df_de["year"],
                         y = df_de["ccpi"],
                         name = "Germany"),
              row=1, col=2)

fig.add_trace(go.Scatter(x = df_nl["year"],
                         y = df_nl["ccpi"],
                         name = "Netherlands"),
              row=1, col=3)

fig.add_trace(go.Scatter(x = df_it["year"],
                         y = df_it["ccpi"],
                         name = "Italy"),
              row=2, col=1)

fig.add_trace(go.Scatter(x = df_fr["year"],
                         y = df_fr["ccpi"],
                         name = "France"),
              row=2, col=2)

fig.add_trace(go.Scatter(x = df_es["year"],
                         y = df_es["ccpi"],
                         name = "Spain"),
              row=2, col=3)

fig.update_layout(
    title = "CCPI",
    autosize=False,
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


# # Load the model

# In[7]:


model_usa = load_model("model_usa.h5")


# # Prepare data for LSTM + forecast function

# In[16]:


def future_forecasting(data_inflation, model, mb=12, mf=12):
    '''
    Returns the future forecasting of the model. Please select the dataset and model you want to use.
    for mb, select the number of months you want to look back to make a prediction.
    for mf, select the number of months you want the prediction to look forward.
    '''
    ccpi = data_inflation.to_numpy()
    scaler = MinMaxScaler(feature_range = (0, 1))
    dataset = scaler.fit_transform(ccpi.reshape(-1,1))
    results = []
    x_input = dataset[-mb:].reshape(1, mb, 1)
    for num in range(mf):
        forecast = model.predict(x_input)
        results.append(forecast[0][0])
        x_input = np.roll(x_input, -1, axis=1)
        x_input[0][-1] = forecast
    results = scaler.inverse_transform(np.array(results).reshape(-1, 1)) # reshape dataset to 2D array
    results = np.array(results).reshape(-1, 1)
    return results


# # Plot function

# In[17]:


def plot_function(data_country, data_forecast_country):
    
    first_date = data_country['year'].min()
    
    last_date = pd.to_datetime(data_country['year'].max()) + pd.offsets.MonthBegin(1)
    
    country_name = data_country["country"].unique()[0]
    
    t = pd.date_range(last_date, periods = user_input, freq='MS')
    forecast_date = t.max()
    out_forecast_df = pd.DataFrame([[x, y] for x, y in zip(t, data_forecast_country)], columns=["year", "Forecast"])
    
    out_forecast_df['Forecast'] = out_forecast_df['Forecast'].apply(lambda x: np.ravel(x)[0])
    
    config = {'displayModeBar': False}
    fig = go.Figure()
    df_short = data_country.loc[data_country["year"] >= "2015-01-01"]

    fig.add_trace(go.Scatter(x = df_short["year"], 
                             y = df_short['ccpi'],
                             line_color = "blue",
                             name = "Core CPI", 
                             mode = "lines"))

    fig.add_trace(go.Scatter(x = out_forecast_df["year"], 
                             y = out_forecast_df['Forecast'], 
                             name = "Forecasted Core CPI",
                             line_dash = "dash", 
                             line_color = "black",
                             mode = "lines"))

    fig.add_vrect(x0 = last_date, 
                  x1 = forecast_date,  
                  fillcolor = "grey", 
                  opacity = 0.25, 
                  line_width = 0)

    fig.update_layout(
        title = f"Out-of-sample forecast until {forecast_date.strftime('%B %Y')} for {country_name}", 
        xaxis_title = "", 
        yaxis_title = "Monthly y-o-y percentage change",
        autosize=False,
        hoverlabel_namelength=-1,
        width=890,
        height=600,
      legend = dict(
            xanchor = "center",
            yanchor = "top",
            y = -0.2, 
            x = 0.5,
            orientation = 'h'
      ),
      xaxis=dict(
          dtick='M12',
          tickangle=45,
          tickfont=dict(size=14)
        ),
      yaxis=dict(
          tickfont=dict(size=14)
        )
    )
    
    fig.update_layout(
    plot_bgcolor='white')
    
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey')

    fig.show(config=config)


# # Test of the plot function

# - The function takes two arguments:
#     - the dataframe of a single country, e.g. *df_us*
#     - the dataframe of the forecast created with the *future_forecasting* function
# - The function takes care of the time range and of the shaded area
# - If we decide to forecast CPI and not CCPI, we have to change the name manually
# - The function use the starting date "2015-01-01" but it can be manually changed

# This is the number of month of the forecast input by the user in the **streamlit app**.

# In[18]:


user_input = 48


# In[19]:


inf_for_us = future_forecasting(data_inflation = df_us["ccpi"], 
                                model = model_usa, 
                                mb = 12, 
                                mf = user_input)


# In[20]:


plot_function(data_country = df_us, 
              data_forecast_country = inf_for_us)

