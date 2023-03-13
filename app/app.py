import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
import requests

urlAPI = "http://127.0.0.1:8000/predict"

st.markdown("""# Inflation predictor
## working title""")
#
#the relative path to the data
<<<<<<< HEAD
csv_path = os.path.join('..','raw_data')
df = pd.read_csv(os.path.join(csv_path,'data_final.csv'))
=======
csv_path = os.path.join('..','inflation-forecasting','raw_data')
file_path = os.path.join(csv_path,'data_final.csv')
df = pd.read_csv(file_path)
>>>>>>> d982eebd6165a0a2c8d093f1dc0ff96132cbc17d

# this slider allows the user to select a number of lines
# to display in the dataframe
# the selected value is returned by st.slider
country_list = df['country'].unique()
with st.sidebar:
    country = st.selectbox('Please select the country you would like to see', country_list)
    inflation_type = st.radio("Select Inflation Type", ('Headline Inflation', 'Core Inflation'))


#st.write('You selected:', country)



df_country = df[df['country'] == country]

fig = go.Figure()

fig.add_trace(go.Scatter(x = df_country["year"],
                         y = df_country["ccpi"],
                         name = "Inflation development"))

fig.update_layout(
    title = "Inflation",
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

st.plotly_chart(fig,use_container_width=True)

if st.button("Predict"):
    st.spinner("Waiting for prediction")
    response = requests.post(urlAPI, files = {"csv_file": file_path})
    result = response.json()["Status"]
    st.write(result)
