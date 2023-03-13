import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
import requests

urlAPI = "http://127.0.0.1:8000/predict"

st.markdown("""# use deep learning to predict inflation!
## use the predict button below""")
#
#the relative path to the data
csv_path = os.path.join('..','raw_data')
df = pd.read_csv(os.path.join(csv_path,'data_final.csv'))

# this slider allows the user to select a number of lines
# to display in the dataframe
# the selected value is returned by st.slider
country_list = df['country'].unique()
with st.sidebar:
    # Add custom CSS for the sidebar
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            padding-top: 20px;
        }
        .sidebar .title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .sidebar .widget-label {
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .sidebar .widget-box {
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add inputs to the sidebar
    st.markdown("<div class='title'>Please select the options:</div>", unsafe_allow_html=True)
    country = st.selectbox('Country', country_list)
    st.markdown("<div class='widget-box'></div>", unsafe_allow_html=True)
    inflation_type = st.radio("Inflation Type", ('Headline Inflation', 'Core Inflation'))
    st.markdown("<div class='widget-box'></div>", unsafe_allow_html=True)
    num_months = st.slider('Number of Months to Predict', 1, 48)

if inflation_type == 'Headline Inflation':
    inflation_type = 'cpi'
else:
    inflation_type = 'ccpi'

df_country = df[df['country'] == country]
fig = go.Figure()

fig.add_trace(go.Scatter(x = df_country["year"],
                         y = df_country[inflation_type],
                         name = f"{inflation_type} development"))

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
    st.spinner("Waiting for prediction...")
    response = requests.post(urlAPI, files = {"csv_file": file_path})
    result = response.json()["Status"]
    st.write(result)
