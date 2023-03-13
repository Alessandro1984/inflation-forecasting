import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os

st.markdown("""# Inflation predictor
## working title""")
#
#the relative path to the data
csv_path = os.path.join('..','raw_data')
df = pd.read_csv(os.path.join(csv_path,'data_final.csv'))

# this slider allows the user to select a number of lines
# to display in the dataframe
# the selected value is returned by st.slider
country_list = df['country'].unique()
with st.sidebar:
    country = st.selectbox('Please select the country you would like to see', country_list)
    inflation_type = st.radio("Select Inflation Type", ('Headline Inflation', 'Core Inflation'))


st.write('You selected:', country)

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
