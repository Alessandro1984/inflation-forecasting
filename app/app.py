import streamlit as st
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
import requests
import json

#urlAPI = "http://127.0.0.1:8000/predict"
#urlAPItest = "http://127.0.0.1:8000/test"
urlAPI = "http://127.0.0.1:8000"

# dict_xxx = {
#   "predictions": "[[5.426177978515625], [5.317831993103027], [5.216428756713867], [5.11897087097168], [5.024357795715332], [4.9318342208862305], [4.84096097946167], [4.751753807067871], [4.6642231941223145], [4.57786750793457], [4.492310047149658], [4.407289981842041], [4.322705268859863], [4.238522052764893], [4.154750347137451], [4.071430683135986], [3.9886269569396973], [3.9064197540283203], [3.824902296066284], [3.744175910949707], [3.6643474102020264], [3.58552622795105], [3.507821798324585], [3.431339979171753]]"
# }

# predictions = json.loads(dict_xxx['predictions'])

# predictions = [[float(x[0])] for x in predictions]  # wrap each value in a list to create a 2D array

# predictions = np.array(predictions)

st.markdown("""# use deep learning to predict inflation!
## use the predict button below""")
#
#the relative path to the data
csv_path = os.path.join('..','raw_data')
df = pd.read_csv(os.path.join(csv_path,'data_final.csv'))

# this slider allows the user to select a number of lines
# to display in the dataframe
# the selected value is returned by st.slider
country_list = df['country_id'].unique()

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
    # Select box country
    country = st.selectbox('Country', country_list)
    st.markdown("<div class='widget-box'></div>", unsafe_allow_html=True)
    # Button inflation
    inflation_type = st.radio("Inflation Type", ('Headline Inflation', 'Core Inflation'))
    st.markdown("<div class='widget-box'></div>", unsafe_allow_html=True)
    # Slider month
    num_months = st.slider('Number of Months to Predict', 1, 48)

if inflation_type == 'Headline Inflation':
    inflation_type = 'cpi'
else:
    inflation_type = 'ccpi'

df_country = df[df['country'] == country]

# fig = plot_function(data_country = df_country,
#                     data_forecast_country = predictions,
#                     user_input = predictions.shape[0])

# st.plotly_chart(fig,
#                 use_container_width=True)

if st.button("predict"):

    # payload = {'num_months': num_months}
    payload = {'country': country,
               'inflation_type': inflation_type,
               'num_months': num_months}

    # response_model = requests.post(f"{urlAPI}/model",
    #                                data = payload)

    file_csv = os.path.join(csv_path,'data_final.csv')
    data_df = pd.read_csv(file_csv, index_col=0)
    # print(data_df.head())
    data_df = data_df[data_df['country_id'] == country]
    print(data_df.head())
    data_df = data_df[inflation_type] # Be carefull here!
    print(data_df.head())
    df = pd.DataFrame(data_df, dtype="f2")
    df_byte = df.to_json().encode()
    response = requests.post(f"{urlAPI}/predict",
                             params= payload,
                             files = {"test_file": df_byte})
    # response = requests.post(f"{urlAPI}/predict_test",
                            # files = {"test_file": df_byte})
    if response.ok:
        prediction = response.json()["predictions"]
        st.success("Prediction successful!")
        # Display prediction
        st.write(prediction)
    else:
        st.error("Prediction failed.")


# Try to convert all variables in df_final in type dtype="u1" and then send the file to the API
# Convert dict to dataframe in fast_api

# if st.button("Predict"):

#     file_csv = os.path.join(csv_path,'data_final.csv')
#     data_df = pd.read_csv(file_csv, index_col=0)
#     data_df = data_df[data_df['country_id'] == country]
#     data_df = data_df.to_json()
#     response = requests.post(urlAPI, files = {"csv_file": data_df})
#     print(response)

st.markdown("""# To do:
        - How to return the whole series of future forecasts and not just the first?
        - The country is hardcoded (default 'USA')
        - The target is hardcoded (default 'ccpi')
        - Number of future months forecasts is hardcoded (dafault 24)
            """)

   # st.spinner("Waiting for prediction...")

    #print(os.path.join(csv_path,'data_final.csv'))

    # uploaded_file =st.file_uploader(
        # "upload your file", type=["csv"],
        # accept_multiple_files=False)
    #print(df.info())
    #st.write(bytes_csv)
    #df_byte = df.to_json()
    #print(df_byte)
    #response = requests.post(urlAPI, files = {"csv_file": df_byte})
    #result = response.json()["predictions"]
    #print(response.json())
    #st.write("Done")
