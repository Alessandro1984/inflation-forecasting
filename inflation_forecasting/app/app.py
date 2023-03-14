import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
import requests

# from sessionstate import SessionState # import SessionState
# https://stackoverflow.com/questions/63988485/modulenotfounderror-no-module-named-sessionstate

#urlAPI = "http://127.0.0.1:8000/predict"
#urlAPItest = "http://127.0.0.1:8000/test"
urlAPI = "http://127.0.0.1:8000"

st.markdown("""# use deep learning to predict inflation!
## use the predict button below""")
#
#the relative path to the data
root_path = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(root_path, 'raw_data')
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
    num_months = st.slider('Number of Months to Predict', 1, 24)

if inflation_type == 'Headline Inflation':
    inflation_type = 'cpi'
else:
    inflation_type = 'ccpi'

df_country = df[df['country_id'] == country]

#if st.button("predict"):

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
prediction = response.json()["predictions"]
# response = requests.post(f"{urlAPI}/predict_test",
                        # files = {"test_file": df_byte})
# if response.ok:
#     prediction = response.json()["predictions"]
#     st.success("Prediction successful!")
#     # Display prediction
#     st.write(prediction)
# else:
#     st.error("Prediction failed.")

# PLOT
fig = go.Figure()

df_short = df_country.loc[df_country["year"] >= "2015-01-01"]
first_date = df_short['year'].min()
last_date = pd.to_datetime(df_short['year'].max()) + pd.offsets.MonthBegin(1)
#country_name = df_short["country"].unique()[0]
t = pd.date_range(last_date, periods = num_months, freq='MS')
forecast_date = t.max()
out_forecast_df = pd.DataFrame([[x, y] for x, y in zip(t, np.array(prediction).reshape(-1, 1))], columns=["year", "Forecast"])
out_forecast_df['Forecast'] = out_forecast_df['Forecast'].apply(lambda x: np.ravel(x)[0])

fig.add_trace(go.Scatter(x = df_short['year'],
                         y = df_short[inflation_type],
                         line_color = "blue",
                         name = "Inflation",
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
        title = f"Out-of-sample forecast until {forecast_date.strftime('%B %Y')}",
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

st.plotly_chart(fig,
                use_container_width=True)
