from dbnomics import fetch_series
import os

series = []

countries = ['DE', 'ES', 'FR', 'IT', 'NL', 'US']

for country in countries:
    series.append('Eurostat/prc_hicp_manr/M.RCH_A.CP00.' + country)

df = fetch_series(series)

df = df[['period', 'value', 'geo']]

df.rename({'period': 'Time',
           'value': 'HCPI',
           'geo': 'Country'},
          axis = 1,
          inplace = True)

path = os.path.dirname(__file__)

file_path = os.path.join(path, "df_HCPI.csv")

df.to_csv(file_path, index = False)
