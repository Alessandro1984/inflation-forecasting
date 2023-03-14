from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os

def filter(data, country = "USA", case="ccpi"):
    data_country = data[data['country_id'] == country]
    return data_country[case]

def scaling(data):
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler.fit(data.to_numpy().reshape(-1,1))
    return scaler

def preprocess(data_inflation, scalar):
    ccpi = data_inflation.to_numpy().reshape(-1,1)
    ccpi_trans = scalar.transform(ccpi)
    return ccpi_trans
