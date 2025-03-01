#!/usr/bin/env python
# coding: utf-8

# # Load packages

# In[4]:


from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


# # Load data

# In[5]:


os.getcwd()

csv_path = os.path.join('..', 'inflation-forecasting', 'raw_data')

df = pd.read_csv(os.path.join(csv_path,'data_final.csv'), index_col=0)


# # Select data for the US

# In[6]:


df_us = df[df['country_id'] == "USA"]


# In[7]:


ccpi = pd.DataFrame(df_us['ccpi'])


# In[8]:


ccpi = ccpi.to_numpy()


# # Load the model for the US

# In[9]:


model = load_model("model_usa.h5")
model.summary()


# # Preprocess the data

# In[13]:


scaler = MinMaxScaler(feature_range = (0, 1))
dataset = scaler.fit_transform(ccpi.reshape(-1,1))


# # Split X and Y into sequences

# In[14]:


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# # Define steps, features and split data

# In[15]:


n_steps_in = 12

n_features = 1

data = dataset[0:]

X, Y = split_sequence(data, n_steps_in)


# # Predict and inverse-transform the data

# In[16]:


forecast = model.predict(X)

Y_hat = scaler.inverse_transform(forecast)

Y_actual = scaler.inverse_transform(Y)


# # Plot the data

# In[17]:


data1 = pd.DataFrame(forecast)
data2 = pd.DataFrame(Y)


# In[18]:


plt.plot(data1)
plt.plot(data2);


# # Define the forecasting function

# In[19]:


def future_forecasting(dataset, model, mb=12, mf=12):
    '''
    Returns the future forecasting of the model. Please select the dataset and model you want to use.
    for mb, select the number of months you want to look back to make a prediction. 
    for mf, select the number of months you want the prediction to look forward.
    '''
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


# # Make the forecast for future months (mf)

# In[20]:


ccpi = ccpi.to_numpy()
scaler = MinMaxScaler(feature_range = (0, 1))
dataset = scaler.fit_transform(ccpi.reshape(-1,1))

inf_for_us = future_forecasting(dataset = dataset, 
                                model = model, 
                                mb = 12, 
                                mf = 12)


# # Plot of the forecast

# In[22]:


pd.DataFrame(inf_for).plot();


# # Preprocessing and fit function

# In[3]:


def preproc_fit_function(data_country):
    ccpi = data_country["ccpi"].to_numpy()
    scaler = MinMaxScaler(feature_range = (0, 1))
    dataset = scaler.fit_transform(ccpi.reshape(-1,1))
    return dataset

