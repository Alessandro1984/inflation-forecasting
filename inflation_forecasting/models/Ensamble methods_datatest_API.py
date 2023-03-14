#!/usr/bin/env python
# coding: utf-8

# # import packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


# # load data

# In[2]:


# Check your current working directory using `os.getcwd()` below 
import os
os.getcwd()


# In[12]:


#csv_path = os.path.join('..', 'data-context-and-setup', 'data', 'csv')
csv_path = os.path.join('..','raw_data')
#pd.read_csv(os.path.join(csv_path, 'olist_sellers_dataset.csv')).head()
df = pd.read_csv(os.path.join(csv_path,'data_us.csv'))


# In[11]:


df


# In[13]:


# rename columns
df = df.rename(columns={
    "DFF": "fed_funds_rate",
    "CPIAUCSL": "CPI",
    "CPILFESL": "core_CPI",
    "UNRATE": "unemp_rate",
    "WTISPLC": "oil_price",
    "INDPRO": "index_ind_prod",
    "MABMM301USM189S": "money_supply_M3",
    "A576RC1": "wage_growth"
})

# drop "Unnamed: 0" column
df = df.drop("Unnamed: 0", axis=1)
df = df.drop('CPI',axis=1)

#set time to time 
df['Time'] = pd.to_datetime(df['Time'])
df = df.set_index('Time')
df


# In[14]:


#drop NaN
df = df.dropna(how='any')


# # baselining

# ## import relevant packages

# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ## create test-train split

# In[17]:


X = df.drop(columns='core_CPI')
y = df['core_CPI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,shuffle=False)

# Use the same function above for the validation set
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size = 0.5, shuffle=False)


X_train.shape, X_test.shape, y_train.shape, y_test.shape, y_val.shape, X_val.shape


# ## baseline model

# let's create a basic model to use as a benchmark

# In[20]:


baseline_1 = mean_squared_error(y_train, X_train["fed_funds_rate"].values)
print(f"Our first baseline give a MSE of {baseline_1}")


# # model development

# ## importing relevant packages

# In[27]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate
import seaborn as sns


# **GRADIENT BOOSTER**

# It is another iterative ensemble learning method that trains weak models sequentially to minimize the loss function using gradient descent. It combines the predictions of multiple decision trees to produce a final prediction with higher accuracy than any individual tree.

# In[28]:


model = GradientBoostingRegressor()
cv_results = cross_validate(model, X_train, y_train, scoring = "neg_mean_squared_error", cv=5)
gradient_booster_mse = cv_results['test_score'].mean()

print(f"Our random forest give a MSE of {-(gradient_booster_mse)}")


# # optimize gradient booster

# Given the gradient booster has the lowest MSE, this model is used to optimize performance

# In[29]:


# Fit the GradientBoostingRegressor to the training data
model.fit(X_train, y_train)

# Predict the inflation values for the validation data
y_pred = model.predict(X_val)

# Compute the MSE between the predicted and actual values for the validation data
mse = mean_squared_error(y_val, y_pred)

# Print the MSE score
print(f"Mean squared error on validation set: {mse:.2f}")


# In[30]:


#plot the graph
y_pred_df = pd.DataFrame(data=y_pred, index=y_val.index, columns=['Predicted'])
combined_df = pd.concat([y_val, y_pred_df], axis=1)

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=combined_df, ax=ax)
ax.set(xlabel='Time', ylabel='Value', title='Actual vs Predicted')
plt.show()


# ## optimize hyperparameters

# In[31]:


# Define the hyperparameter grid to search through
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [200,500,1000],
    'max_depth': [3, 4, 5],
    'alpha':[0.001,0.01,0.1]
}

# Create the GridSearchCV object with the GradientBoostingRegressor model and the hyperparameter grid
grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1
)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameter combination and its corresponding mean squared error
print("Best hyperparameters: ", grid_search.best_params_)
print("Best mean squared error: ", -grid_search.best_score_)


# In[32]:


# Define the best hyperparameters from the grid search
best_learning_rate = 0.1
best_n_estimators = 100
best_max_depth = 3
best_max_features = 10 # reduced number of features used for each split
best_alpha = 0.001

# Create a new GradientBoostingRegressor with the best hyperparameters
model = GradientBoostingRegressor(
    learning_rate=best_learning_rate,
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    max_features=best_max_features,
     alpha=best_alpha
)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the performance of the model using mean squared error
mse = mean_squared_error(y_val, y_pred)
print("Validation set mean squared error: ", mse)


# In[33]:


#plot the data
y_pred_df = pd.DataFrame(data=y_pred, index=y_val.index, columns=['Predicted'])
combined_df = pd.concat([y_val, y_pred_df], axis=1)

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=combined_df, ax=ax)
ax.set(xlabel='Time', ylabel='Inflation', title='Actual vs Predicted')
plt.show()


# # predict test data

# In[34]:


# Make predictions on the validation set
y_pred = model.predict(X_test)

# Evaluate the performance of the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Validation set mean squared error: ", mse)


# In[35]:


#plot the data
y_pred_df = pd.DataFrame(data=y_pred, index=y_test.index, columns=['Predicted'])
combined_df = pd.concat([y_test, y_pred_df], axis=1)

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=combined_df, ax=ax)
ax.set(xlabel='Time', ylabel='Inflation', title='Actual vs Predicted')
plt.show()


# In[36]:


np.sqrt(mse)


# # check for overfitting

# In[37]:


summ = model.predict(X_train) - y_train
summ.max()


# In[38]:


# Make predictions on the validation set
y_pred = model.predict(X_train)

# Evaluate the performance of the model using mean squared error
mse = mean_squared_error(y_train, y_pred)
print("Validation set mean squared error: ", mse)


# In[39]:


# Plot the data
y_pred_df = pd.DataFrame(data=y_pred, index=y_train.index, columns=['Predicted'])
combined_df = pd.concat([y_train, y_pred_df], axis=1)

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=combined_df, ax=ax, linestyle='-')
ax.set(xlabel='Time', ylabel='Inflation', title='Actual vs Predicted')
#ax.set_xlim(pd.Timestamp('1980-01-01'), pd.Timestamp('1981-12-31'))
plt.show()


# # Train on all data

# In[40]:


# Fit the GradientBoostingRegressor to the training data
model.fit(X, y)


# In[41]:


# Make predictions on the validation set
y_pred = model.predict(X)

# Evaluate the performance of the model using mean squared error
mse = mean_squared_error(y, y_pred)
print("All data mean squared error: ", mse)


# In[42]:


# Plot the data
y_pred_df = pd.DataFrame(data=y_pred, index=y.index, columns=['Predicted'])
combined_df = pd.concat([y, y_pred_df], axis=1)

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=combined_df, ax=ax, linestyle='-')
ax.set(xlabel='Time', ylabel='Inflation', title='Actual vs Predicted')
#ax.set_xlim(pd.Timestamp('1980-01-01'), pd.Timestamp('1981-12-31'))
plt.show()


# In[ ]:




