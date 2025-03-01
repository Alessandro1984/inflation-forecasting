import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns
import glob
from params import *
#import time


##if function needed to select the right get data function?
def get_data(country):
    '''
    Gets the data and converts it to a panda data frame. Does some first optimization for further work with the data
    '''
    # Check your current working directory using `os.getcwd()` below
    os.getcwd()

    #csv_path = os.path.join('..', 'data-context-and-setup', 'data', 'csv')
    csv_path = os.path.join('..','raw_data')

    #pd.read_csv(os.path.join(csv_path, 'olist_sellers_dataset.csv')).head()
    df = pd.read_csv(os.path.join(csv_path,'data_final.csv'))

    # drop cpi column
    df = df.drop('cpi',axis=1)

    #set time to time
    df['year'] = pd.to_datetime(df['year'])
    df = df.set_index('year')

    #drop NaN
    df = df.dropna(how='any')

    df_country = df[df['country'] == country]

    return df_country

def defining_variables(df_country):
    '''
    Defines the variables for the  the already cleaned data
    '''
    X = df_country.drop(columns='core_CPI')
    y = df_country['core_CPI']

    return X, y

def initialize_model():
    '''
    Initializes the model
    '''
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

    return model

def train_model(model, X, y):
    '''
    Trains the compiled model
    '''

    # Fit the GradientBoostingRegressor to the training data
    trained_model = model.fit(X, y)
    return trained_model


def predict_model(trained_model, X):
    '''
    Predicts for the trained model
    '''
    # Make predictions on the validation set
    y_pred = trained_model.predict(X)
    return y_pred

def save_y_pred(y_pred, country):
    '''
    Persist predicted targets locally
    '''
    #timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save y_pred locally resp. in the container's filesystem
    #@TA; how to get the filepath in the docker container
    saved_prediction_path =  LOCAL_FILE_PATH
    y_pred = y_pred.save(saved_prediction_path)


def load_y_pred(country):
    '''
    Return saved predicted y_pred
    '''
    local_prediction_directory = LOCAL_FILE_PATH
    #local_prediction_path = glob.glob(f"{local_prediction_directory}/*")
    with open(local_prediction_directory/f"{country}.h5", "r") as function:
        y_pred = function.read()
    return y_pred

    #most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
    #print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)
    #lastest_model = keras.models.load_model(most_recent_model_path_on_disk)

if __name__ == "__main__":
    df = get_data()
    defining_variables(df)
    model = initialize_model()
    train_model()
    predict_model()
    save_y_pred()
    load_y_pred()
