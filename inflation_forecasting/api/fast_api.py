import pandas as pd
from io import StringIO
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
import json

from pydantic import BaseModel

from inflation_forecasting.ml_logic.preprocessor import scaling, preprocess, filter
from inflation_forecasting.ml_logic.registry import load_models
from inflation_forecasting.ml_logic.predictions import future_forecasting

# To-Do: check if here more functions need to be added from our code,which we have to import (e.g. anything with modeling etc.)

class FilteringData(BaseModel):
    country: str
    inflation_type: str
    num_months: int

app = FastAPI()

# @app.post("/model")
# def load_model(filteringData: FilteringData):
#     app.state.model = load_models(
#         model_country=filteringData.country,
#         case=filteringData.inflation_type)
#     return app.state.model
#app.state.model = #To-Do: load models --> discuss: does this make sense if we have several models?

# Optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# @app.post("/item")
# def load_model(filteringData: FilteringData):
#     model = load_models(
#         model_country=filteringData.country,
#         case=filteringData.inflation_type)
#     return filteringData

# http://127.0.0.1:8000/predict?country={country}
# @app.post("/predict_test")
# def predict(test_file: UploadFile = File(...)):
#     contents = test_file.file.read()
#     #print(contents)
#     buffer = StringIO(contents.decode('utf-8'))
#     #print(buffer)
#     buffer_str = buffer.getvalue()  # Convert the StringIO object to a string
#     df = json.loads(buffer_str)
#     print(df)
#     # df = pd.DataFrame.from_dict(df['ccpi'], orient='index') # Change name of variable
#     df = pd.DataFrame.from_dict(df, orient='index') # Change name of variable
#     #data_filtered = filter(df)
#     scaler_current = scaling(df)
#     data_preproc = preprocess(df, scaler_current)
#     model_country = load_models()
#     predictions = future_forecasting(dataset = data_preproc,
#                                      model = model_country,
#                                      scaler_model = scaler_current,
#                                      mb = 12,
#                                      mf = 24)
#     print(predictions)
#     result =  [float(value) for value in predictions]
#     print(result)
#     return {"predictions": result}

@app.post("/predict")
def predict(filteringData: FilteringData = Depends(),
            test_file: UploadFile = File(...)):
    print(filteringData.num_months)
    contents = test_file.file.read()
    #print(contents)
    buffer = StringIO(contents.decode('utf-8'))
    #print(buffer)
    buffer_str = buffer.getvalue()  # Convert the StringIO object to a string
    df = json.loads(buffer_str)
    # df = pd.DataFrame.from_dict(df['ccpi'], orient='index') # Change name of variable
    df = pd.DataFrame.from_dict(df, orient='index') # Change name of variable
    #data_filtered = filter(df)
    scaler_current = scaling(df)
    data_preproc = preprocess(df, scaler_current)
    # model_country = load_models()
    model_country = load_models(
        model_country=filteringData.country,
        case=filteringData.inflation_type)
    predictions = future_forecasting(dataset = data_preproc,
                                     model = model_country,
                                     scaler_model = scaler_current,
                                     mb = 12,
                                     mf = filteringData.num_months)
    result =  [float(value) for value in predictions]
    print(result)
    return {"predictions": result}


# @app.post("/predict")
# async def predict(filteringData: FilteringData = Depends(),
#               test_file: UploadFile = File(...)):
#     print(filteringData)
#     contents = test_file.file.read()
#     buffer = StringIO(contents.decode('utf-8'))
#     buffer_str = buffer.getvalue()  # Convert the StringIO object to a string
#     df = json.loads(buffer_str)
#     #df = pd.DataFrame.from_dict(df['ccpi'], orient='index') # Change name of variable
#     df = pd.DataFrame.from_dict(df, orient='index') # Change name of variable
#     #data_filtered = filter(df)
#     scaler_current = scaling(df)
#     data_preproc = preprocess(df, scaler_current)
#     model_country = load_models()
#     predictions = future_forecasting(dataset = data_preproc,
#                                      model = model_country,
#                                      scaler_model = scaler_current,
#                                      mb = 12,
#                                      mf = 24)

#     return {"predictions": float(predictions[0][0])}

    #print(df)
    #return {"test": "ok"}

# def predict():
#     ###Using a dummy model from sklearn###

#     # Split the dataset into training and testing sets
#     X_train_preprocessed, X_test_preprocessed, y_train_preprocessed, y_test_preprocessed = train_test_split(X_preprocessed, y_preprocessed, test_size=0.2, random_state=42)

#     # Create a dummy regressor that predicts the mean target value
#     dummy_regr = DummyRegressor(strategy='mean')

#     # Fit the dummy regressor on the training data
#     dummy_regr.fit(X_train_preprocessed, y_train_preprocessed)

#     # Use the dummy regressor to make predictions on new data
#     y_pred = dummy_regr.predict(X_test_preprocessed)
#     return y_pred

#return {"{X_test_preprocessed}": "{y_pred}"}
@app.get("/")
def root():
    return {"Status": "ok"}
