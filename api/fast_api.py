import pandas as pd
from io import StringIO
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor

# To-Do: check if here more functions need to be added from our code,which we have to import (e.g. anything with modeling etc.)

app = FastAPI()
#app.state.model = #To-Do: load models --> discuss: does this make sense if we have several models?

# Optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
def upload_file(csv_file: UploadFile = File(...)):
    contents = csv_file.file.read()
    buffer = StringIO(contents.decode('utf-8'))
    df = pd.read_csv(buffer)
    print(df.info())

    #process df
    return {"Status": "ok"} #df.to_dict()

# http://127.0.0.1:8000/predict?country={country}
# @app.get("/predict")
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
