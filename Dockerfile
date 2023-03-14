#FROM Importing
FROM python:3.10.6-slim-buster
#FROM DummyClassifier
#FROM from sklearn.dummy import DummyClassifier
#FROM tensorflow/tensorflow:2.10.0

#COPY files
COPY inflation_forecasting /inflation_forecasting
COPY requirements_working.txt /requirements_working.txt
#COPY .ENV /ENV

# RUN pip upgrade and install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements_working.txt

#CMD launch API web server
CMD uvicorn inflation_forecasting.api.fast_api:app --host 0.0.0.0 --port $PORT

#Stuff not needed now
#COPY setup.py /setup.py
#COPY Makefile /Makefile
#COPY .env /.env
#COPY .gitignore /.gitignore
