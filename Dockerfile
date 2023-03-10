#FROM Importing
FROM python:3.10.6-buster
#FROM DummyClassifier
#FROM from sklearn.dummy import DummyClassifier
#FROM tensorflow/tensorflow:2.10.0

#COPY files
#COPY app /app
COPY requirements.txt /requirements.txt
COPY model.py /model.py
COPY raw_data/data_final.csv raw_data/data_final.csv
COPY api /api
COPY .ENV /ENV


# RUN pip upgrade and install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN mkdir /saved_predictions
RUN python model.py

#CMD launch API web server
CMD uvicorn api.fast_api:app --host 0.0.0.0

#Stuff not needed now
#COPY setup.py /setup.py
#COPY Makefile /Makefile
#COPY .env /.env
#COPY .gitignore /.gitignore
