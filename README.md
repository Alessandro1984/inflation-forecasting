# Inflation forecasting with deep learning

## Final project of the 9-week bootcamp in Data Science at Le Wagon Berlin, Germany (batch-1142)

*Project members: Stefanie Mossa, Gunnar David, Joep Lamers, Alessandro Bramucci*

The goal of the project was to forecast inflation using deep learning models as an alternative to traditional time-series econometric techniques. More specifically, we have performed a univariate time-series forecasting exercise with LSTM. An LSTM model, short for "Long Short-Term Memory", is a particular type of Recurrent Neural Network used to making predictions with time-series type of data, originally developed for speech and text comprehension. A gentle introduction to this topic can be found [here](https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/).

The project was developed in only 8 days of work. During the creation of the project there have been many challenging moments beyond the pure modelling part. A big thank goes to all the teachers and TAs who helped us along the way. Here is a brief chronology of the project's development from beginning to end:
- Download of the data for inflation (core and headline inflation) for the US and some selected eurozone countries from FRED and Eurostat using the respective APIs
- Preparation of the data, setting up of the model with *tensorflow*
- Programming of the API with *fastapi*
- Creation of the dashboard with *streamlit*
- Building of the *Docker* image and pushing of the *Docker* image to Google Cloud Platform
- The app is published online. The app allows the user to select the country, the type of inflation, and the time horizon over which to produce the forecast

**DISCLAIMER:** Obviously for us it was a training exercise. No pretense of being scientific here. We are fully aware that our prediction must be taken with the grain of salt. Still, we learned a lot and that was the most important thing for us!


