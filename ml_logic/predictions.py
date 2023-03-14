import numpy as np

def future_forecasting(dataset, model, scaler_model, mb = 12, mf = 12):

    results = []

    x_input = dataset[-mb:].reshape(1, mb, 1)

    for num in range(mf):
        forecast = model.predict(x_input)
        results.append(forecast[0][0])
        x_input = np.roll(x_input, -1, axis=1)
        x_input[0][-1] = forecast
    results = scaler_model.inverse_transform(np.array(results).reshape(-1, 1)) # reshape dataset to 2D array
    results = np.array(results).reshape(-1, 1)

    return results
