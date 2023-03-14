from tensorflow.keras.models import load_model
import os

def load_models(model_country = "USA", case = "ccpi"):

    model_name = f"model_{case.lower()}_{model_country.lower()}.h5"

    current_path = os.getcwd()

    model_path = os.path.join(current_path, 'models', model_name)

    print(model_path)
    print(current_path)

    model_loaded = load_model(model_path)

    return model_loaded
