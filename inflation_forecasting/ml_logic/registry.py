from tensorflow.keras.models import load_model
import os

def load_models(model_country = "USA", case = "ccpi"):

    model_name = f"model_{case.lower()}_{model_country.lower()}.h5"

    root_path = os.path.dirname(os.path.dirname(__file__))

    model_path = os.path.join(root_path, 'models', model_name)

    print(model_path)

    model_loaded = load_model(model_path)

    return model_loaded
