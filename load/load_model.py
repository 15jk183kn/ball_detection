from keras.models import model_from_json


def load_json(json_path: object) -> object:
    with open(json_path, 'r') as jsonf_file:
        loaded_model_json = jsonf_file.read()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model


