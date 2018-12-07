import pickle
import os

def create_save_path(save_dir, model, ext):
    filename = str(model) + ext
    return os.path.join(save_dir, filename)


def save_model(save_dir, model):
    ext = ".json"
    save_path = create_save_path(save_dir, model, ext)
    model_json = model.to_json()
    with open(save_path, "w") as json_file:
        json_file.write(model_json)


def save_weights(save_dir, model):
    ext = ".h5"
    save_path = create_save_path(save_dir, model, ext)
    model.save_weights(save_path)