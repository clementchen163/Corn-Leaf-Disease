import os
from tensorflow.keras.models import model_from_json

def save_model(model, directory):
    model_json = model.to_json()
    with open(directory + 'model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(directory + 'model.h5')
    print('Saved model to disk')