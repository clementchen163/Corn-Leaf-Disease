import os
from tensorflow.keras.models import model_from_json

def save_model(model, directory):
    model_json = model.to_json()
    with open(directory + 'model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(directory + 'model.h5')
    print('Saved model to disk')
    
def plot_hist(history, title):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
    plt.axhline(y=0.9, color='r', linestyle='-')
    plt.title('Training and Validation Accuracy - '+ title)
    plt.legend()
    plt.figure()
    return None