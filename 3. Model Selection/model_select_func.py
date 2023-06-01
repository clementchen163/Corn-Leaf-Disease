import os
from tensorflow.keras.models import model_from_json

def save_model(model, directory):
    model_json = model.to_json()
    with open(directory + 'model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(directory + 'model.h5')
    print('Saved model to disk')
    
def load_model(model_name):
    json_file = open('./saved_models/' + model_name + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('./saved_models/' + model_name + '/model.h5')
    print('Loaded ' + model_name + ' from disk')

    loaded_model.compile(optimizer='sgd', # Common optimizers include 'adam', 'sgd', and 'rmsprop'.
                         loss='sparse_categorical_crossentropy', # Common loss functions include 'binary_crossentropy', 'categorical_crossentropy', and 'mse'.
                         metrics=['accuracy']) #Common metrics include 'accuracy', 'precision', 'recall', and 'f1_score'.
    return loaded_model

def record_results(model_name, model, dataset):
    '''
    Records metrics for given classifier
    ---Parameters---
    model_name (str) name of model
    model (keras model) fitted classifier
    dataset (tf.data.Dataset) test dataset
    ---Returns---
    list of metrics for classifier 
    '''
    y_pred_probas = model.predict(dataset)
    y_pred_labels = np.argmax(y_pred_probas, axis = 1)
    y_true_labels = []
    for images, labels in dataset:
        y_true_labels.extend(labels.numpy())
    f1 = f1_score(y_true_labels,y_pred_labels, average='weighted')
    precision= precision_score(y_true_labels, y_pred_labels, average='weighted')
    test_acc= accuracy_score(y_true_labels, y_pred_labels)
    recall=recall_score(y_true_labels, y_pred_labels, average='weighted')
    
    #roc=roc_auc_score(y_test, y_prob)
    return [model_name, f1, test_acc, precision, recall]
    #return [model_name, f1, test_acc, roc, precision, recall]








