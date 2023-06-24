
import time

import numpy as np
from mnist_loader import load_data
from network import NeuralNetwork


def preproccess_data(dataset):
    """ Carga y preprocesa el conjunto de datos MNIST para el entrenamiento de
    la red neuronal.

    Las imágenes se normalizan al rango 0-1 y se remodelan de matrices 2D de
    28x28 a vectores 1D de longitud 784.
    Las etiquetas(labels) se convierten a representación one-hot.

    Parámetros:
        dataset (str, opcional):
        Tipo de conjunto de datos a cargar: 'training' o 'testing'.
        Por defecto es 'training'.

    Returns:
        images(numpy array): Imágenes normalizadas y remodeladas del
        conjunto de datos.
        labels(numpy array): Etiquetas en representación one-hot
        correspondientes a las imágenes.
    """
    images, labels = load_data(dataset)

    images = images.reshape(-1, 28*28)/255

    labels = np.eye(10)[labels]

    return images, labels


images_training, labels_training = preproccess_data("training")
images_tests, labels_tests = preproccess_data("testing")


def calculate_accuracy():
    """
    Calcula la  precisión de la red neuronal después del entrenamiento.

    Returns:
        precisión de la red en porcentaje. 
    """
    predictions = network.predict(images_tests)
    predicted_labels = np.argmax(predictions, axis=1)

    true_labels = np.argmax(labels_tests, axis=1)
    correct = np.sum(predicted_labels == true_labels)

    return correct/len(labels_tests) * 100


#
network = NeuralNetwork()
start_time = time.time()

network.load_model("./model/model.pkl")
network.train(images_training, labels_training, 25000)
network.save_model("./model/model.pkl")

end_time = time.time()
print(calculate_accuracy())
print(f"training time {end_time-start_time}")
