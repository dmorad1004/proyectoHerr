import gzip

import numpy as np


def load_data(dataset="training"):
    """Carga y procesa el conjunto de datos MNIST para entrenar o probar una
    red neuronal.

    Parámetros:
        dataset (str, opcional):
        Tipo de conjunto de datos a cargar: 'training' o 'testing'.
        Por defecto es 'training'.

    Returns:
        images(numpy array): Imágenes del conjunto de datos
        labels(numpy array): Etiquedas correspondientes a las imágenes
    """

    if dataset == "training":
        images_path = "./data/train/train-images-idx3-ubyte.gz"
        labels_path = "./data/train/train-labels-idx1-ubyte.gz"
    elif dataset == "testing":
        images_path = "./data/test/t10k-images-idx3-ubyte.gz"
        labels_path = "./data/test/t10k-labels-idx1-ubyte.gz"
    else:
        raise ValueError("data set debe ser 'training' o testing'")

    with gzip.open(images_path, "rb") as images_file:
        np.frombuffer(images_file.read(4), dtype=">i4")

        number_of_images = np.frombuffer(images_file.read(4), dtype=">i4")[0]
        number_of_rows = np.frombuffer(images_file.read(4), dtype=">i4")[0]
        number_of_columns = np.frombuffer(images_file.read(4), dtype=">i4")[0]

        images = np.frombuffer(images_file.read(), dtype=np.uint8).reshape(
            number_of_images, number_of_rows, number_of_columns)

    with gzip.open(labels_path, "rb") as images_file:
        np.frombuffer(images_file.read(8), dtype=">i4")
        labels = np.frombuffer(images_file.read(), dtype=np.uint8)

    return images, labels
