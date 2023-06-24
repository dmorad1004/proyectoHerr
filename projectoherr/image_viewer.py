import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mnist_loader import load_data

matplotlib.use("TkAgg")


def plot_image(N=1):
    """
    Grafica una imagen  del MNIST.

    Parametros:
        N(int): valor entre 0 y 59999 que indica del la imagen a graficar.
        por defecto su valor es 1


    """
    images, labels = load_data("training")

    label = labels[N]
    image = images[N]/255

    plt.imshow(np.asarray(image), cmap='gray')

    plt.colorbar()
    plt.title(str(label))

    plt.xticks(np.arange(-0.5, 28, 1), [])
    plt.yticks(np.arange(-0.5, 28, 1), [])

    plt.grid()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        plot_image(n)
    else:
        plot_image()
