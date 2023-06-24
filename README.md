# Red Neuronal con el Algoritmo de Metropolis Monte Carlo de Cero Temperatura

Este proyecto implementa un modelo de una red neuronal utilizando el algoritmo
de metrópolis Monte carlo de cero temperatura para su entrenamiento. El
propósito de este proyecto es tener un primer acercamiento a la inteligencia
artificial y buscar conexiones con la física. Para el entrenamiento de la red se
hace uso del la base datos [MNIST](http://yann.lecun.com/exdb/mnist/), que
consta de números de 0 a 10 escritos a mano.

### Contenido

- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

# Requisitos

Para asegurar la reproducibilidad del código, se hace uso de Poetry para crear
un entorno virtual. Esto no es obligatorio, pero es lo recomendable. Para la
instalación de poetry [ver aqui.](https://python-poetry.org/docs/) Sin embargo
estas son las librerias que usaron para el proyecto:

- Numpy
- Matpotlib
- tkinter
- Libreria estandar de python

# Instalación

1. Clonar el repositorio:

```
git clone https://github.com/dmorad/proyecto.git
```

2. Navegar a la carpeta del proyecto:

```
cd proyecto
```

3. Instalar las dependencias necesarias para el proyecto:

```
poetry install
```

# Uso

#### Para visualizar las imágenes del conjunto de datos:

```
poetry run python image_viewer n
```

Alternativamente:

```
python image_viewer n
```

Donde **n** es el indice de la imagen del conjunto de datos que se desea
visualizar. El índice debe ser un número entre 1 y 59999. Por ejemplo para
visualizar la imagen en el índice 31415:

```
poetry run python image_viewer 31415
```

#### Entrenamiento de la red

Por defecto se hacen 25000 iteraciones, para lograr mejores resultados se debe
iteras muchas más veces. El coste computacional es bastante alto, por ello se
recomienda hacerlo en pasos de 25000 hasta lograr la precisión deseada.

```
poetry run python train.py
```

# Licencia

Dudo que alguien esté leyendo esto o haya llegado hasta aquí, pero el código lo
pueden usar como deseen, tan solo darme atribución. No es mucho, pero es trabajo
honesto.

Cualquier duda este es mi email: dmorad@unal.edu.co
