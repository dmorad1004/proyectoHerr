
import pickle

import numpy as np


class NeuralNetwork:

    """
    Red neuronal con 1 capa de entrada de 784 neuras, 1 capa de salida de 10 neuronas
    y dos capas ocultas de 14 neuronas cada una.

    """

    def __init__(self):
        #
        # Se inicializan los pesos para las capas ocultas y la capa de salida
        self.weight_1 = np.random.normal(loc=0.0, scale=0.7, size=(784, 14))
        self.weight_2 = np.random.normal(loc=0.0, scale=0.7, size=(14, 14))
        self.weight_3 = np.random.normal(loc=0.0, scale=0.7, size=(14, 10))

        # Se inicializan los sesgos para las capas ocultas y la capa de salida

        self.bias_1 = np.zeros((1, 14))
        self.bias_2 = np.zeros((1, 14))
        self.bias_3 = np.zeros((1, 10))

    @staticmethod
    def sigmoid_activation(x):
        """ Función de activación sigmoide."""
        return 1/(1+np.exp(-x))

    @staticmethod
    def mse_loss_function(y_real, y_predicted):
        """ Función de pérdida, calcula el error entre el  valor predicho por la
        red y el valor real
        """
        return np.mean((y_real-y_predicted)**2)

    def predict(self, data):
        hidden_layer1_output = self.sigmoid_activation(
            np.dot(data, self.weight_1) + self.bias_1)
        hidden_layer2_output = self.sigmoid_activation(
            np.dot(hidden_layer1_output, self.weight_2) + self.bias_2)
        output_layer_output = self.sigmoid_activation(
            np.dot(hidden_layer2_output, self.weight_3) + self.bias_3)

        return output_layer_output

    def __calculate_energy(self, data, real_values):
        predicted_data = self.predict(data)
        return self.mse_loss_function(real_values, predicted_data)

    def __zero_temperature_metropolis(self, data, real_values):
        current_energy = self.__calculate_energy(data, real_values)
        new_weight_1 = self.weight_1 + \
            np.random.normal(loc=0.0, scale=0.05, size=self.weight_1.shape)
        new_weight_2 = self.weight_2 + \
            np.random.normal(loc=0.0, scale=0.05, size=self.weight_2.shape)
        new_weight_3 = self.weight_3 + \
            np.random.normal(loc=0.0, scale=0.05, size=self.weight_3.shape)

        new_bias_1 = self.bias_1 + \
            np.random.normal(loc=0.0, scale=0.1, size=self.bias_1.shape)
        new_bias_2 = self.bias_2 + \
            np.random.normal(loc=0.0, scale=0.1, size=self.bias_2.shape)
        new_bias_3 = self.bias_3 + \
            np.random.normal(loc=0.0, scale=0.1, size=self.bias_3.shape)

        original_weights_biases = (
            self.weight_1, self.weight_2, self.weight_3, self.bias_1, self.bias_2, self.bias_3)

        self.weight_1 = new_weight_1
        self.weight_2 = new_weight_2
        self.weight_3 = new_weight_3

        self.bias_1 = new_bias_1
        self.bias_2 = new_bias_2
        self.bias_3 = new_bias_3

        new_state_energy = self.__calculate_energy(data, real_values)

        if (new_state_energy < current_energy):
            pass
        else:
            self.weight_1, self.weight_2, self.weight_3, self.bias_1, self.bias_2, self.bias_3 = original_weights_biases

    def train(self, data, real_values, steps):
        for _ in range(steps):
            self.__zero_temperature_metropolis(data, real_values)

        print("trainging finished")

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'weight_1': self.weight_1,
                'weight_2': self.weight_2,
                'weight_3': self.weight_3,
                'bias_1': self.bias_1,
                'bias_2': self.bias_2,
                'bias_3': self.bias_3
            }, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.weight_1 = data['weight_1']
            self.weight_2 = data['weight_2']
            self.weight_3 = data['weight_3']
            self.bias_1 = data['bias_1']
            self.bias_2 = data['bias_2']
            self.bias_3 = data['bias_3']
