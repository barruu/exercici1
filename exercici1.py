import numpy as np


class NeuralNetwork:
    def __init__(self):
        # Configuració de la xarxa neuronal
        self.input_size = 2  # Nombre d'entrades
        self.hidden_size = 3  # Nombre de neurones a la capa oculta
        self.output_size = 2  # Nombre de neurones a la capa de sortida

        # Inicialització dels pesos i bias
        self.input_layer_weights = np.ones((self.input_size, self.hidden_size))
        self.input_layer_bias = np.ones((1, self.hidden_size))

        self.output_layer_weights = np.ones((self.hidden_size, self.output_size))
        self.output_layer_bias = np.ones((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # Funció d'activació sigmoid

    def feedforward(self, X):
        # Capa oculta
        self.hidden_layer_input = np.dot(X, self.input_layer_weights) + self.input_layer_bias
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        # Capa de sortida
        self.output_layer_input = np.dot(self.hidden_layer_output, self.output_layer_weights) + self.output_layer_bias
        self.output_layer_output = self.sigmoid(self.output_layer_input)

        return self.output_layer_output  # Retorna la sortida de la xarxa


# Crear una instància de la xarxa neuronal
nn = NeuralNetwork()

# Entrada [0, 1]
input_data = np.array([[0, 1]])

output = nn.feedforward(input_data)

print("Sortida desitjada per a [0, 1]:", output)