from network_base.dense_layer import Dense
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)


class NN:
    def __init__(self):
        self.layers = []
        self.loss_func = None

    def add(self, layer):
        self.layers.append(layer)

    def build(self, loss_func):
        self.loss_func = loss_func
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.is_input_layer = True
            else:
                layer.initialize(self.layers[i - 1].units)

    def summary(self):
        for i, layer in enumerate(self.layers):
            if i > 0:
                print(f'------------------------------Layer {i}------------------------------')
                print('Shape of weight:', layer.W.shape)
                print('Shape of bias:', layer.b.shape)

    def fit(self, X_train, y_train):
        nx, m = X_train.shape
        if nx != self.layers[0].units:
            raise AttributeError('The data shape is not match!')
        costs = []
        for _ in range(20):
            A_temp = X_train
            for layer in self.layers:
                A_temp = layer.forward(A_temp)
            AL = A_temp
            dAL = None
            cost = None
            match self.loss_func:
                case 'binary_cross_entropy':
                    cost = -1 / m * np.sum(y_train * np.log(AL) + (1 - y_train) * np.log(1 - AL), axis=1, keepdims=True)
                    dAL = -(y_train / AL) - ((1 - y_train) / (1 - AL))
                case 'mean_squared_error':
                    cost = 1 / m * np.sum(1.0 / 2 * np.power(y_train - AL, 2))
                    dAL = y_train - AL
            dA_temp = dAL
            costs.append(np.squeeze(cost))
            for layer in reversed(self.layers):
                dA_temp = layer.backward(dA_temp)
            print(cost)
        plt.plot(costs)
        plt.show()

    def predict(self, X):
        nx, m = X.shape
        if nx != self.layers[0].units:
            raise AttributeError('The data shape is not match!')
        A_temp = X
        for layer in self.layers:
            A_temp = layer.forward(A_temp)
        AL = A_temp
        return AL
