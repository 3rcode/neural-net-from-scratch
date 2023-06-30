import numpy as np
np.random.seed(0)


class Dense:
    def __init__(self, units, activation=None, learning_rate=0.01, is_input_layer=False):
        self.units = units
        self.activation = activation
        self.learning_rate = learning_rate
        self.W = None
        self.dW = None
        self.b = None
        self.db = None
        self.Z = None
        self.A = None
        self.X = None
        self.is_input_layer = is_input_layer

    def initialize(self, back_units):
        self.W = np.random.randn(self.units, back_units)
        self.b = np.zeros((self.units, 1))
        if not self.activation:
            self.activation = 'linear'

    def forward(self, X):
        if self.is_input_layer:
            self.X = X
            self.Z = X
            self.A = X
        else:
            self.X = X
            self.Z = np.dot(self.W, self.X) + self.b
            match self.activation:
                case 'linear':
                    self.A = self.Z
                case 'sigmoid':
                    self.A = self._sigmoid(self.Z)
                case 'tanh':
                    self.A = self._tanh(self.Z)
                case 'relu':
                    self.A = self._relu(self.Z)
                case 'leaky relu':
                    self.A = self._leaky_relu(self.Z)
        return self.A

    def backward(self, dA):
        if not self.is_input_layer:
            m = self.X.shape[1]
            dZ = dA * self._cal_grads()
            self.dW = 1 / m * np.dot(dZ, self.X.T)
            self.db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            self.W = self.W - self.learning_rate * self.dW
            self.b = self.b - self.learning_rate * self.db
            dX = np.dot(self.W.T, dZ)
            return dX

    def _cal_grads(self):
        match self.activation:
            case 'linear':
                return 1
            case 'sigmoid':
                return self.Z * (1 - self.Z)
            case 'tanh':
                return 1 - np.power(self.Z, 2)
            case 'relu':
                return np.where(self.Z >= 0, 1, 0)
            case 'leaky relu':
                return np.where(self.Z >= 0, 1, 0.01)

    @staticmethod
    def _sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def _tanh(Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    @staticmethod
    def _relu(Z):
        return np.where(Z >= 0, Z, 0)

    @staticmethod
    def _leaky_relu(Z):
        return np.where(Z >= 0, Z, Z * 0.01)