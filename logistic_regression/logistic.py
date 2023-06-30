import numpy as np
import matplotlib.pyplot as plt


class Logistic:
    def __init__(self, iteration, learning_rate=0.005):
        self.n = None  # Number of features
        self.m = None  # Number of samples
        self.W = None  # Weights
        self.b = None  # Bias
        self.iteration = iteration
        self.learning_rate = learning_rate

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _forward_prop(self, X):
        Z = np.dot(self.W.T, X) + self.b
        A = self._sigmoid(Z)
        return A

    def _back_prop(self, Y, A, X):
        cost_func = -1 / self.m * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)))
        dW = 1 / self.m * np.dot(X, (A - Y).T)
        db = 1 / self.m * np.sum(A - Y)
        cost_func = np.squeeze(cost_func)
        return cost_func, dW, db

    def fit(self, X, Y):
        assert type(X) == np.ndarray
        assert type(Y) == np.ndarray

        self.n, self.m = X.shape
        self.W = np.zeros((self.n, 1), dtype=float)
        self.b = float(0)
        cost_funcs = []
        for i in range(self.iteration):
            A = self._forward_prop(X)
            cost_func, dW, db = self._back_prop(Y, A, X)
            self.W = self.W - self.learning_rate * dW
            self.b = self.b - self.learning_rate * db
            cost_funcs.append(cost_func)
        print(cost_funcs)
        plt.plot(cost_funcs)
        plt.show()

    def predict(self, X):
        assert type(X) == np.ndarray
        output = self._forward_prop(X)
        prediction = np.where(output >= 0.5, 1, 0)
        return prediction


if __name__ == '__main__':
    model = Logistic(iteration=100)
    X_train = np.random.randn(10, 1000)
    y_train = np.random.randint(0, 2, 1000, dtype=int)
    y_train = np.reshape(y_train, (1, 1000))
    model.fit(X_train, y_train)
    X_test = np.random.randn(10, 10)
    predictions = model.predict(X_test)
    print(predictions)

