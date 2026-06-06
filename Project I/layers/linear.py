import numpy as np

class Linear:

    def __init__(self, in_features, out_features):
        self.weights = (
            np.random.randn(out_features, in_features)
            * np.sqrt(2.0 / in_features)
        )
        self.biases = np.zeros(out_features)

    # ----------------------------------------------------------

    def forward(self, input):
        self.input = input
        return self.weights @ input + self.biases

    # ----------------------------------------------------------

    def backward(self, d_output, learning_rate):
        # dL/dW
        d_weights = np.outer(d_output, self.input)
        # dL/db
        d_biases = d_output.copy()
        # dL/dX
        d_input = self.weights.T @ d_output
        # update
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_input