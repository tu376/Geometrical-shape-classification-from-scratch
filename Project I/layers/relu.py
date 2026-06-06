import numpy as np

class ReLu:

    def forward(self, input):
        self.input = input
        output = np.maximum(0, input)
        return output

    # ----------------------------------------------------------

    def backward(self, d_output):
        d_input = d_output.copy()
        d_input[self.input <= 0] = 0
        return d_input