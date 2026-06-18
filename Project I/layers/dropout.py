import numpy as np


class Dropout:
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None

    def forward(self, input, training=True):
        # input: (N, ...) — works for any shape
        if not training:
            return input
        self.mask = np.random.rand(*input.shape) > self.rate
        return input * self.mask / (1 - self.rate)   # inverted dropout

    def backward(self, d_out):
        return d_out * self.mask / (1 - self.rate)

# import numpy as np

# class Dropout:
#     def __init__(self, rate=0.5):
#         self.rate = rate
#         self.mask = None

#     def forward(self, input, training=True):
#         if not training:
#             return input
#         self.mask = np.random.rand(*input.shape) > self.rate
#         return input * self.mask / (1 - self.rate)  # inverted dropout

#     def backward(self, d_out):
#         return d_out * self.mask / (1 - self.rate)