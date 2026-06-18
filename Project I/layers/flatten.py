import numpy as np


class Flatten:
    def forward(self, input):
        # input: (N, C, H, W) → (N, C*H*W)
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.input_shape)

# import numpy as np

# class Flatten:
#     def forward(self, input):
#         self.input_shape = input.shape
#         return input.reshape(-1)

#     # ----------------------------------------------------------

#     def backward(self, d_out):
#         return d_out.reshape(self.input_shape)