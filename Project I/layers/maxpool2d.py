import numpy as np


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.mask = None

    # ----------------------------------------------------------

    def forward(self, input):
        # input: (N, C, H, W)
        self.input = input
        N, C, H, W = input.shape
        P, S = self.pool_size, self.stride

        H_out = (H - P) // S + 1
        W_out = (W - P) // S + 1

        output = np.zeros((N, C, H_out, W_out))
        self.mask = np.zeros_like(input)

        for i in range(H_out):
            for j in range(W_out):
                h_start, w_start = i * S, j * S
                patch = input[:, :, h_start:h_start+P, w_start:w_start+P]  # (N, C, P, P)

                max_val = patch.max(axis=(2, 3))                             # (N, C)
                output[:, :, i, j] = max_val

                # build mask — True at max position
                max_mask = (patch == max_val[:, :, None, None])
                # if multiple equal maxima, keep only the first
                flat = max_mask.reshape(N, C, -1)
                first = np.zeros_like(flat)
                idx = np.argmax(flat, axis=2)
                np.put_along_axis(first, idx[:, :, None], 1, axis=2)
                self.mask[:, :, h_start:h_start+P, w_start:w_start+P] += first.reshape(N, C, P, P)

        return output  # (N, C, H_out, W_out)

    # ----------------------------------------------------------

    def backward(self, d_output):
        # d_output: (N, C, H_out, W_out)
        N, C, H_out, W_out = d_output.shape
        P, S = self.pool_size, self.stride
        d_input = np.zeros_like(self.input)

        for i in range(H_out):
            for j in range(W_out):
                h_start, w_start = i * S, j * S
                patch_mask = self.mask[:, :, h_start:h_start+P, w_start:w_start+P]
                d_input[:, :, h_start:h_start+P, w_start:w_start+P] += (
                    patch_mask * d_output[:, :, i:i+1, j:j+1]
                )

        return d_input  # (N, C, H, W)

# import numpy as np

# class MaxPool2D:
#     """
#     MaxPool2D layer with forward and backward pass.
#     """

#     def __init__(self, pool_size=2, stride=2):
#         self.pool_size = pool_size
#         self.stride = stride

#         self.input = None
#         self.mask = None

#     # ----------------------------------------------------------

#     def forward(self, input):
#         """
#         Input : (C, H, W)
#         Output: (C, H_out, W_out)
#         """

#         self.input = input

#         C, H, W = input.shape

#         H_out = (H - self.pool_size) // self.stride + 1
#         W_out = (W - self.pool_size) // self.stride + 1

#         output = np.zeros((C, H_out, W_out))

#         self.mask = np.zeros_like(input)

#         for c in range(C):

#             for i in range(H_out):

#                 for j in range(W_out):

#                     h_start = i * self.stride
#                     h_end = h_start + self.pool_size

#                     w_start = j * self.stride
#                     w_end = w_start + self.pool_size

#                     patch = input[c, h_start:h_end, w_start:w_end]

#                     max_val = np.max(patch)

#                     output[c, i, j] = max_val

#                     # argmax position
#                     max_idx = np.argmax(patch)

#                     row = max_idx // self.pool_size
#                     col = max_idx % self.pool_size

#                     self.mask[
#                         c,
#                         h_start + row,
#                         w_start + col
#                     ] = 1

#         return output

#     # ----------------------------------------------------------

#     def backward(self, d_output):
#         """
#         d_output: (C, H_out, W_out)
#         """

#         C, H_out, W_out = d_output.shape

#         d_input = np.zeros_like(self.input)

#         for c in range(C):

#             for i in range(H_out):

#                 for j in range(W_out):

#                     h_start = i * self.stride
#                     h_end = h_start + self.pool_size

#                     w_start = j * self.stride
#                     w_end = w_start + self.pool_size

#                     patch_mask = self.mask[
#                         c,
#                         h_start:h_end,
#                         w_start:w_end
#                     ]

#                     d_input[
#                         c,
#                         h_start:h_end,
#                         w_start:w_end
#                     ] += patch_mask * d_output[c, i, j]

#         return d_input