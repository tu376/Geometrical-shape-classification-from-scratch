import numpy as np


class Conv2D:
    def __init__(self, input_shape, kernel_size, C_out, padding=0):
        C_in, H_in, W_in = input_shape

        H_out = H_in + 2 * padding - kernel_size + 1
        W_out = W_in + 2 * padding - kernel_size + 1

        self.C_in = C_in
        self.H_in = H_in
        self.W_in = W_in
        self.C_out = C_out
        self.H_out = H_out
        self.W_out = W_out
        self.kernel_size = kernel_size
        self.padding = padding

        self.output_shape = (C_out, H_out, W_out)
        self.kernel_shape = (C_out, C_in, kernel_size, kernel_size)

        fan_in = C_in * kernel_size * kernel_size
        self.kernels = np.random.randn(*self.kernel_shape) * np.sqrt(2.0 / fan_in)
        self.biases = np.zeros(C_out)

        self.d_kernels = np.zeros_like(self.kernels)
        self.d_biases = np.zeros_like(self.biases)

    # ----------------------------------------------------------

    def pad_input(self, input):
        # input: (N, C, H, W)
        if self.padding == 0:
            return input
        return np.pad(
            input,
            ((0, 0), (0, 0),
             (self.padding, self.padding),
             (self.padding, self.padding)),
            mode='constant'
        )

    # ----------------------------------------------------------

    def im2col(self, input_padded):
        # input_padded: (N, C, H_pad, W_pad)
        N  = input_padded.shape[0]
        K  = self.kernel_size
        C  = self.C_in
        Ho = self.H_out
        Wo = self.W_out

        shape = (N, C, K, K, Ho, Wo)
        strides = (
            input_padded.strides[0],
            input_padded.strides[1],
            input_padded.strides[2],
            input_padded.strides[3],
            input_padded.strides[2],
            input_padded.strides[3],
        )
        cols = np.lib.stride_tricks.as_strided(input_padded, shape=shape, strides=strides)
        # (C*K*K, N*Ho*Wo)
        return cols.transpose(1, 2, 3, 0, 4, 5).reshape(C * K * K, N * Ho * Wo)

    # ----------------------------------------------------------

    def col2im(self, d_input_col, N):
        # d_input_col: (C*K*K, N*Ho*Wo)
        K  = self.kernel_size
        Ho = self.H_out
        Wo = self.W_out
        padded_H = self.H_in + 2 * self.padding
        padded_W = self.W_in + 2 * self.padding

        d_col = d_input_col.reshape(self.C_in, K, K, N, Ho, Wo)
        d_input_padded = np.zeros((N, self.C_in, padded_H, padded_W))

        for i in range(K):
            for j in range(K):
                # d_col[:, i, j, :, :, :] shape: (C_in, N, Ho, Wo)
                d_input_padded[:, :, i:i+Ho, j:j+Wo] += d_col[:, i, j, :, :, :].transpose(1, 0, 2, 3)

        if self.padding == 0:
            return d_input_padded
        return d_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

    # ----------------------------------------------------------

    def forward(self, input):
        # input: (N, C, H, W)
        self.input = input
        N = input.shape[0]
        self.input_padded = self.pad_input(input)
        self.input_col = self.im2col(self.input_padded)   # (C*K*K, N*Ho*Wo)

        kernels_col = self.kernels.reshape(self.C_out, -1) # (C_out, C*K*K)
        output = kernels_col @ self.input_col + self.biases[:, None]  # (C_out, N*Ho*Wo)
        return output.reshape(self.C_out, N, self.H_out, self.W_out).transpose(1, 0, 2, 3)
        # returns (N, C_out, H_out, W_out)

    # ----------------------------------------------------------

    def backward(self, d_output, learning_rate):
        # d_output: (N, C_out, H_out, W_out)
        N = d_output.shape[0]
        d_output_col = d_output.transpose(1, 0, 2, 3).reshape(self.C_out, -1)  # (C_out, N*Ho*Wo)

        # dL/db — sum over N and spatial positions
        self.d_biases = d_output_col.sum(axis=1)

        # dL/dW
        d_kernels_col = d_output_col @ self.input_col.T   # (C_out, C*K*K)
        self.d_kernels = d_kernels_col.reshape(self.kernel_shape)

        # dL/dX
        kernels_col = self.kernels.reshape(self.C_out, -1)
        d_input_col = kernels_col.T @ d_output_col        # (C*K*K, N*Ho*Wo)
        d_input = self.col2im(d_input_col, N)              # (N, C, H, W)

        self.kernels -= learning_rate * self.d_kernels
        self.biases  -= learning_rate * self.d_biases

        return d_input

# import numpy as np

# class Conv2D:
#     def __init__(self, input_shape, kernel_size, C_out, padding=0):
#         C_in, H_in, W_in = input_shape

#         H_out = H_in + 2 * padding - kernel_size + 1
#         W_out = W_in + 2 * padding - kernel_size + 1

#         self.C_in = C_in
#         self.H_in = H_in
#         self.W_in = W_in
#         self.C_out = C_out
#         self.H_out = H_out
#         self.W_out = W_out
#         self.kernel_size = kernel_size
#         self.padding = padding

#         self.output_shape = (C_out, H_out, W_out)
#         self.kernel_shape = (C_out, C_in, kernel_size, kernel_size)

#         # He initialization
#         fan_in = C_in * kernel_size * kernel_size

#         self.kernels = np.random.randn(*self.kernel_shape) * np.sqrt(2.0 / fan_in)
#         self.biases = np.zeros(C_out)

#         self.d_kernels = np.zeros_like(self.kernels)
#         self.d_biases = np.zeros_like(self.biases)

#     # ----------------------------------------------------------

#     def pad_input(self, input):
#         if self.padding == 0:
#             return input

#         return np.pad(
#             input,
#             (
#                 (0, 0),
#                 (self.padding, self.padding),
#                 (self.padding, self.padding)
#             ),
#             mode='constant'
#         )

#     # ----------------------------------------------------------

#     def im2col(self, input_padded):
#         K  = self.kernel_size
#         C  = self.C_in
#         Ho = self.H_out
#         Wo = self.W_out

#         shape  = (C, K, K, Ho, Wo)
#         strides = (
#             input_padded.strides[0],
#             input_padded.strides[1],
#             input_padded.strides[2],
#             input_padded.strides[1],
#             input_padded.strides[2],
#         )

#         cols = np.lib.stride_tricks.as_strided(input_padded, shape=shape, strides=strides)
#         return cols.reshape(C * K * K, Ho * Wo)

#     # ----------------------------------------------------------

#     def col2im(self, d_input_col):
#         K  = self.kernel_size
#         Ho = self.H_out
#         Wo = self.W_out
#         padded_H = self.H_in + 2 * self.padding
#         padded_W = self.W_in + 2 * self.padding

#         d_input_padded = np.zeros((self.C_in, padded_H, padded_W))
#         d_col = d_input_col.reshape(self.C_in, K, K, Ho, Wo)

#         for i in range(K):
#             for j in range(K):
#                 d_input_padded[:, i:i+Ho, j:j+Wo] += d_col[:, i, j, :, :]

#         if self.padding == 0:
#             return d_input_padded

#         return d_input_padded[:, self.padding:-self.padding, self.padding:-self.padding]

#     # ----------------------------------------------------------

#     def forward(self, input):
#         self.input = input
#         self.input_padded = self.pad_input(input)
#         self.input_col = self.im2col(self.input_padded)

#         kernels_col = self.kernels.reshape(self.C_out, -1)
#         output = kernels_col @ self.input_col + self.biases[:, None]
#         return output.reshape(self.C_out, self.H_out, self.W_out)

#     # ----------------------------------------------------------

#     def backward(self, d_output, learning_rate):
#         d_output_col = d_output.reshape(self.C_out, -1)
#         # dL/db
#         self.d_biases = d_output_col.sum(axis=1)
#         # dL/dW
#         d_kernels_col = d_output_col @ self.input_col.T
#         self.d_kernels = d_kernels_col.reshape(self.kernel_shape)
#         # dL/dX
#         kernels_col = self.kernels.reshape(self.C_out, -1)
#         d_input_col = kernels_col.T @ d_output_col
#         d_input = self.col2im(d_input_col)

#         self.kernels -= learning_rate * self.d_kernels
#         self.biases -= learning_rate * self.d_biases

#         return d_input