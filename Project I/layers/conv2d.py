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

        # He initialization
        fan_in = C_in * kernel_size * kernel_size

        self.kernels = np.random.randn(*self.kernel_shape) * np.sqrt(2.0 / fan_in)
        self.biases = np.zeros(C_out)

        self.d_kernels = np.zeros_like(self.kernels)
        self.d_biases = np.zeros_like(self.biases)

    # ----------------------------------------------------------

    def pad_input(self, input):
        if self.padding == 0:
            return input

        return np.pad(
            input,
            (
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding)
            ),
            mode='constant'
        )

    # ----------------------------------------------------------

    def im2col(self, input_padded):
        K  = self.kernel_size
        C  = self.C_in
        Ho = self.H_out
        Wo = self.W_out

        shape  = (C, K, K, Ho, Wo)
        strides = (
            input_padded.strides[0],
            input_padded.strides[1],
            input_padded.strides[2],
            input_padded.strides[1],
            input_padded.strides[2],
        )

        cols = np.lib.stride_tricks.as_strided(input_padded, shape=shape, strides=strides)
        return cols.reshape(C * K * K, Ho * Wo)

    # ----------------------------------------------------------

    def col2im(self, d_input_col):
        K  = self.kernel_size
        Ho = self.H_out
        Wo = self.W_out
        padded_H = self.H_in + 2 * self.padding
        padded_W = self.W_in + 2 * self.padding

        d_input_padded = np.zeros((self.C_in, padded_H, padded_W))
        d_col = d_input_col.reshape(self.C_in, K, K, Ho, Wo)

        for i in range(K):
            for j in range(K):
                d_input_padded[:, i:i+Ho, j:j+Wo] += d_col[:, i, j, :, :]

        if self.padding == 0:
            return d_input_padded

        return d_input_padded[:, self.padding:-self.padding, self.padding:-self.padding]

    # ----------------------------------------------------------

    def forward(self, input):
        self.input = input
        self.input_padded = self.pad_input(input)
        self.input_col = self.im2col(self.input_padded)

        kernels_col = self.kernels.reshape(self.C_out, -1)
        output = kernels_col @ self.input_col + self.biases[:, None]
        return output.reshape(self.C_out, self.H_out, self.W_out)

    # ----------------------------------------------------------

    def backward(self, d_output, learning_rate):
        d_output_col = d_output.reshape(self.C_out, -1)
        # dL/db
        self.d_biases = d_output_col.sum(axis=1)
        # dL/dW
        d_kernels_col = d_output_col @ self.input_col.T
        self.d_kernels = d_kernels_col.reshape(self.kernel_shape)
        # dL/dX
        kernels_col = self.kernels.reshape(self.C_out, -1)
        d_input_col = kernels_col.T @ d_output_col
        d_input = self.col2im(d_input_col)

        self.kernels -= learning_rate * self.d_kernels
        self.biases -= learning_rate * self.d_biases

        return d_input