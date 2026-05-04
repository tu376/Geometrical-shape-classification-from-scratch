import numpy as np

# Forward: Z -> ReLu -> A -> Linear -> Z -> Softmax -> A -> CrossEntropy -> L(Loss)
# Backward: Z <- ReLu <- A <- Linear <- Z <- Softmax <- A <- CrossEntropy <- L(Loss)
class Conv2D:

    def __init__(self, input_shape, kernel_size, C_out):
        C_in, H_in, W_in = input_shape
        C_out, H_out, W_out = C_out, H_in - kernel_size + 1, W_in - kernel_size + 1
        
        self.C_in = C_in
        self.C_out = C_out
        self.H_out = H_out
        self.W_out = W_out
        self.kernel_size = kernel_size

        self.output_shape = (C_out, H_out, W_out)
        self.kernels_shape = (C_out, C_in, kernel_size, kernel_size)

        # He Initialization
        fan_in = C_in * kernel_size * kernel_size
        self.kernels = np.random.randn(*self.kernels_shape) * np.sqrt(2.0 / fan_in)
        self.biases = np.zeros(C_out)

    def forward(self, input):
        self.input = input
        output = np.zeros(self.output_shape)

        for oc in range(self.C_out):
            for i in range(self.H_out):
                for j in range(self.W_out):
                    s = 0

                    for ic in range(self.C_in):
                        region = input[ic, i: i + self.kernel_size, j: j + self.kernel_size]
                        s += np.sum(region * self.kernels[oc, ic])

                    output[oc, i, j] = s + self.biases[oc]
        
        return output
    
    def backward(self, d_output , learning_rate):
        d_input = np.zeros_like(self.input)
        d_kernels = np.zeros_like(self.kernels)
        d_biases = np.zeros_like(self.biases)

        for oc in range(self.C_out):
            for i in range(self.H_out):
                for j in range(self.W_out):

                    # Bias Gradient
                    d_biases[oc] += d_output[oc, i, j]

                    for ic in range(self.C_in):
                        region = self.input[ic, i: i + self.kernel_size, j: j + self.kernel_size]

                        # Kernel Gradient
                        d_kernels[oc, ic] += d_output[oc, i, j] * region

                        # Input Gradient
                        d_input[ic, i: i + self.kernel_size, j: j + self.kernel_size] += d_output[oc, i, j] * self.kernels[oc,ic]

        # Update
        self.kernels -= learning_rate * d_kernels
        self.biases -= learning_rate * d_biases

        return d_input

class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        C_in, H_in, W_in = input.shape

        self.H_out = (H_in - self.kernel_size) // self.stride + 1
        self.W_out = (W_in - self.kernel_size) // self.stride + 1

        output = np.zeros((C_in, self.H_out, self.W_out))

        # lưu vị trí max để backward
        self.max_pos = {}

        for ic in range(C_in):
            for i in range(self.H_out):
                for j in range(self.W_out):
                    h_start = i * self.stride
                    w_start = j * self.stride

                    region = input[ic, h_start: h_start + self.kernel_size, w_start: w_start + self.kernel_size]

                    max_val = np.max(region)
                    output[ic, i, j] = max_val 

                    # lưu vị trí max (tọa độ thật trong input)
                    idx = np.unravel_index(np.argmax(region), region.shape)
                    self.max_pos[(ic, i, j)] = (h_start + idx[0], w_start + idx[1])

        return output

    def backward(self, d_output):
        # d_output shape: (C, H_out, W_out)
        d_input = np.zeros_like(self.input)

        for (ic, i, j), (h, w) in self.max_pos.items():
            d_input[ic, h, w] += d_output[ic, i, j]

        return d_input

class ReLu:
    def forward(self, input):
        self.input = input
        output = np.maximum(0, input) # ReLu = Max(0, input)
        return output
    
    def backward(self, d_output):
        d_input = d_output.copy()
        d_input[self.input <= 0] = 0 # Let all positions which have input <= 0 -> d_input = 0
        return d_input
    
class Flatten:
    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(-1)
    
    def backward(self, d_out):
        return d_out.reshape(self.input_shape)
    
class Linear:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.biases = np.zeros(out_features)

    def forward(self, input):
        self.input = input
        return self.weights @ input + self.biases
    
    def backward(self, d_output, learning_rate):
        d_weights = np.outer(d_output, self.input)
        d_biases = d_output.copy()
        d_input = self.weights.T @ d_output


        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_input

class SoftMaxCrossEntropy:
    def forward(self, input, output_true):
        exps = np.exp(input - np.max(input))
        self.probs = exps / np.sum(exps)
        self.output_true = output_true

        loss = - np.log(self.probs[output_true] + 1e-9)
        return loss
    
    def backward(self):
        d_output = self.probs.copy()
        d_output[self.output_true] -= 1
        return d_output