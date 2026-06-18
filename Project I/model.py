import numpy as np

from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dropout, Linear, SoftmaxCrossEntropy

SHAPES = ["circle", "ellipse", "square", "triangle", "rectangle", "hexagon", "octagon"]


class CNN:

    def __init__(self, learning_rate=0.01):

        self.learning_rate = learning_rate
        self.training = True

        # --------------------------------------------------

        self.conv1  = Conv2D(input_shape=(1, 64, 64), kernel_size=3, C_out=8, padding=1)
        self.relu1  = ReLU()
        self.pool1  = MaxPool2D(pool_size=2, stride=2)

        # --------------------------------------------------

        self.conv2  = Conv2D(input_shape=(8, 32, 32), kernel_size=3, C_out=16, padding=1)
        self.relu2  = ReLU()
        self.pool2  = MaxPool2D(pool_size=2, stride=2)

        # --------------------------------------------------

        self.flatten = Flatten()
        self.dropout = Dropout()
        self.linear  = Linear(in_features=16 * 16 * 16, out_features=len(SHAPES))
        self.loss_fn = SoftmaxCrossEntropy()

    # ======================================================

    def forward(self, input, labels=None):
        """
        input:  (N, C, H, W)
        labels: (N,) int array — required for training, None for inference
        """
        x = self.conv1.forward(input)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.flatten.forward(x)
        x = self.dropout.forward(x, training=self.training)
        logits = self.linear.forward(x)          # (N, num_classes)

        # --------------------------------------------------

        if labels is not None:
            loss = self.loss_fn.forward(logits, labels)
            return loss

        # --------------------------------------------------

        # inference — return softmax probs (N, num_classes)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exps  = np.exp(shifted)
        probs = exps / exps.sum(axis=1, keepdims=True)
        return probs

    # ======================================================

    def backward(self):
        d = self.loss_fn.backward()
        d = self.linear.backward(d, learning_rate=self.learning_rate)
        d = self.dropout.backward(d)
        d = self.flatten.backward(d)

        # --------------------------------------------------

        d = self.pool2.backward(d)
        d = self.relu2.backward(d)
        d = self.conv2.backward(d, learning_rate=self.learning_rate)

        # --------------------------------------------------

        d = self.pool1.backward(d)
        d = self.relu1.backward(d)
        d = self.conv1.backward(d, learning_rate=self.learning_rate)

    # ======================================================

    def predict(self, input):
        """
        input: (N, C, H, W)
        Returns: pred indices (N,), probs (N, num_classes)
        """
        probs = self.forward(input)              # (N, num_classes)
        return np.argmax(probs, axis=1), probs

    # ======================================================

    def save(self, path="weights.npy"):
        weights = {
            "conv1_kernels":  self.conv1.kernels,
            "conv1_biases":   self.conv1.biases,
            "conv2_kernels":  self.conv2.kernels,
            "conv2_biases":   self.conv2.biases,
            "linear_weights": self.linear.weights,
            "linear_biases":  self.linear.biases,
        }
        np.save(path, weights)
        print(f"Saved weights to {path}")

    # ======================================================

    def load(self, path="weights.npy"):
        weights = np.load(path, allow_pickle=True).item()
        self.conv1.kernels   = weights["conv1_kernels"]
        self.conv1.biases    = weights["conv1_biases"]
        self.conv2.kernels   = weights["conv2_kernels"]
        self.conv2.biases    = weights["conv2_biases"]
        self.linear.weights  = weights["linear_weights"]
        self.linear.biases   = weights["linear_biases"]
        print(f"Loaded weights from {path}")

# import numpy as np

# from layers import *

# SHAPES = ["circle", "ellipse", "square", "triangle", "rectangle", "hexagon", "octagon"]

# class CNN:

#     def __init__(self, learning_rate=0.01):

#         self.learning_rate = learning_rate
#         self.training = True

#         # --------------------------------------------------

#         self.conv1 = Conv2D(input_shape=(1, 64, 64), kernel_size=3, C_out=8, padding=1)
#         self.relu1 = ReLu()
#         self.pool1 = MaxPool2D(pool_size=2, stride=2) 

#         # --------------------------------------------------

#         self.conv2 = Conv2D(input_shape=(8, 32, 32), kernel_size=3, C_out=16, padding=1)
#         self.relu2 = ReLu()
#         self.pool2 = MaxPool2D(pool_size=2, stride=2)

#         # --------------------------------------------------

#         self.flatten = Flatten()
#         self.dropout = Dropout()
#         self.linear = Linear(in_features=16 * 16 * 16, out_features=7)
#         self.loss_fn = SoftMaxCrossEntropy()

#     # ======================================================

#     def forward(self, input, label=None):

#         x = self.conv1.forward(input)
#         x = self.relu1.forward(x)
#         x = self.pool1.forward(x)

#         x = self.conv2.forward(x)
#         x = self.relu2.forward(x)
#         x = self.pool2.forward(x)

#         x = self.flatten.forward(x)
#         x = self.dropout.forward(x, training=self.training)
#         logits = self.linear.forward(x)

#         # --------------------------------------------------

#         if label is not None:
#             loss = self.loss_fn.forward(logits,label)
#             return loss

#         # --------------------------------------------------

#         exps = np.exp(logits - np.max(logits))
#         probs = exps / np.sum(exps)
#         return probs

#     # ======================================================

#     def backward(self):
#         d = self.loss_fn.backward()
#         d = self.linear.backward(d, learning_rate=self.learning_rate)
#         d = self.dropout.backward(d)
#         d = self.flatten.backward(d)

#         # --------------------------------------------------

#         d = self.pool2.backward(d)
#         d = self.relu2.backward(d)
#         d = self.conv2.backward(d, learning_rate=self.learning_rate)

#         # --------------------------------------------------

#         d = self.pool1.backward(d)
#         d = self.relu1.backward(d)
#         d = self.conv1.backward(d, learning_rate=self.learning_rate)

#     # ======================================================

#     def predict(self, input):
#         probs = self.forward(input)
#         return np.argmax(probs), probs

#     # ======================================================

#     def save(self, path="weights.npy"):
#         weights = {
#             "conv1_kernels": self.conv1.kernels,
#             "conv1_biases": self.conv1.biases,
#             "conv2_kernels": self.conv2.kernels,
#             "conv2_biases": self.conv2.biases,
#             "linear_weights": self.linear.weights,
#             "linear_biases": self.linear.biases,
#         }
#         np.save(path, weights)
#         print(f"Saved weights to {path}")

#     # ======================================================
#     def load(self, path="weights.npy"):
#         weights = np.load(path, allow_pickle=True).item()
#         self.conv1.kernels = weights["conv1_kernels"]
#         self.conv1.biases = weights["conv1_biases"]
#         self.conv2.kernels = weights["conv2_kernels"]
#         self.conv2.biases = weights["conv2_biases"]
#         self.linear.weights = weights["linear_weights"]
#         self.linear.biases = weights["linear_biases"]
#         print(f"Loaded weight from {path}")