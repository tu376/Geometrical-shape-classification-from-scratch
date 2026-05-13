import numpy as np
from .layers import Conv2D, MaxPool2D, ReLu, Flatten, Linear, SoftMaxCrossEntropy

# Kiến trúc: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> Linear
# Input: (1, 64, 64) - ảnh grayscale 64x64
# Conv1(8 filters, 3x3) -> (8, 62, 62)
# Pool1(2x2) -> (8, 31, 31)
# Conv2(16 filters, 3x3) -> (16, 29, 29)
# Pool2(2x2) -> (16, 14, 14)
# Flatten -> 3136
# Linear -> 7 (số lớp hình học)

SHAPES = ["circle", "ellipse", "square", "triangle", "rectangle", "hexagon", "octagon"]

class CNN:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

        self.conv1 = Conv2D(input_shape=(1, 64, 64), kernel_size=3, C_out=8)
        self.relu1 = ReLu()
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = Conv2D(input_shape=(8, 31, 31), kernel_size=3, C_out=16)
        self.relu2 = ReLu()
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)

        self.flatten = Flatten()
        self.linear = Linear(in_features=16 * 14 * 14, out_features=7)

        self.loss_fn = SoftMaxCrossEntropy()

    def forward(self, input, label=None):
        x = self.conv1.forward(input)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.flatten.forward(x)
        x = self.linear.forward(x)

        if label is not None:
            loss = self.loss_fn.forward(x, label)
            return loss
        
        # Nếu không có label -> trả về xác suất (dùng khi predict)
        exps = np.exp(x - np.max(x))
        probs = exps / np.sum(exps)
        return probs

    def backward(self):
        d = self.loss_fn.backward()
        d = self.linear.backward(d, self.learning_rate)
        d = self.flatten.backward(d)

        d = self.pool2.backward(d)
        d = self.relu2.backward(d)
        d = self.conv2.backward(d, self.learning_rate)

        d = self.pool1.backward(d)
        d = self.relu1.backward(d)
        d = self.conv1.backward(d, self.learning_rate)

    def predict(self, input):
        probs = self.forward(input)
        return np.argmax(probs), probs

    def save(self, path="weights.npy"):
        weights = {
            "conv1_kernels": self.conv1.kernels,
            "conv1_biases": self.conv1.biases,
            "conv2_kernels": self.conv2.kernels,
            "conv2_biases": self.conv2.biases,
            "linear_weights": self.linear.weights,
            "linear_biases": self.linear.biases,
        }
        np.save(path, weights)
        print(f"Đã lưu weights vào {path}")

    def load(self, path="weights.npy"):
        weights = np.load(path, allow_pickle=True).item()
        self.conv1.kernels = weights["conv1_kernels"]
        self.conv1.biases = weights["conv1_biases"]
        self.conv2.kernels = weights["conv2_kernels"]
        self.conv2.biases = weights["conv2_biases"]
        self.linear.weights = weights["linear_weights"]
        self.linear.biases = weights["linear_biases"]
        print(f"Đã load weights từ {path}")