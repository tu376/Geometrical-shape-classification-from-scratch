import numpy as np

from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dropout, Linear, SoftmaxCrossEntropy

SHAPES = ["circle", "ellipse", "square", "triangle", "rectangle", "hexagon", "octagon"]


class CNN:
    """
    Improved CNN for 7-class geometric shape classification (64x64 grayscale).

    Architecture changes vs. the original model:
    ─────────────────────────────────────────────
    OLD: Flatten(4096) → Dropout → Linear(4096→7) → Softmax
    NEW: Flatten(4096) → Dropout → Linear(4096→512) → ReLU
                       → Dropout → Linear(512→7)   → Softmax

    Motivation: the original single linear layer was the bottleneck.
    Classical models (RF 99.8%, SVM 99.5%) trained on the CNN's own
    Flatten-layer features outperformed the CNN (96.4%), proving that
    the feature space was already highly discriminative. Adding one
    hidden layer gives the classifier enough capacity to exploit it.

    Parameter count:
        Conv1 kernels:  8 × 1 × 3 × 3 + 8  =    80
        Conv2 kernels: 16 × 8 × 3 × 3 + 16  = 1,168
        Linear1:       4096 × 512 + 512      = 2,097,664   ← main change
        Linear2:        512 × 7   + 7        =     3,591
        ─────────────────────────────────────────────────
        Total                                ≈ 2,102,503
    """

    def __init__(self, learning_rate=0.01, dropout_rate=0.5):

        self.learning_rate = learning_rate
        self.training = True

        # ── Block 1: (1, 64, 64) → (8, 32, 32) ──────────────────────────
        self.conv1 = Conv2D(input_shape=(1, 64, 64), kernel_size=3, C_out=8,  padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)

        # ── Block 2: (8, 32, 32) → (16, 16, 16) ─────────────────────────
        self.conv2 = Conv2D(input_shape=(8, 32, 32), kernel_size=3, C_out=16, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)

        # ── MLP Classifier head ───────────────────────────────────────────
        #   Flatten: (16, 16, 16) → 4096
        #   Linear1: 4096 → 512  + ReLU + Dropout   (NEW hidden layer)
        #   Linear2:  512 → 7    + Softmax
        self.flatten  = Flatten()
        self.dropout1 = Dropout(rate=dropout_rate)   # after flatten
        self.linear1  = Linear(in_features=16 * 16 * 16, out_features=512)
        self.relu3    = ReLU()
        self.dropout2 = Dropout(rate=dropout_rate)   # after hidden layer
        self.linear2  = Linear(in_features=512, out_features=len(SHAPES))

        self.loss_fn  = SoftmaxCrossEntropy()

    # ======================================================================

    def forward(self, x, labels=None):
        """
        Parameters
        ----------
        x      : ndarray (N, 1, 64, 64)  — batch of normalised images
        labels : ndarray (N,) int | None  — class indices (training only)

        Returns
        -------
        loss  (float)               if labels is not None
        probs (N, num_classes)      otherwise
        """
        # ── Convolutional feature extractor ──────────────────────────────
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)                    # (N,  8, 32, 32)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)                    # (N, 16, 16, 16)

        # ── MLP classifier head ───────────────────────────────────────────
        x = self.flatten.forward(x)                  # (N, 4096)
        x = self.dropout1.forward(x, training=self.training)

        x = self.linear1.forward(x)                  # (N, 512)
        x = self.relu3.forward(x)
        x = self.dropout2.forward(x, training=self.training)

        logits = self.linear2.forward(x)             # (N, 7)

        # ── Loss (training) or probabilities (inference) ──────────────────
        if labels is not None:
            return self.loss_fn.forward(logits, labels)

        # Numerically stable softmax
        shifted = logits - logits.max(axis=1, keepdims=True)
        exps    = np.exp(shifted)
        return exps / exps.sum(axis=1, keepdims=True)

    # ======================================================================

    def backward(self):
        """Full backpropagation through all layers."""
        d = self.loss_fn.backward()

        # ── MLP head ──────────────────────────────────────────────────────
        d = self.linear2.backward(d,  learning_rate=self.learning_rate)
        d = self.dropout2.backward(d)
        d = self.relu3.backward(d)
        d = self.linear1.backward(d,  learning_rate=self.learning_rate)
        d = self.dropout1.backward(d)
        d = self.flatten.backward(d)

        # ── Convolutional blocks ───────────────────────────────────────────
        d = self.pool2.backward(d)
        d = self.relu2.backward(d)
        d = self.conv2.backward(d,    learning_rate=self.learning_rate)

        d = self.pool1.backward(d)
        d = self.relu1.backward(d)
        d = self.conv1.backward(d,    learning_rate=self.learning_rate)

    # ======================================================================

    def predict(self, x):
        """
        Parameters
        ----------
        x : ndarray (N, 1, 64, 64)

        Returns
        -------
        preds : ndarray (N,)              — predicted class indices
        probs : ndarray (N, num_classes)  — softmax probabilities
        """
        self.training = False
        probs = self.forward(x)
        self.training = True
        return np.argmax(probs, axis=1), probs

    # ======================================================================

    def get_features(self, x):
        """
        Extract Flatten-layer activations for use with classical ML models.

        Parameters
        ----------
        x : ndarray (N, 1, 64, 64)

        Returns
        -------
        features : ndarray (N, 4096)
        """
        self.training = False

        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        features = self.flatten.forward(x)           # (N, 4096)

        self.training = True
        return features

    # ======================================================================

    def save(self, path="weights.npy"):
        weights = {
            "conv1_kernels":   self.conv1.kernels,
            "conv1_biases":    self.conv1.biases,
            "conv2_kernels":   self.conv2.kernels,
            "conv2_biases":    self.conv2.biases,
            "linear1_weights": self.linear1.weights,
            "linear1_biases":  self.linear1.biases,
            "linear2_weights": self.linear2.weights,
            "linear2_biases":  self.linear2.biases,
        }
        np.save(path, weights)
        print(f"[CNN] Weights saved → {path}")

    # ======================================================================

    def load(self, path="weights.npy"):
        weights = np.load(path, allow_pickle=True).item()
        self.conv1.kernels    = weights["conv1_kernels"]
        self.conv1.biases     = weights["conv1_biases"]
        self.conv2.kernels    = weights["conv2_kernels"]
        self.conv2.biases     = weights["conv2_biases"]
        self.linear1.weights  = weights["linear1_weights"]
        self.linear1.biases   = weights["linear1_biases"]
        self.linear2.weights  = weights["linear2_weights"]
        self.linear2.biases   = weights["linear2_biases"]
        print(f"[CNN] Weights loaded ← {path}")