import numpy as np

class SoftMaxCrossEntropy:

    def forward(self, input, output_true):

        # numerical stability
        exps = np.exp(input - np.max(input))
        self.probs = exps / np.sum(exps)
        self.output_true = output_true
        loss = -np.log(self.probs[output_true] + 1e-9)

        return loss

    # ----------------------------------------------------------

    def backward(self):
        d_output = self.probs.copy()
        d_output[self.output_true] -= 1
        return d_output