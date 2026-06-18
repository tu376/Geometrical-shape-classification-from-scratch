import numpy as np


class SoftmaxCrossEntropy:
    def forward(self, input, output_true):
        # input:       (N, C) — raw logits
        # output_true: (N,)   — integer class indices
        shifted = input - input.max(axis=1, keepdims=True)  # numerical stability
        exps = np.exp(shifted)
        self.probs = exps / exps.sum(axis=1, keepdims=True)  # (N, C)
        self.output_true = output_true

        N = input.shape[0]
        correct_log_probs = -np.log(self.probs[np.arange(N), output_true] + 1e-9)
        loss = correct_log_probs.mean()   # scalar — mean loss over batch
        return loss

    # ----------------------------------------------------------

    def backward(self):
        N = self.probs.shape[0]
        d_output = self.probs.copy()                   # (N, C)
        d_output[np.arange(N), self.output_true] -= 1
        d_output /= N                                  # average over batch
        return d_output

# import numpy as np

# class SoftMaxCrossEntropy:

#     def forward(self, input, output_true):

#         # numerical stability
#         exps = np.exp(input - np.max(input))
#         self.probs = exps / np.sum(exps)
#         self.output_true = output_true
#         loss = -np.log(self.probs[output_true] + 1e-9)

#         return loss

#     # ----------------------------------------------------------

#     def backward(self):
#         d_output = self.probs.copy()
#         d_output[self.output_true] -= 1
#         return d_output