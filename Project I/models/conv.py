import numpy as np

# Forward: Z -> ReLu -> A -> Linear -> Z -> Softmax -> A -> CrossEntropy -> L(Loss)
# Backward: Z <- ReLu <- A <- Linear <- Z <- Softmax <- A <- CrossEntropy <- L(Loss)
class ReLu:
    def forward(self, z):
        self.z = z
        a = np.maximum(0, z) # ReLu = Max(0, z)
        return a
    
    def backward(self, dA):
        """
        dA = dL/da
        """
        dZ = dA.copy()
        dZ[self.z <= 0] = 0 # Let all positions which have Z <= 0 -> dZ = 0
        return dZ