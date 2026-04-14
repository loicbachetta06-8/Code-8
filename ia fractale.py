"""
Fractal neural network based on the Code 8 structure (simplified SU(3)-equivariant layer).
Author: Loïc Bachetta
License: Creative Commons Attribution 4.0 International
"""

import numpy as np

# Gell-Mann matrices (generators of SU(3)) as 3x3 complex matrices.
def gell_mann_matrices():
    """Returns the 8 Gell-Mann 3x3 matrices."""
    l = []
    # lambda_1
    l.append(np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex))
    # lambda_2
    l.append(np.array([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=complex))
    # lambda_3
    l.append(np.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=complex))
    # lambda_4
    l.append(np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=complex))
    # lambda_5
    l.append(np.array([[0,0,-1j],[0,0,0],[1j,0,0]], dtype=complex))
    # lambda_6
    l.append(np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=complex))
    # lambda_7
    l.append(np.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=complex))
    # lambda_8
    l.append(np.array([[1,0,0],[0,1,0],[0,0,-2]], dtype=complex) / np.sqrt(3))
    return l

class SU3EquivariantLayer:
    """
    Neural network layer respecting SU(3) symmetry.
    Input: tensor of shape (batch, 3) or (batch, 8) depending on representation.
    """
    def __init__(self, input_dim=3, output_dim=3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Learnable weights (8 coefficients)
        self.weights = np.random.randn(8) * 0.01
        # Optional bias
        self.bias = np.zeros(output_dim)
        # Generators
        self.generators = gell_mann_matrices()

    def forward(self, x):
        """
        x : array of shape (batch, input_dim)
        Returns : array of shape (batch, output_dim)
        """
        batch_size = x.shape[0]
        out = np.zeros((batch_size, self.output_dim), dtype=complex)
        # For simplicity, we assume input_dim=3 (fundamental representation)
        for a in range(8):
            # Apply generator lambda_a to the input
            # x is treated as a complex vector of dimension 3
            transformed = np.einsum('ij,bj->bi', self.generators[a], x)
            out += self.weights[a] * transformed
        out += self.bias
        # Activation function (e.g., ReLU on real part, or absolute value)
        return np.abs(out)

# Example usage
if __name__ == "__main__":
    # Create a layer
    layer = SU3EquivariantLayer(input_dim=3, output_dim=3)
    # Dummy input (batch of 4 vectors of dimension 3)
    x = np.random.randn(4, 3) + 1j * np.random.randn(4, 3)
    y = layer.forward(x)
    print("Layer output:\n", y)
