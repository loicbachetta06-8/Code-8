"""
Pseudo-random number generator based on the Code 8 fractal structure.
Author: Loïc Bachetta
License: Creative Commons Attribution 4.0 International
"""

import numpy as np

# SU(3) structure constants (absolute values)
# Non-zero f_abc triplets (a,b,c) -> value
F_ABC = {
    (1,2,3): 1.0,
    (1,4,7): 0.5, (1,5,6): 0.5,
    (2,4,6): 0.5, (2,5,7): 0.5,
    (3,4,5): 0.5, (3,6,7): 0.5,
    (4,5,8): np.sqrt(3)/2, (6,7,8): np.sqrt(3)/2
}

# Symmetric filling (antisymmetric)
def get_f(a, b, c):
    if (a,b,c) in F_ABC:
        return F_ABC[(a,b,c)]
    elif (b,c,a) in F_ABC:
        return F_ABC[(b,c,a)]
    elif (c,a,b) in F_ABC:
        return F_ABC[(c,a,b)]
    elif (a,c,b) in F_ABC:
        return -F_ABC[(a,c,b)]
    elif (b,a,c) in F_ABC:
        return -F_ABC[(b,a,c)]
    elif (c,b,a) in F_ABC:
        return -F_ABC[(c,b,a)]
    else:
        return 0.0

# Evolution matrix A_ij = sum_m f_mij
A = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        for m in range(8):
            A[i,j] += get_f(m+1, i+1, j+1)

phi = (1 + np.sqrt(5)) / 2
H = np.array([8.0, 0, 0, 0, 0, 0, 0, 0])

def step(v):
    """
    v : array of shape (8,8) : 8 vectors of dimension 8.
    Returns the next state after tanh normalization.
    """
    v_next = np.zeros_like(v)
    # Coupling term
    total = np.sum(v, axis=0)
    delta = (H - total) / 8.0
    for i in range(8):
        # Apply matrix A (rotation)
        rot = np.zeros(8)
        for j in range(8):
            rot += A[i,j] * v[j]
        v_next[i] = np.tanh( (1/phi) * rot + delta )
    return v_next

def extract_bits(v, num_bits=512):
    """
    Extracts bits from the state v.
    """
    bits = []
    for i in range(8):
        for d in range(8):
            x = v[i,d]
            y = (x + 1.0) / 2.0  # normalize to [0,1]
            k = int(y * 65536)   # 2^16
            for j in range(8):
                bits.append( (k >> j) & 1 )
    return bits[:num_bits]

def generate_random_bits(seed=None, iterations=100, num_bits=1024):
    """
    Generates a pseudo-random bit sequence.
    """
    if seed is None:
        seed = np.random.randint(0, 2**32)
    np.random.seed(seed)
    v = np.random.randn(8, 8) * 0.1  # initialization
    for _ in range(iterations):
        v = step(v)
    return extract_bits(v, num_bits)

# Example usage
if __name__ == "__main__":
    bits = generate_random_bits(iterations=200, num_bits=128)
    print("Generated bits:", bits)
    print("Proportion of 1s:", sum(bits)/len(bits))
