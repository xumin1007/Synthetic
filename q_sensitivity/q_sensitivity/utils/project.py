import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# L1-ball projection


def project_l1(matrix, W):
    # Flatten matrix to a vector
    flat = matrix.flatten()
    abs_flat = np.abs(flat)

    # If current L1 norm is already <= W, return a copy unchanged
    if abs_flat.sum() <= W:
        return matrix.copy()

    # Sort by absolute value in descending order
    sorted_indices = np.argsort(abs_flat)[::-1]
    sorted_abs = abs_flat[sorted_indices]
    cumulative_sum = np.cumsum(sorted_abs)

    # Find largest rho such that sorted_abs[rho] > (cumulative_sum[rho] - W) / (rho + 1)
    rho = np.nonzero(sorted_abs > (cumulative_sum - W) / np.arange(1, len(sorted_abs)+1))[0]
    if len(rho) == 0:
        theta = 0
    else:
        rho = rho[-1]
        theta = (cumulative_sum[rho] - W) / (rho + 1.0)

    # Apply projection
    projected_flat = np.sign(flat) * np.maximum(abs_flat - theta, 0)

    # Reshape back to original matrix shape
    projected_matrix = projected_flat.reshape(matrix.shape)
    return projected_matrix
