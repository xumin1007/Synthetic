import numpy as np

def p_norm(x, p):
    return np.sum(np.abs(x) ** p) ** (1 / p)