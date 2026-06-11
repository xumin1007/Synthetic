import numpy as np

def randomP(choices, n):
    p = np.zeros(n)
    for i in range(n):
        p[i] = np.random.choice(choices)  # draw one value at random
    return p
