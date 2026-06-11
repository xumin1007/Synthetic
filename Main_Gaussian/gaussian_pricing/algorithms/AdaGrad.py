import numpy as np
import time

from ..utils.project import project_l1

class AdaGradOnline:
    """
    AdaGrad implementation for online least squares with optional L1 projection.
    
    This matches the problem structure of AMLE-CP and ILQX, but processes
    data recursively (one sample at a time) for O(1) step complexity.
    """
    def __init__(self, n, d, W, lr=0.1, epsilon=1e-8):
        self.n = n
        self.d = d
        self.W = W
        self.lr = lr
        self.epsilon = epsilon
        
        # Initial parameters (theta and gamma flattened together, same as PGD initial_point)
        # We start with 0.5 everywhere matching gradient_descent_projected_likelihood
        self.vars = np.full(n * n * d + n * n, 0.5, dtype=np.float64)
        self.G = np.zeros_like(self.vars)
        
    def update(self, X_t, P_t, D_t):
        """
        X_t: feature vector, shape (d*n, 1) or conceptually (d, n) but passed as flattened column
        P_t: price vector, shape (n, 1)
        D_t: demand vector, shape (n, 1) or (n,)
        
        Returns:
            theta_new: updated theta, shape (n, n*d)
            gamma_new: updated gamma, shape (n, n)
            update_time: time taken for this recursive update
        """
        start_time = time.time()
        
        # Restore theta and gamma
        theta = self.vars[:self.n * self.n * self.d].reshape((self.n, self.n * self.d))
        gamma = self.vars[self.n * self.n * self.d:].reshape((self.n, self.n))
        
        col_xt = np.reshape(X_t, (-1,))
        p_vec = np.reshape(P_t, (self.n,))
        d_vec = np.reshape(D_t, (self.n,))
        
        # Predictions and residual
        pred = (theta @ col_xt - gamma @ p_vec).reshape(self.n,)
        residual = d_vec - pred
        
        # Compute gradient of the least squares loss l_t = 0.5 * || d_vec - (theta * x - gamma * p) ||^2
        # d_lt/d_theta = (d_vec - pred) * (-x^T) = residual * (-x^T)
        # d_lt/d_gamma = (d_vec - pred) * (p^T) = residual * (p^T)
        grad_theta = np.outer(residual, -col_xt)
        grad_gamma = np.outer(residual, p_vec)
        
        grad_vars = np.concatenate([grad_theta.flatten(), grad_gamma.flatten()])
        
        # AdaGrad update
        self.G += grad_vars ** 2
        adjusted_lr = self.lr / (np.sqrt(self.G) + self.epsilon)
        self.vars = self.vars - adjusted_lr * grad_vars
        
        # Projection onto L1 ball W
        self.vars = project_l1(self.vars, self.W)
        
        update_time = time.time() - start_time
        
        # Extract new parameters to return
        theta_new = self.vars[:self.n * self.n * self.d].reshape((self.n, self.n * self.d))
        gamma_new = self.vars[self.n * self.n * self.d:].reshape((self.n, self.n))
        
        return theta_new, gamma_new, update_time
