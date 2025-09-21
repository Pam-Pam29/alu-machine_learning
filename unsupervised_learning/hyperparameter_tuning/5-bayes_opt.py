#!/usr/bin/env python3
"""
Bayesian Optimization
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian optimization"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Acquisition function"""
        mu, sigma = self.gp.predict(self.X_s)
        
        if self.minimize:
            mu_sample = np.min(self.gp.Y)
            imp = mu_sample - mu - self.xsi
        else:
            mu_sample = np.max(self.gp.Y)
            imp = mu - mu_sample - self.xsi
            
        Z = np.zeros(sigma.shape)
        with np.errstate(divide='ignore'):
            Z = np.where(sigma > 0, imp / sigma, Z)
        
        # Calculate Expected Improvement
        first_term = imp * (0.5 + 0.5 * np.erf(Z / np.sqrt(2)))
        second_term = sigma * np.exp(-0.5 * Z**2) / np.sqrt(2 * np.pi)
        ei = first_term + second_term
        ei[sigma == 0] = 0
        
        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei

    def optimize(self, iterations=100):
        """Optimize the black-box function"""
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            
            # Check if this point has already been sampled
            if any(np.isclose(X_next, x, atol=1e-6) for x in self.gp.X):
                break
                
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
        
        # Find the optimal point
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)
            
        return self.gp.X[idx], self.gp.Y[idx]
