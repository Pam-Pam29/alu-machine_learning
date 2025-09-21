#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5-bayes_opt.py
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process.
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor.

        Arguments:
            - f: black-box function to optimize
            - X_init: numpy.ndarray (t,1) of initial input samples
            - Y_init: numpy.ndarray (t,1) of initial output samples
            - bounds: tuple (min, max) of search space
            - ac_samples: number of samples for acquisition
            - l: length parameter for kernel
            - sigma_f: standard deviation of outputs
            - xsi: exploration-exploitation factor
            - minimize: True to minimize, False to maximize
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample using Expected Improvement.

        Returns:
            X_next: numpy.ndarray of shape (1,)
            EI: numpy.ndarray of shape (ac_samples,)
        """
        mu_sample, sigma_sample = self.gp.predict(self.X_s)

        if self.minimize:
            Y_sample = np.min(self.gp.Y)
            imp = Y_sample - mu_sample - self.xsi
        else:
            Y_sample = np.max(self.gp.Y)
            imp = mu_sample - Y_sample - self.xsi

        with np.errstate(divide='ignore'):
            Z = imp / sigma_sample
            EI = (imp * norm.cdf(Z)) + (sigma_sample * norm.pdf(Z))
            EI[sigma_sample == 0.0] = 0.0

        # Mask EI at already sampled points
        for i, x in enumerate(self.X_s):
            if np.any(np.isclose(x, self.gp.X)):
                EI[i] = -np.inf

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function.

        Arguments:
            iterations: maximum number of iterations

        Returns:
            X_opt, Y_opt
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            X_next = X_next.reshape(1, 1)

            # Stop if X_next has already been sampled
            if np.any(np.isclose(X_next, self.gp.X)):
                break  # stop immediately, do not add extra points

            # Evaluate function and update GP
            Y_next = self.f(X_next)
            self.gp.update(X_next, np.array([[Y_next]]))

        # Return the optimal point/value
        idx = np.argmin(self.gp.Y) if self.minimize else np.argmax(self.gp.Y)
        X_opt = self.gp.X[idx].reshape(1,)
        Y_opt = self.gp.Y[idx].reshape(1,)
        return X_opt, Y_opt
