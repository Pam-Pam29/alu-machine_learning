#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5-bayes_opt
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process.
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor

        Arguments:
            - f: the black-box function to optimize
            - X_init: numpy.ndarray of shape (t, 1) of initial input samples
            - Y_init: numpy.ndarray of shape (t, 1) of initial output samples
            - bounds: tuple (min, max) of the search space
            - ac_samples: number of samples to analyze during acquisition
            - l: length parameter for the kernel
            - sigma_f: standard deviation of the black-box function outputs
            - xsi: exploration-exploitation factor
            - minimize: bool, True if minimizing, False if maximizing
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement.

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

        # Mask EI at points that have already been sampled
        for i, x in enumerate(self.X_s):
            if np.any(np.isclose(x, self.gp.X)):
                EI[i] = -np.inf  # prevent already sampled points

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function.

        Arguments:
            iterations: maximum number of iterations to perform

        Returns:
            X_opt: numpy.ndarray of shape (1,) representing the optimal point
            Y_opt: numpy.ndarray of shape (1,) representing the optimal value
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            X_next_reshaped = X_next.reshape(1, 1)
            Y_next = self.f(X_next)
            self.gp.update(X_next_reshaped, np.array([[Y_next]]))

        idx_opt = np.argmin(self.gp.Y) if self.minimize else np.argmax(self.gp.Y)
        X_opt = self.gp.X[idx_opt].reshape(1,)
        Y_opt = self.gp.Y[idx_opt].reshape(1,)
        return X_opt, Y_opt
