#!/usr/bin/env python3
'''
Defines a function that calculates the mean and covariance of a data set
'''
import numpy as np


def mean_cov(X):
    '''
    Calculates the mean and covariance of a data set
    '''
    # Check if X is a 2D numpy array
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    # Get the shape of X
    n, d = X.shape

    # Check if there are multiple data points
    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate the mean
    mean = np.mean(X, axis=0, keepdims=True)

    # Center the data
    X_centered = X - mean

    # Calculate the covariance matrix
    cov = np.dot(X_centered.T, X_centered) / (n - 1)

    return mean, cov
