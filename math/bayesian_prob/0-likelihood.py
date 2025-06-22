#!/usr/bin/env python3
'''
Calculates the likelihood of obtaining this data
given various hypothetical probabilities
of developing severe side effects
'''
import numpy as np


def likelihood(x, n, P):
    '''
    Determines the likelihood of obtaining the data
    '''
    # Check if n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Check if x is a non-negative integer
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that"
                         " is greater than or equal to 0")

    # Check if x is not greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Check if P is a 1D numpy.ndarray
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Check if all values in P are in the range [0, 1]
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the binomial coefficient
    log_coeff = np.sum(np.log(np.arange(n - x + 1, n + 1))
                       ) - np.sum(np.log(np.arange(1, x + 1)))

    # Create masked arrays to avoid log(0) and log(1)
    P_masked = np.ma.masked_outside(P, 0, 1)
    log_P = np.ma.log(P_masked)
    log_1_minus_P = np.ma.log(1 - P_masked)

    # Calculate log likelihood
    log_likelihood = log_coeff + x * log_P + (n - x) * log_1_minus_P

    log_likelihood = log_likelihood.filled(-np.inf)

    # Convert log likelihood to likelihood
    return np.exp(log_likelihood)
