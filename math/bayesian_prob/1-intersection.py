#!/usr/bin/env python3
'''
Finds the intersection of obtaining this data with
'''
import numpy as np


def intersection(x, n, P, Pr):
    '''
    Function that calculates the intersection of obtaining this data with
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

    # Check if Pr is a numpy.ndarray with the same shape as P
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    # Check if all values in P are in the range [0, 1]
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Check if all values in Pr are in the range [0, 1]
    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    # Check if Pr sums to 1
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the likelihood
    likelihood = np.zeros_like(P)
    for i, p in enumerate(P):
        if p == 0:
            likelihood[i] = 0 if x > 0 else 1
        elif p == 1:
            likelihood[i] = 0 if x < n else 1
        else:
            likelihood[i] = np.exp(
                np.sum(np.log(np.arange(n - x + 1, n + 1))) -
                np.sum(np.log(np.arange(1, x + 1))) +
                x * np.log(p) + (n - x) * np.log(1 - p)
            )

    # Calculate the intersection
    return likelihood * Pr
