#!/usr/bin/env python3
'''
Defines a function that calculates the correlation of a data set
'''
import numpy as np


def correlation(C):
    '''
    Function is used to find the corr of a data set
    '''
    # Check if C is a numpy array
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    # Check if C is a square matrix
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Calculate the standard deviations
    std_devs = np.sqrt(np.diag(C))

    # Calculate the correlation matrix
    corr = C / np.outer(std_devs, std_devs)

    # Ensure the diagonal is exactly 1
    # (to handle potential floating-point errors)
    np.fill_diagonal(corr, 1)

    return corr
