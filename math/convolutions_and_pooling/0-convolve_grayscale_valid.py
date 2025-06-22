#!/usr/bin/env python3
"""Module for performing convolution operations on grayscale images."""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Returns:
        numpy.ndarray: The convolved images
    """
    m = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    convoluted = np.zeros((m, height - kh + 1, width - kw + 1))

    for h in range(height - kh + 1):
        for w in range(width - kw + 1):
            output = np.sum(images[:, h:h + kh, w:w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, h, w] = output

    return convoluted
