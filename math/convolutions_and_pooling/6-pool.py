#!/usr/bin/env python3
"""
Module that performs pooling operations on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images

    Returns:
        numpy.ndarray: Pooled images
    """
    # Get dimensions
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    out_h = ((h - kh) // sh) + 1
    out_w = ((w - kw) // sw) + 1

    # Initialize output array
    output = np.zeros((m, out_h, out_w, c))

    # Perform pooling using only 2 for loops
    row = 0
    for i in range(0, h - kh + 1, sh):
        col = 0
        for j in range(0, w - kw + 1, sw):
            # Extract window
            window = images[:, i:i + kh, j:j + kw, :]

            # Apply pooling operation based on mode
            if mode == 'max':
                pool_result = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                pool_result = np.mean(window, axis=(1, 2))

            output[:, row, col, :] = pool_result
            col += 1
        row += 1

    return output
