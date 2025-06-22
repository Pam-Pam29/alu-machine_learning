#!/usr/bin/env python3
"""
Module that performs convolution on images using multiple kernels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs convolution on images using multiple kernels

    Returns:
        numpy.ndarray: Convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    # Calculate padding
    if padding == 'same':
        ph = ((((h - 1) * sh) + kh - h) // 2) + 1
        pw = ((((w - 1) * sw) + kw - w) // 2) + 1
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    # Pad images
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant',
                    constant_values=0)

    # Calculate output dimensions
    out_h = ((h + (2 * ph) - kh) // sh) + 1
    out_w = ((w + (2 * pw) - kw) // sw) + 1

    # Initialize output array
    output = np.zeros((m, out_h, out_w, nc))

    # Perform convolution using only 3 for loops
    for k in range(nc):  # Loop through each kernel
        curr_kernel = kernels[:, :, :, k]
        i = 0
        for row in range(0, h + (2 * ph) - kh + 1, sh):  # Height stride
            j = 0
            for col in range(0, w + (2 * pw) - kw + 1, sw):  # Width stride
                # Extract window and perform element-wise multiplication
                window = padded[:, row:row + kh, col:col + kw, :]
                # Sum across all dimensions except batch (m)
                conv_result = np.sum(window * curr_kernel, axis=(1, 2, 3))
                output[:, i, j, k] = conv_result
                j += 1
            i += 1

    return output
