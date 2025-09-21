#!/usr/bin/env python3
'''
This script demonstrates the use of dropout
regularization in a neural network
'''

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    '''
    Function that updates the weights of a neural network
    with Dropout regularization using gradient descent

    Arguments:
     - Y: numpy.ndarray (1, m) with correct labels
          m is the number of labels
     - weights: dictionary with the weights and biases of the network
     - cache: dictionary with the outputs of each layer of the network
     - alpha: learning rate
     - keep_prob: probability that a node will be kept
     - L: number of layers of the network
    '''
    m = Y.shape[1]
    Al = cache['A' + str(L)]
    dAl = Al - Y

    for layer in reversed(range(1, L + 1)):
        w_key = 'W' + str(layer)
        b_key = 'b' + str(layer)
        Al_key = 'A' + str(layer)
        Al1_key = 'A' + str(layer - 1)
        D_key = 'D' + str(layer)

        Al = cache[Al_key]
        gld = 1 - np.power(Al, 2)
        if layer == L:
            dZl = dAl
        else:
            dZl = dAl * gld
            dZl *= cache[D_key] / keep_prob

        Wl = weights[w_key]
        Al1 = cache[Al1_key]
        dWl = (1 / m) * np.matmul(dZl, Al1.T)
        dbl = (1 / m) * np.sum(dZl, axis=1, keepdims=True)
        dAl = np.matmul(Wl.T, dZl)
        weights[w_key] = weights[w_key] - alpha * dWl
        weights[b_key] = weights[b_key] - alpha * dbl
