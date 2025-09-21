#!/usr/bin/env python3
'''
This script implements the forward propagation of the dropout
layer. The dropout layer is a regularization technique that
prevents overfitting by randomly setting some neurons to zero
during training. This script implements the forward propagation
of the dropout layer.
'''

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    '''
    The goal of this function is to implement the forward
    propagation of the dropout layer. The dropout layer is
    a regularization technique that prevents overfitting by
    randomly setting some neurons to zero during training.

    Arguments:
        X : numpy array - input data
        weights : dict - weights of the neural network
        L : int - number of layers in the neural network
        keep_prob : float - probability of keeping a neuron active
    '''
    outputs = {}
    outputs["A0"] = X
    for index in range(L):
        weight = weights["W{}".format(index + 1)]
        bias = weights["b{}".format(index + 1)]
        z = np.matmul(weight, outputs["A{}".format(index)]) + bias
        dropout = np.random.binomial(1, keep_prob, size=z.shape)
        if index != (L - 1):
            A = np.tanh(z)
            A *= dropout
            A /= keep_prob
            outputs["D{}".format(index + 1)] = dropout
        else:
            A = np.exp(z)
            A /= np.sum(A, axis=0, keepdims=True)
        outputs["A{}".format(index + 1)] = A
    return outputs
