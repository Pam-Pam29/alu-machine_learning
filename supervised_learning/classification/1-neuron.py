#!/usr/bin/env python3
"""
The definition of a single neuron performing binary classification

A bit more complex than the previous version
"""


import numpy as np


class Neuron:
    """
    This class mimics the behavior of a single neuron in a neural network

    nx: The number of input features to the neuron
    """

    def __init__(self, nx):
        """
        Initializes a neuron

        nx is a positive integer that represents the number
        of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter for private class attribute __W
        """
        return (self.__W)

    @property
    def b(self):
        """
        Getter for private class attribute __b
        """
        return (self.__b)

    @property
    def A(self):
        """
        Getter for private class attribute __A
        """
        return (self.__A)
