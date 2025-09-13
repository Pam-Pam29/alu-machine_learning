#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification. Neural network
"""

import numpy as np


class Neuron:
    """
    This class defines a single neuron that mimicas the behaavior of tensorflow

    By default, the bias b is initialized to 0.
    Upon instantiation, a neuron takes in a single parameter:
    """

    def __init__(self, nx):
        """
        class constructor

        This is how the class is called upon instatiation
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
