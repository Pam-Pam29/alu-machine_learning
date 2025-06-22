#!/usr/bin/env python3
'''
Calculates the posterior probability that the probability
of developing severe side effects is within
a specific range given the data
'''
from scipy import special


def posterior(x, n, p1, p2):
    '''
    Calculates the posterior probability that the probability
    '''
    # Input validations
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is"
                         " greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or not 0 <= p1 <= 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or not 0 <= p2 <= 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # Calculate the parameters of the Beta distribution
    alpha = x + 1
    beta = n - x + 1

    # Calculate the cumulative probability
    # using the regularized incomplete beta function
    cdf_p2 = special.betainc(alpha, beta, p2)
    cdf_p1 = special.betainc(alpha, beta, p1)

    # Calculate the posterior probability
    posterior_prob = cdf_p2 - cdf_p1

    return posterior_prob
