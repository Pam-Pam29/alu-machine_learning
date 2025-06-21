#!/usr/bin/env python3
'''
Binonial distro
'''


class Binomial:
    '''
    Class for binomial distribution
    '''
    def __init__(self, data=None, n=1, p=0.5):
        '''
        Constructor
        '''
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum([(x - mean) ** 2 for x in data]) / len(data)
            p = 1 - variance / mean
            n = round(mean / p)
            p = mean / n
            self.n = n
            self.p = p

    def pmf(self, k):
        '''
        Calculates the value of the PMF for a given number of “successes”
        '''
        if k < 0:
            return 0
        k = int(k)
        return (self.factorial(self.n) /
                (self.factorial(k) * self.factorial(self.n - k))) * \
            (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        '''
        Calculates the value of the CDF for a given number of “successes”
        '''
        if k < 0:
            return 0
        k = int(k)
        return sum([self.pmf(i) for i in range(k + 1)])

    def factorial(self, k):
        '''
        Factorial
        '''
        if k == 0:
            return 1
        return k * self.factorial(k - 1)
