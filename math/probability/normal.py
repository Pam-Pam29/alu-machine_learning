#!/usr/bin/env python3
'''
Normal Distribution
'''


class Normal:
    '''
    Class Normal
    '''
    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))

            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        '''
        Calculates the z-score of a given x-value
        '''
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        '''
        Calculates the x-value of a given z-score
        '''
        return self.mean + z * self.stddev

    def pdf(self, x):
        '''
        Calculates the value of the PDF for a given x-value
        '''
        return (2.7182818285 **
                ((-1/2) * ((x - self.mean) / self.stddev) ** 2))\
            / (self.stddev * (2 * 3.1415926536) ** 0.5)

    def cdf(self, x):
        """
        calculates the value of the CDF for a given x-value

        parameters:
            x: x-value

        return:
            the CDF value for x
        """
        mean = self.mean
        stddev = self.stddev
        pi = 3.1415926536
        value = (x - mean) / (stddev * (2 ** (1 / 2)))
        erf = value - ((value ** 3) / 3) + ((value ** 5) / 10)
        erf = erf - ((value ** 7) / 42) + ((value ** 9) / 216)
        erf *= (2 / (pi ** (1 / 2)))
        cdf = (1 / 2) * (1 + erf)
        return cdf
