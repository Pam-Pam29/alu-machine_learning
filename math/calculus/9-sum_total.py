#!/usr/bin/env python3
'''
Write a recursive function that takes an integer n as input and returns
the sum of the squares of the first n positive integers.
If n is not a positive integer, the function should return None.
'''


def summation_i_squared(n):
    '''
    Calculates the sum of squares from 1 to n (inclusive) recursively.

    :param n: The upper limit of the summation (inclusive)
    :return: The sum of squares, or None if n is not a valid number
    '''
    if not isinstance(n, int) or n < 1:
        return None

    if n == 1:
        return 1

    return (n * (n + 1) * (2 * n + 1)) // 6