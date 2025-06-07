#!/usr/bin/env python3
'''
Write a function that takes a list of integers as input
and returns the derivative of the polynomial represented by the list.
'''


def poly_derivative(poly):
    '''
    Returns the derivative of the polynomial represented by the list poly.
    '''
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for power, coeff in enumerate(poly[1:], start=1):
        derivative.append(coeff * power)

    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()

    return derivative if any(derivative) else [0]