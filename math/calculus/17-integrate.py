#!/usr/bin/env python3
'''
Write a function that calculates the integral of a polynomial.
'''


def poly_integral(poly, C=0):
    '''
    Function that calculates the integral of a polynomial.
    '''
    if not isinstance(poly, list) or not isinstance(C, int):
        return None

    # if not all(isinstance(coef, (int, float)) for coef in poly):
    #     return None

    if len(poly) == 0:
        return None

    # Calculate the integral
    integral = [C]  # Start with the integration constant
    for power, coef in enumerate(poly):
        new_coef = coef / (power + 1)
        # If the new coefficient is a whole number, convert it to an integer
        if new_coef.is_integer():
            new_coef = int(new_coef)
        integral.append(new_coef)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
