#!/usr/bin/env python3
'''
Module that calculates the shape of a matrix
'''


def matrix_shape(matrix):
    '''
    calculates the shape of a matrix
    '''
    ptr = matrix
    measurements = []
    while isinstance(ptr, list):
        measurements.append(len(ptr))
        ptr = ptr[0]
    return measurements
