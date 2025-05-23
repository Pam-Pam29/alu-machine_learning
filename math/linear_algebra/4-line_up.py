#!/usr/bin/env python3
'''
Module that adds two arrays
'''


def add_arrays(arr1, arr2):
    '''
    Returns a new array that adds two other arrays together
    '''
    n1 = len(arr1)
    n2 = len(arr2)
    result = []

    if n1 == n2:
        for i in range(n1):
            result.append(arr1[i] + arr2[i])
        return result
    return None
