#!/usr/bin/env python3
'''
This is a simple script that demonstrates the use of the numpy library to solve a system of linear equations.
'''
import numpy as np


def cat_matrices(mat1, mat2, axis=0):
    '''
    Concatenates two matrices along a specific axis
    '''
    try:
        arr1 = np.array(mat1)
        arr2 = np.array(mat2)
        
        result = np.concatenate((arr1, arr2), axis=axis)
        
        return result.tolist()
    except ValueError:
        return None
