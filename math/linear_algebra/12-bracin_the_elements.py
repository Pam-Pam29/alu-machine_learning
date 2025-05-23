#!/usr/bin/env python3
'''
Module that performs element-wise operations on two matrices
'''


def np_elementwise(mat1, mat2):
    '''
    Module that performs element-wise operations on two matrices
    '''
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
