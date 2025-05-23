#!/usr/bin/env python3
'''
Module that multiplies two matrices
'''


def mat_mul(mat1, mat2):
    '''
    Multiplies two matrices and returns the result matrix
    '''
    r1, c1 = len(mat1), len(mat1[0])
    r2, c2 = len(mat2), len(mat2[0])

    if c1 != r2:
        return None

    result_matrix = [[0 for _ in range(c2)] for _ in range(r1)]

    for i in range(r1):
        for j in range(c2):
            for k in range(c1):
                result_matrix[i][j] += mat1[i][k] * mat2[k][j]

    return result_matrix
