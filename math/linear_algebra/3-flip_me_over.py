#!/usr/bin/env python3
'''
This module contains a function that returns a transposed matrix
'''


def matrix_transpose(matrix):
    '''
    Returns a transposed matrix based on the arguments given
    '''
    r = len(matrix)
    c = len(matrix[0])

    transposed = [[None for _ in range(r)] for _ in range(c)]
    for i in range(r):
        for j in range(c):
            transposed[j][i] = matrix[i][j]
    return transposed
