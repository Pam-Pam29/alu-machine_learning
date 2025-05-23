#!/usr/bin/env python3
'''
Defines a function that adds two two-dimensional arrays
'''


def add_matrices2D(mat1, mat2):
    '''
    Returns a new array that adds two other two-dimesional arrays
    '''
    r1 = len(mat1)
    c1 = len(mat1[0])
    r2 = len(mat2)
    c2 = len(mat2[0])

    if r1 == r2 and c1 == c2:
        matrix = [[None for _ in range(c1)] for _ in range(r1)]
        for r in range(r1):
            for c in range(c1):
                matrix[r][c] = mat1[r][c] + mat2[r][c]

        return matrix

    return None
