#!/usr/bin/env python3
'''
This is a simple script that demonstrates the use
of the numpy library to solve a system of linear equations. | Python | Shobi
'''


def add_matrices(mat1, mat2):
    '''
    Adds two matrices together.
    '''
    if type(mat1) != type(mat2):
        return None

    if isinstance(mat1, (int, float)):
        return mat1 + mat2

    if len(mat1) != len(mat2):
        return None

    result = []

    for i in range(len(mat1)):
        if isinstance(mat1[i], list):
            sub_result = add_matrices(mat1[i], mat2[i])
            if sub_result is None:
                return None
            result.append(sub_result)
        elif isinstance(mat1[i], (int, float)):
            result.append(mat1[i] + mat2[i])
        else:
            return None

    return result