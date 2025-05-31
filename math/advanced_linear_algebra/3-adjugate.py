#!/usr/bin/env python3
'''
This module contains the function to calculate the adjugate of a matrix.
'''


def adjugate(matrix):
    '''
    This function calculates the adjugate of a matrix.
    '''
    if not isinstance(matrix, list) or len(matrix) == 0\
       or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case for 1x1 matrix
    if n == 1:
        return [[1]]

    # Helper function to get the submatrix
    def submatrix(matrix, i, j):
        return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

    # Helper function to calculate determinant
    def determinant(matrix):
        if len(matrix) == 1:
            return matrix[0][0]
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        return sum(((-1) ** j) * matrix[0][j] *
                   determinant(submatrix(matrix, 0, j))
                   for j in range(len(matrix)))

    # Calculate the cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            minor = determinant(submatrix(matrix, i, j))
            cofactor_row.append((-1) ** (i + j) * minor)
        cofactor_matrix.append(cofactor_row)

    # Transpose the cofactor matrix to get the adjugate
    adjugate_matrix = [[cofactor_matrix[j][i]
                        for j in range(n)] for i in range(n)]

    return adjugate_matrix
