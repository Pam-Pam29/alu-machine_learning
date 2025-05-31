#!/usr/bin/env python3
'''
This module contains a function that calculates the determinant of a matrix.
'''


def determinant(matrix):
    '''
    This function calculates the determinant of a matrix.
    '''
    if not isinstance(matrix, list) or len(matrix) == 0 \
       or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Handle 0x0 matrix
    if matrix == [[]]:
        return 1

    # Check if matrix is square
    matrix_size = len(matrix)
    if any(len(row) != matrix_size for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case: 1x1 matrix
    if matrix_size == 1:
        return matrix[0][0]

    # Base case: 2x2 matrix
    if matrix_size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case: nxn matrix
    det_value = 0
    for col in range(matrix_size):
        # Create submatrix by removing first row and current column
        submatrix = [row[:col] + row[col+1:] for row in matrix[1:]]
        # Recursive call and accumulate determinant
        cofactor = (-1) ** col
        det_value += cofactor * matrix[0][col] * determinant(submatrix)

    return det_value
