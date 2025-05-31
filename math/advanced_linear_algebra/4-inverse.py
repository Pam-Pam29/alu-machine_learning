#!/usr/bin/env python3
'''
This script demonstrates how to calculate the inverse of a matrix
without using the numpy library.
'''

def inverse(matrix):
    '''
    This function calculates the inverse of a matrix.
    '''
    if not isinstance(matrix, list) or len(matrix) == 0 or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    
    # Special case for 1x1 matrix
    if n == 1:
        if matrix[0][0] == 0:
            return None
        return [[round(1 / matrix[0][0], 1)]]
    
    # Create an augmented matrix [A|I] with high precision
    augmented = []
    for i, row in enumerate(matrix):
        new_row = [float(x) for x in row]
        identity_part = [1.0 if i == j else 0.0 for j in range(n)]
        augmented.append(new_row + identity_part)
    
    # Gaussian elimination with partial pivoting
    for i in range(n):
        # Find the row with the largest absolute value in column i (partial pivoting)
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k
        
        # Check if matrix is singular
        if abs(augmented[max_row][i]) < 1e-12:
            return None
        
        # Swap rows if necessary
        if max_row != i:
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        
        # Make all rows below this one 0 in current column
        pivot = augmented[i][i]
        
        # Scale the pivot row
        for j in range(2 * n):
            augmented[i][j] /= pivot
        
        # Eliminate all other rows
        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]
    
    # Extract the inverse matrix from the right side
    result = []
    for i in range(n):
        row = []
        for j in range(n, 2 * n):
            val = augmented[i][j]
            # Clean up floating point errors more aggressively
            if abs(val) < 1e-15:
                val = 0.0
            # Apply minimal rounding to reduce floating point noise
            val = round(val, 16)  # Round to 16 decimal places to reduce noise
            row.append(val)
        result.append(row)
    
    return result
