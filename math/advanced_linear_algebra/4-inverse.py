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
    
    # Create an augmented matrix [A|I] using float conversion
    augmented = []
    for i, row in enumerate(matrix):
        new_row = [float(x) for x in row]  # Convert to float
        identity_part = [float(int(i == j)) for j in range(n)]
        augmented.append(new_row + identity_part)
    
    # Gaussian elimination with partial pivoting
    for i in range(n):
        # Find pivot (row with largest absolute value in column i)
        pivot_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[pivot_row][i]):
                pivot_row = k
        
        # Check for singular matrix
        if abs(augmented[pivot_row][i]) < 1e-14:
            return None  # Matrix is singular
        
        # Swap rows if needed
        if pivot_row != i:
            augmented[i], augmented[pivot_row] = augmented[pivot_row], augmented[i]
        
        # Scale current row to make diagonal element 1
        pivot = augmented[i][i]
        for j in range(2 * n):
            augmented[i][j] /= pivot
        
        # Eliminate column entries above and below the pivot
        for k in range(n):
            if k != i and abs(augmented[k][i]) > 1e-14:
                factor = augmented[k][i]
                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]
    
    # Extract inverse from the right half of the augmented matrix
    result = []
    for i in range(n):
        row = []
        for j in range(n, 2 * n):
            val = augmented[i][j]
            # Clean up floating point errors
            if abs(val) < 1e-14:
                val = 0.0
            row.append(val)
        result.append(row)
    
    return result
