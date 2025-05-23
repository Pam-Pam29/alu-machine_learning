#!/usr/bin/env python3
"""
Concatenates two matrices along a specific axis
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specified axis.

    Parameters:
    mat1 (list): The first matrix (list of lists or deeper).
    mat2 (list): The second matrix.
    axis (int): The axis along which to concatenate. Defaults to 0.

    Returns:
    list or None: A new matrix if shapes are compatible; None otherwise.
    """
    from copy import deepcopy

    def shape(matrix):
        """Recursively determines the shape of a matrix"""
        if not isinstance(matrix, list):
            return []
        return [len(matrix)] + shape(matrix[0])

    def is_shape_compatible(s1, s2, axis):
        """Check if two shapes are compatible for concatenation"""
        if len(s1) != len(s2):
            return False
        for i in range(len(s1)):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    s1 = shape(mat1)
    s2 = shape(mat2)

    if not is_shape_compatible(s1, s2, axis):
        return None

    def concat(m1, m2, level=0):
        """Recursively concatenate matrices"""
        if level == axis:
            return deepcopy(m1) + deepcopy(m2)
        return [concat(m1[i], m2[i], level + 1) for i in range(len(m1))]

    return concat(mat1, mat2)
