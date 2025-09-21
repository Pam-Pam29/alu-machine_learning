#!/usr/bin/env python3
'''
This script creates an early stopping callback
using tensorflow library.
'''


def early_stopping(cost, opt_cost, threshold, patience, count):
    '''
    The function early_stopping creates an early stopping callback.
    It uses the tensorflow library.

    Arguments:
    cost -- the current cost of the network
    opt_cost -- the lowest recorded cost of the network
    threshold -- the threshold used to determine early stopping
    patience -- the patience count used for early stopping
    count -- the count of how long the threshold has not been met
    '''
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1
    if (count == patience):
        boolean = True
    else:
        boolean = False

    return boolean, count
