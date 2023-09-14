import numpy as np


class BaseAlgorithm(object):
    def argmax_rand(self, x):
        """ return the index of the maximum element in the array, ignoring nans. 
        If there are multiple max valued elements return 1 at random"""
        max_val = np.nanmax(x)
        indicies = np.where(x == max_val)[0]
        return np.random.choice(indicies) 