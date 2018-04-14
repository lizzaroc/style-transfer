import numpy as np
import tensorflow as tf

NUM_DIGITS = 10

def binary_encode(i, num_digits):
    return np.array([i >> d & 1 ])