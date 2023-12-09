import random
import math
import numpy as np
import time

seed = 1
np.random.seed(seed)

def naive(X, prime=False):
    return X
    
def ReLU(X, prime=False):
    if prime:
        return X > 0
    return np.maximum(0, X)
    
def sigmoid(X, prime=False):
    term = np.exp(-X)
    if prime:
        return term/((term+1)*(term+1))
    return 1./(term+1)

def tanh(X, prime=False):
    x_tanh = math.tanh(x)
    if prime:
        return 1 -(x_tanh * x_tanh)
    return x_tanh

def softmax(X, prime=False):
    exps = np.exp(X - X.max())
    exps_sum = np.sum(exps, axis=0)
    ratio = exp / exps_sum
        
    if prime:
        return ratio * (1 - ratio)
    return ratio