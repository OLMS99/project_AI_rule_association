import random
import math
import numpy as np
import time

def naive(X, prime=False):
    return X

def ReLU(X, prime=False):
    if prime:
        return X > 0
    return np.maximum(0, X)

def sigmoid(X, slope = 1, prime = False):
    term = np.exp(- X * slope)
    if prime:
        sigma = sigmoid(X, slope = slope)
        return slope * sigma * (1 - sigma)
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