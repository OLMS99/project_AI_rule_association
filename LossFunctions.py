import random
import math
import numpy as np
import time

seed = 1
np.random.seed(seed)

def naive_loss(y, output):
    output_reshaped = output.reshape(-1,1)
    y_reshaped = y.reshape(-1,1)
    return output_reshaped - y_reshaped

def naive_loss_prime(y, output):
    output_reshaped = output.reshape(-1,1)
    y_reshaped = y.reshape(-1,1)
    return output_reshaped - y_reshaped

def mae(y_true, y_pred):
    diff = abs(y_true - y_pred)
    return np.mean(diff* diff)

def mae_prime(y_true, y_pred):
    if y_pred >= y_true:
        return 1

    if y_pred < t_true:
        return -1

def mse(y_true, y_pred):
    diff = y_true - y_pred
    return np.mean(diff* diff)

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true)/ np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)