import numpy as np


def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)



def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Konstanta za sprječavanje dijeljenja s nulom
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Osiguraj da su vrijednosti unutar (epsilon, 1-epsilon)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

def categorical_cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-15  # Konstanta za sprječavanje dijeljenja s nulom
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Osiguraj da su vrijednosti unutar (epsilon, 1-epsilon)
    return -(y_true / y_pred) / len(y_true)