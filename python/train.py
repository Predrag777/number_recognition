from activation_functions import *
from convolution import *
from losses import *
from utils import *
import numpy as np



def train(c, max_pool, r, d1, d2, softmax, x_train, y_train, learning_rate, epochs):
    for i in range(epochs):
        error = 0
        if i == 2:  ######Na trecoj iteraciji promeni lr
            learning_rate = 0.001
        if i == 4:
            learning_rate = 0.0001
        for x, y in zip(x_train, y_train):
            out = c.forward(x)  ###Convolutional
            out = max_pool.forward(out)
            # print(out.shape)

            out = sigmoid(out)
            out = r.forward(out)
            out = d1.forward(out)
            out = sigmoid(out)
            out = d2.forward(out)
            out = softmax.forward(out)

            error += categorical_cross_entropy(y, out)

            grad = categorical_cross_entropy_derivative(y, out)

            grad = softmax.backward(grad, learning_rate)
            grad = d2.backward(grad, learning_rate)
            grad = np.multiply(grad, sigmoid_prime(grad))
            grad = d1.backward(grad, learning_rate)
            grad = r.backward(grad, learning_rate)
            grad = np.multiply(grad, sigmoid_prime(grad))

            grad = max_pool.backprop(grad)
            grad = c.backward(grad, learning_rate)

        error /= len(x_train)
        print((i + 1), '  ', error)
    
    file_path = "test_fajl.txt"

    ####PREPISI BITNE VREDNOSTI ZA MREZU####
    with open(file_path, "w") as file:
        for i in c.kernels:
            for j in i:
                for jj in j:
                    for qq in jj:
                        s=str(qq)+'|'
                        file.writelines(s)
        for i in c.biases:
            for j in i:
                for jj in j:
                    s=str(jj)+'|'
                    file.writelines(s)
        for i in d1.weights:
            for j in i:
                s=str(j)+'|'
                file.writelines(s)
        for i in d2.weights:
            for j in i:
                s = str(j) + '|'
                file.writelines(s)






def predict(c, max_pool, r, d1, d2, softmax, x):
    out = c.forward(x)  ###Convolutional
    out = max_pool.forward(out)

    out = sigmoid(out)
    out = r.forward(out)
    out = d1.forward(out)
    out = sigmoid(out)
    out = d2.forward(out)
    out = softmax.forward(out)

    return np.argmax(out)