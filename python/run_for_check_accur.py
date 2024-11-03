import shutil

import numpy as np
from torchvision import transforms
import cv2
import os

from keras.src.datasets import mnist

from activation_functions import *
from convolution import *
from losses import *
from utils import *
from train import train, predict
from test import test
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test, y_test = preprocess_data(x_test, y_test, 100)

epochs=20
r=Reshape((2,13,26), (2 * 13 * 26, 1))
d1=Dense(2 * 13 * 26,100)
d2=Dense(100,10)
softmax=Softmax()

learning_rate=0.1
c=Convolutional((1, 28, 28), 3, 5)
max_pool=MaxPool()


file_path = "/test_fajl.txt"
with open(file_path, "r") as file:
    counter=0
    content = file.readline().split('|')
    for i in range(c.kernels.shape[0]):
        for j in range(c.kernels.shape[1]):
            for f in range(c.kernels.shape[2]):
                for t in range(c.kernels.shape[3]):
                    c.kernels[i,j,f,t]=float(content[counter])
                    counter+=1
    for i in range(c.biases.shape[0]):
        for j in range(c.biases.shape[1]):
            for f in range(c.biases.shape[2]):
                c.biases[i,j,f]=float(content[counter])
                counter+=1

    for i in range(d1.weights.shape[0]):
        for j in range(d1.weights.shape[1]):
            d1.weights[i,j]=float(content[counter])
            counter+=1

    for i in range(d2.weights.shape[0]):
        for j in range(d2.weights.shape[1]):
            d2.weights[i,j]=float(content[counter])
            counter+=1

test(c,max_pool,r,d1,d2,softmax,x_test,y_test)