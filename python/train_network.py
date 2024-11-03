import shutil
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

#print("Prikupljaju se podaci...\n")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 100)
print("ENDE\n")


######Initialisation
#print("Obavlja se inicijalizacija...")
epochs=10
learning_rate=0.1

c=Convolutional((1, 28, 28), 3, 5)
max_pool=MaxPool()
r=Reshape((2,13,26), (2 * 13 * 26, 1))
d1=Dense(2 * 13 * 26,100)
d2=Dense(100,10)
softmax=Softmax()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#print("treniranje je pokrenuto")
train(c, max_pool, r, d1, d2, softmax, x_train, y_train, learning_rate, epochs)

#print("treniranje je zavrseno")
#print("Testiranje...")
test(c, max_pool, r, d1, d2, softmax, x_test, y_test)