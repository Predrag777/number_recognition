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


epochs=20
r=Reshape((2,13,26), (2 * 13 * 26, 1))
d1=Dense(2 * 13 * 26,100)
d2=Dense(100,10)
softmax=Softmax()

learning_rate=0.1
c=Convolutional((1, 28, 28), 3, 5)
max_pool=MaxPool()


file_path = "test_fajl.txt"
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


from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


image_path = "izlazne_slike/slika.png"  # Stavite putanju do vaše slike

image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detekcija linija
ret, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Kreiranje i ispraznjenje foldera za slike
output_folder = "slika"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# Cuvanje svakog slova kao posebne slike
for idx, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    letter_image = image[y:y + h, x:x + w]
    letter_path = os.path.join(output_folder, f"slovo_{idx}.png")
    cv2.imwrite(letter_path, letter_image)

folder_path="slika"
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):  
        #####Konacno formatiranje
        first_formating(file_path)
        img = Image.open(file_path)
        img = centering_img(img)  # Centriranje slike
        img = img.resize((28, 28), Image.BICUBIC)

        tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        tensor = transform(img)
        tensor = tensor.unsqueeze_(0)
        x = np.array(tensor)
        x = x.reshape(1, 28, 28)

        print("Nacrtao si: ", predict(c, max_pool, r, d1, d2, softmax, x))