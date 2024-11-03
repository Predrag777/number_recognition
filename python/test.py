from activation_functions import *
from convolution import *
from losses import *
from utils import *
import numpy as np
import os
from torchvision import transforms
import cv2
import shutil

def test(c, max_pool, r, d1, d2, softmax, x_test, y_test):
    count = 0
    for x, y in zip(x_test, y_test):
        out = c.forward(x)  ###Convolutional
        out = max_pool.forward(out)

        out = sigmoid(out)
        out = r.forward(out)
        out = d1.forward(out)
        out = sigmoid(out)
        out = d2.forward(out)
        out = softmax.forward(out)
        if np.argmax(out) == np.argmax(y):
            count += 1

    print("Acc: ", ((count / len(x_test)) * 100), '   ', count)

    acc=((count / len(x_test)) * 100)
    return acc

def SS(c, max_pool, r, d1, d2, softmax, x_test):
    # Putanja do slike
    image_path = "/opt/lampp/htdocs/Diplomski/izlazne_slike/slika.png"
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detekcija kontura
    ret, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Kreiranje i ispraznjenje foldera za slike
    output_folder = "slika"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # Čuvanje svakog slova kao posebne slike
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        letter_image = image[y:y + h, x:x + w]
        letter_path = os.path.join(output_folder, f"slovo_{idx}.png")
        cv2.imwrite(letter_path, letter_image)

    #folder_path="/home/predrag/PycharmProjects/Waffen/izlazne_slike"
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    folder_path="/opt/lampp/htdocs/Diplomski/slika"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):  # Provera da li je putanja do fajla

            #file_name=file_path.split("\\")[1]
            ss=first_formating(file_path)
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

            out = c.forward(x)  ###Convolutional
            out = max_pool.forward(out)

            out = sigmoid(out)
            out = r.forward(out)
            out = d1.forward(out)
            out = sigmoid(out)
            out = d2.forward(out)
            out = softmax.forward(out)

            print("BROJEVI SU: ",np.argmax(out))

