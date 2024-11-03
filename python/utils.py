import numpy as np
import pandas as pd
from keras.utils import to_categorical
from PIL import Image, ImageChops, ImageOps



def preprocess_data(x, y, limit):
    a = np.where(y == 0)[0][:limit]
    b = np.where(y == 1)[0][:limit]
    c = np.where(y == 2)[0][:limit]
    d = np.where(y == 3)[0][:limit]
    e = np.where(y == 4)[0][:limit]
    f = np.where(y == 5)[0][:limit]
    g = np.where(y == 6)[0][:limit]
    h = np.where(y == 7)[0][:limit]
    i = np.where(y == 8)[0][:limit]
    j = np.where(y == 9)[0][:limit]


    all_indices = np.hstack((a,b,c,d,e,f,g,h,i,j))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255-0.5
    y = to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y


def normalize_image(image_matrix):
    min_value = np.min(image_matrix)
    max_value = np.max(image_matrix)

    normalized_image = -0.5 + (image_matrix - min_value) / (max_value - min_value) * 1.0

    return normalized_image


def centering_img(img):
    img = img.convert("L")  # Konvertiranje u crno-bijeli format
    w, h = img.size[:2]
    left, top, right, bottom = w, h, -1, -1
    imgpix = img.getdata()

    for y in range(h):
        offset_y = y * w
        for x in range(w):
            if imgpix[offset_y + x] > 0:
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    shift_x = (left + (right - left) // 2) - w // 2
    shift_y = (top + (bottom - top) // 2) - h // 2
    centered_img = ImageChops.offset(img, -shift_x, -shift_y)

    #print(f"Shift X: {shift_x}, Shift Y: {shift_y}")  # Dodajte ispis pomaka
    #print(f"Originalna veličina: {w}x{h}, Nova veličina: {centered_img.size[0]}x{centered_img.size[1]}")

    return centered_img

def first_formating(image_path):
    # Učitavanje slike
    original_image = Image.open(image_path)

    # Invertovanje boja (crna pozadina, beli trag)
    inverted_image = ImageOps.invert(original_image)

    # Dodavanje paddinga
    padded_inverted_image = ImageOps.expand(inverted_image, border=50, fill='black')

    padded_inverted_image.save(image_path)



class Reshape:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)