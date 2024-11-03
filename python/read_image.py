import cv2
import os
import shutil

# Putanja do slike
image_path = "izlazne_slike/slika.png"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detekcija kontura
ret, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Kreiranje i čišćenje foldera za slike
output_folder = "slika"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)
os.chmod(output_folder, 0o777)

# Čuvanje svakog slova kao posebne slike
for idx, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    letter_image = image[y:y + h, x:x + w]
    letter_path = os.path.join(output_folder, f"slovo_{idx}.png")
    cv2.imwrite(letter_path, letter_image)
    os.chmod(letter_path, 0o777)

print("Slike su sačuvane u folderu 'slika'")



