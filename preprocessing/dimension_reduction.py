import os
import cv2

file_path = 'noisy_train/mask_crack'
file_names = os.listdir(file_path)

for filename in file_names:
    # print(filename)
    img = cv2.imread('noisy_train/mask_crack/' + filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('noisy_train/mask_crack/' + filename, gray)
    print(gray.shape)
