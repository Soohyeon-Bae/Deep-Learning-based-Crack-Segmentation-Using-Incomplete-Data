import os
import cv2

file_path = 'noisy_train/img_128'
file_names = os.listdir(file_path)

for filename in file_names:

    img = cv2.imread('noisy_train/img_128/' + filename)
    print("img.shape = {0}".format(img.shape))

    resize_img = cv2.resize(img, (512, 512))
    print("resize_img.shape = {0}".format(resize_img.shape))

    cv2.imwrite('noisy_train/img_resize/'+filename, resize_img)

