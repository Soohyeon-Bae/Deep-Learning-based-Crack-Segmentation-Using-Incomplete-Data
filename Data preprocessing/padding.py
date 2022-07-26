import os
import cv2


def padding(img, set_size=512):
    h, w, c = img.shape
    delta_w = set_size - w
    delta_h = set_size - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return new_img


file_path = 'noisy_trainimg_128'
file_names = os.listdir(file_path)

for filename in file_names:
    img = cv2.imread('noisy_trainimg_128/' + filename)
    new_img = padding(img)
    cv2.imwrite('noisy_train/mask_padding/' + filename, new_img)
