import tifffile as tifi
from tqdm import tqdm
import cv2
import os
import matplotlib.pyplot as plt


# img = tifi.imread('0211_clean.tif')
# cnt = 0
# for i in tqdm(range(len(img[0]))):
#     for j in range(len(img[1])):
#         if img[i,j] == 1:
#             cnt += 1
#
# print(cnt)
# crack = (cnt / 182680000) * 100
# print(crack, "%")


file_path = 'noisy_train/mask_crack'
file_names = os.listdir(file_path)

crack = []

for filename in file_names:
    # print(filename)
    img = cv2.imread('noisy_train/mask_crack/' + filename)
    cnt = 0
    for i in tqdm(range(len(img[0]))):
        for j in range(len(img[1])):
            if img[i,j] == 1:
                cnt += 1
    crack.append(cnt)

print(crack)

plt.hist(crack)

plt.show()

