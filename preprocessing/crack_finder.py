import os
import numpy as np
import cv2
from tqdm import tqdm
import shutil

# 균열이 있는 레이블 찾기
path_dir = 'noisy_train/mask_128'
files = os.listdir(path_dir)
yes_crack = []
for filename in tqdm(files):
    img = cv2.imread('noisy_train/mask_128/' + filename)
    h = 128
    w = 128
    # 레이블에서 값이 1인 픽셀 찾기(균열인 부분)
    for y in range(0, h):
        for x in range(0, w):
            if img[y, x][0] == 1:
                yes_crack.append(filename)
                pass

my_list = yes_crack
new_list = []
for v in my_list:
    if v not in new_list:
        new_list.append(v)
print(new_list)

# 균열이 있는 마스크만 이동시키기
path_dir_2 = 'noisy_train/mask_128/'
new_path = 'noisy_train/mask_128_crack/'
if not os.path.exists(new_path):
    os.mkdir(new_path)

for n in new_list:
    file = path_dir_2 + n
    shutil.copy(path_dir_2 + n, new_path + n)
    print('{} has been moved to new folder!'.format(n))

# 균열이 있는 이미지만 이동시키기
path_dir_3 = 'noisy_train/img_128/'
new_path = 'noisy_train/img_128_crack/'
for n in new_list:
    file = path_dir_3 + n
    shutil.copy(path_dir_3 + n, new_path + n)
    print('{} has been moved to new folder!'.format(n))

# # 평행 이동 (translate.py)
#
# import cv2
# import numpy as np
#
# img = cv2.imread('../img/fish.jpg')
# rows, cols = img.shape[0:2]  # 영상의 크기
#
# dx, dy = 100, 50            # 이동할 픽셀 거리
#
# # ---① 변환 행렬 생성
# mtrx = np.float32([[1, 0, dx],
#                    [0, 1, dy]])
# # ---② 단순 이동
# dst = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy))
#
# # ---③ 탈락된 외곽 픽셀을 0으로 보정
# dst2 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (0, 0, 0))
#
# cv2.imshow('original', img)
# cv2.imshow('trans', dst)
# cv2.imshow('BORDER_CONSTATNT', dst2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
