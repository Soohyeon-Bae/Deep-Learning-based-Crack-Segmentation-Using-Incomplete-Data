import cv2
import os
from tqdm import tqdm
import numpy as np
import tifffile as tifi
import glob
import natsort

def img_clip(folder_dir,
             dst_dir,
             width,
             height,
             stride):
    """
    clip image into small size you set on "width" and "height". Marginal parts are filled with zero value.
    :parma folder_dir : directory for images.
    :param dst_dir: folder to save result images.
    :param width :
    :param height :
    :param stride :
    :return:
    """
    folder_dir = folder_dir
    img_list = os.listdir(folder_dir)
    stride = 128
    for idx, img_dir in enumerate(img_list):

        print("Processing {0}th image of {1}.".format(str(idx + 1), len(img_list)))
        name = os.path.splitext(img_dir)[0]

        abs_dir = os.path.join(folder_dir, img_dir)

        # flag : -1 = cv2.IMREAD_UNCHANGED
        _, ext = os.path.splitext(abs_dir)
        # print(".tif")
        # img = tifi.imread(abs_dir)
        # code = ".tif"
        img = cv2.imread(abs_dir).astype(np.float64)
        code = "cv2"
        print(img.shape)
        img_shp = np.shape(img)
        width_size = img_shp[1] // 128 + 1
        height_size = img_shp[0] // 128 + 1

        # clipping img instance and save
        # tqdm : 상태 진행률을 시각적으로 표현
        for i in tqdm(range(height_size)):

            height_front = stride * i
            height_rear = height + stride * i

            for j in range(width_size):

                width_front = stride * j
                width_rear = width + stride * j
                print(np.shape(img_shp)[0])
                if np.shape(img_shp)[0] == 3:
                    print("RGB COLOR IMAGE.")
                    frame = np.zeros([width, height, img_shp[2]])

                    if height_rear > img_shp[0]: height_rear = img_shp[0]
                    if width_rear > img_shp[1]: width_rear = img_shp[1]

                    if width_rear < width_front: continue
                    if height_rear < height_front: continue

                    img_part = img[height_front: height_rear, width_front: width_rear, :]
                    frame[0:height_rear - height_front, 0:width_rear - width_front, :] = img_part

                elif np.shape(img_shp)[0] == 2:
                    print("GRAYSCALE IMAGE.")
                    frame = np.zeros([width, height])

                    if height_rear > img_shp[0]: height_rear = img_shp[0]
                    if width_rear > img_shp[1]: width_rear = img_shp[1]

                    if width_rear < width_front: continue
                    if height_rear < height_front: continue

                    img_part = img[height_front: height_rear, width_front: width_rear]
                    frame[0:height_rear - height_front, 0:width_rear - width_front] = img_part

                # if code == ".tif":
                # file_dst_dir = dst_dir + "\\" + str(name) + "_{0}_{1}.tif".format(str(i), str(j))
                # if np.shape(img_shp)[0] == 3:
                #     frame = np.flip(frame, 2)
                # tifi.imwrite(file_dst_dir, frame)
                # elif code == ".cv2":
                file_dst_dir = dst_dir + "\\" + str(name) + "_{0}_{1}.png".format(str(i), str(j))
                cv2.imwrite(file_dst_dir, frame)

    print("Finished to clip and save images at {0}.".format(str(dst_dir)))

if __name__ == "__main__":
    # # Testing for image clipping of preprocessing
    folder_dir = "noisy_train/img_org"
    width = 128
    height = 128
    dst_dir = "noisy_train/img_128"
    img_clip(folder_dir = folder_dir, dst_dir = dst_dir, width = width, height = height, stride=128)
    # allFile_list = list(glob.glob(os.path.join(folder_dir, '*.png')))
    # print(allFile_list)
    # allFile_list = natsort.natsorted(allFile_list)
    # print(allFile_list)
    # i = 0
    # for name in allFile_list:
    #     src = os.path.join(name)
    #     dst = str(i) + '.png'
    #     dst = os.path.join(folder_dir, dst)
    #     os.rename(src, dst)
    #     i += 1