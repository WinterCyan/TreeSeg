from email.mime import image
from PIL import Image
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

dataset_dir = "/home/winter/code-resources/treeseg/dataset/"
pan_dir = dataset_dir + "pan/"
ndvi_dir = dataset_dir + "ndvi/"
annotation_dir = dataset_dir + "annotation/"
weight_dir = dataset_dir + "weight/"

ndvi_list = [ndvi_dir+n for  n in os.listdir(ndvi_dir)]
pan_list = [pan_dir+n for  n in os.listdir(pan_dir)]
annotation_list = [annotation_dir+n for  n in os.listdir(annotation_dir)]
weight_list = [weight_dir+n for  n in os.listdir(weight_dir)]


all_img = ndvi_list + pan_list + annotation_list + weight_list


resize_dataset_dir = "/home/winter/code-resources/treeseg/dataset_resize/"
if not os.path.exists(resize_dataset_dir):
    os.mkdir(resize_dataset_dir)


fig = plt.figure()
for img_name in all_img:
    img_arr = np.load(img_name)
    # fig.add_subplot(1,2,1)
    # plt.imshow(img_arr)
    resize_arr = zoom(img_arr, (256/84, 256/84))
    # fig.add_subplot(1,2,2)
    # print(resize_arr)
    # plt.imshow(resize_arr)
    # plt.show()
    
    save_path = img_name.replace("dataset", "dataset_resize")
    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    np.save(save_path, resize_arr)


# for n in annotation_list[:5]:
#     img = Image.open(n)
#     print(img.size)
#     print(img.getchannel)
#     img_arr = np.array(img)
#     print(img_arr.shape)
#     print(img_arr.tolist())
#     print(np.max(img_arr))
#     print(np.min(img_arr))
    

    