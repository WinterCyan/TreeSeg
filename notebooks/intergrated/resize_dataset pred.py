from email.mime import image
from PIL import Image
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

dataset_dir = "/media/lenovo/Elements SE/resize/"
pan_dir = dataset_dir + "pan/"
#ndvi_dir = dataset_dir + "ndvi/"
#annotation_dir = dataset_dir + "annotation/"
#weight_dir = dataset_dir + "weight/"

ndvi_list = [dataset_dir+n for  n in os.listdir(dataset_dir)]
#pan_list = [pan_dir+n for  n in os.listdir(pan_dir)]
#annotation_list = [annotation_dir+n for  n in os.listdir(annotation_dir)]
#weight_list = [weight_dir+n for  n in os.listdir(weight_dir)]


all_img = ndvi_list #+ pan_list + annotation_list + weight_list


resize_dataset_dir = "/media/lenovo/Elements SE/resize_test/"
if not os.path.exists(resize_dataset_dir):
    os.mkdir(resize_dataset_dir)


fig = plt.figure()
for img_name in all_img:
    img_arr = np.load(img_name)
    resize_arr = zoom(img_arr, (256/100, 256/100))
    print( resize_arr )
    save_path = img_name[35:]
    np.save(resize_dataset_dir+save_path,resize_arr)


