from __future__ import annotations
from ast import Import
from re import S
import sys
from time import time
sys.path.append("/home/lenovo/code/TreeSeg/notebooks/")
import os
from configx.Preprocessing_prediction import *
from PIL import Image
Image.MAX_IMAGE_PIXELS= None
from core.dataset_generator import DataGenerator_Input
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import cv2 as cv
from scipy.ndimage import zoom
import rasterio                  # I/O raster data (netcdf, height, geotiff, ...)
from configx import UNetTraining
config = UNetTraining.Configuration()
from core.frame_info import FrameInfo_Input
config = Configuration("full_area3")
SPLIT_UNIT = 55
HALF_SPLIT_UNIT = 54
dataset_dir = "/media/lenovo/palsar(部分压缩包)/resize/"
TYPE_ENUM = ["ndvi","pan"]#,"annotation","weight"

# read from pan_*, ndvi_*, boundary_*, annotation_* files, split into square dataset imgs.
def cut_area(fn, source_dir, mean_thr=0.1):
    frames = []
    half_mean_thr = mean_thr/2
    img_format = ".tif"
    full_path1 = os.path.join(source_dir, "ndvi.tif")
    full_path2 = os.path.join(source_dir, "pan.tif")

    #ndvi_im = np.array(Image.open(full_path1))
    #pan_im = np.array(Image.open(full_path2))
    ndvi_img = rasterio.open(full_path1)
    pan_img = rasterio.open(full_path2)
    ndvi_im =ndvi_img.read(1)
    pan_im1 =pan_img.read(1)
    pan_im2 =pan_img.read(2)
    pan_im3=pan_img.read(3)
    full_images = np.array([ndvi_im,pan_im1,pan_im2,pan_im3])#, annotation_im, weight_im
    f = FrameInfo_Input(full_images)
    frames.append(f)
    # annotation_channels = config.input_label_channel + config.input_weight_channel
    
    train_generator = DataGenerator_Input(config.input_image_channel, config.patch_size, full_images, frames, augmenter = 'iaa').random_generator(config.BATCH_SIZE, normalize = config.normalize)

    #annotation_im = np.array(Image.open(full_path.replace(config.extracted_ndvi_filename ,config.extracted_annotation_filename)))
    #weight_im = np.array(Image.open(full_path.replace(config.extracted_ndvi_filename ,config.extracted_boundary_filename)))


    (_, height, width) = full_images.shape

    row_split_count = int(height/SPLIT_UNIT)
    col_split_count = int(width/SPLIT_UNIT)
    #print(f"spliting {full_path} into {row_split_count}x{col_split_count} squares...")

    trim_img = full_images[:,:row_split_count*SPLIT_UNIT, :col_split_count*SPLIT_UNIT]
    split_row = np.split(trim_img, row_split_count, axis=1)
    split_col = [np.split(row_img, col_split_count, axis=2) for row_img in split_row]
    for i in range(row_split_count):
        for j in range(col_split_count):
            idx = col_split_count*i+j
            square_img = split_col[i][j]
            #mean_v = np.mean(square_img[2,:,:])
            #mean_1 = np.mean(square_img[2,0:HALF_SPLIT_UNIT,0:HALF_SPLIT_UNIT])
            #mean_2 = np.mean(square_img[2,0:HALF_SPLIT_UNIT,HALF_SPLIT_UNIT:])
            #mean_3 = np.mean(square_img[2,HALF_SPLIT_UNIT:,0:HALF_SPLIT_UNIT])
            #mean_4 = np.mean(square_img[2,HALF_SPLIT_UNIT:,HALF_SPLIT_UNIT:])
            #if mean_1>half_mean_thr and mean_2>half_mean_thr and mean_3>half_mean_thr and mean_4>half_mean_thr and mean_v>mean_thr:
            for k in range(len(TYPE_ENUM)):
                    # if k==1:
                    #     print(square_img[k])
                    #     print(square_img[k].shape)
                    #     np.save("/home/winter/Desktop/1.npy", square_img[k])
                    #     # cv.imwrite("/home/winter/Desktop/1.png", square_img[k])
                    #     # tif_img = Image.fromarray(square_img[k])
                    #     # tif_img.save("/home/winter/Desktop/1.tif")
                    #     # temp_img = Image.fromarray((square_img[k]*255).astype(np.uint8), mode='L')
                    #     # # temp_img.show()
                    #     load_npy = np.load("/home/winter/Desktop/1.npy")
                    #     plt.imshow(load_npy)
                    #     # load_img = np.array(temp_img)
                #resize_arr = zoom(square_img[k], (256/SPLIT_UNIT, 256/SPLIT_UNIT))
                np.save(dataset_dir+f"{fn}_{idx}_{TYPE_ENUM[k]}.npy",square_img[k])
    plt.show()

if __name__ == '__main__':
    for i in range(0,226):
        cut_area(str(i), source_dir=config.path_to_write)