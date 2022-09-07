from __future__ import annotations
from inspect import BoundArguments
from nis import match
import sys
sys.path.append("/home/winter/code/TreeSeg/notebooks/")
import os
from config.Preprocessing import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

config = Configuration("full_area2")
SPLIT_UNIT = 108
HALF_SPLIT_UNIT = 54
dataset_dir = "/home/winter/code-resources/treeseg/dataset/"
TYPE_ENUM = ["ndvi","pan","annotation","weight"]

# read from pan_*, ndvi_*, boundary_*, annotation_* files, split into square dataset imgs.
def cut_area(fn, source_dir, mean_thr=0.1):
    img_format = ".png"
    full_path = os.path.join(source_dir, config.extracted_ndvi_filename+"_"+fn+img_format)

    ndvi_im = np.array(Image.open(full_path))
    pan_im = np.array(Image.open(full_path.replace(config.extracted_ndvi_filename ,config.extracted_pan_filename)))
    annotation_im = np.array(Image.open(full_path.replace(config.extracted_ndvi_filename ,config.extracted_annotation_filename)))
    weight_im = np.array(Image.open(full_path.replace(config.extracted_ndvi_filename ,config.extracted_boundary_filename)))

    print("shapes: ", ndvi_im.shape, pan_im.shape, annotation_im.shape, weight_im.shape)

    full_images = np.array([ndvi_im, pan_im, annotation_im, weight_im])
    (_, height, width) = full_images.shape

    row_split_count = int(height/SPLIT_UNIT)
    col_split_count = int(width/SPLIT_UNIT)
    print(f"spliting {full_path} into {row_split_count}x{col_split_count} squares...")

    trim_img = full_images[:,:row_split_count*SPLIT_UNIT, :col_split_count*SPLIT_UNIT]
    split_row = np.split(trim_img, row_split_count, axis=1)
    split_col = [np.split(row_img, col_split_count, axis=2) for row_img in split_row]
    print("split shape: ", split_col[0][0].shape)
    for i in range(row_split_count):
        for j in range(col_split_count):
            idx = col_split_count*i+j
            square_img = split_col[i][j]
            mean_v = np.mean(square_img[2,:,:])
            mean_1 = np.mean(square_img[2,0:HALF_SPLIT_UNIT,0:HALF_SPLIT_UNIT])
            mean_2 = np.mean(square_img[2,0:HALF_SPLIT_UNIT,HALF_SPLIT_UNIT:])
            mean_3 = np.mean(square_img[2,HALF_SPLIT_UNIT:,0:HALF_SPLIT_UNIT])
            mean_4 = np.mean(square_img[2,HALF_SPLIT_UNIT:,HALF_SPLIT_UNIT:])
            if mean_1>0.05 and mean_2>0.05 and mean_3>0.05 and mean_4>0.05 and mean_v>0.1:
                print(f"4 means: {mean_1},{mean_2},{mean_3},{mean_4}")
                for k in range(len(TYPE_ENUM)):
                    plt.imsave(dataset_dir+f"{fn}_{idx}_{TYPE_ENUM[k]}.png",square_img[k])
    plt.show()

if __name__ == '__main__':
    for i in range(0,5):
        cut_area(str(i), source_dir=config.path_to_write)