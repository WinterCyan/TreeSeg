import cv2 as cv
import imageio
from PIL import Image
from rasterio import open as rstopen
from os import path
from time import time
from matplotlib import pyplot as plt
import numpy as np

def gau_blur(denoisy,k):
    conn = cv.GaussianBlur(denoisy, (k,k), cv.BORDER_DEFAULT)
    bin_conn = conn
    bin_conn[conn>0] = 1.0
    int_conn = (bin_conn * 255).astype(int)
    return int_conn

def pipe(img, denoisy_k, denoisy_t, conn_k):
    # blur
    b = cv.blur(img, (denoisy_k, denoisy_k))
    # int
    int_b = (b * img * 255).astype(int)

    # threshold
    bin_b = int_b
    bin_b[int_b<denoisy_t] = 0
    bin_b[int_b>=denoisy_t] = 255

    # conn
    denoisy = (bin_b/255).astype(float)
    conn = gau_blur(denoisy, conn_k)

    return conn


if __name__ == '__main__':
    segmap_path = "/home/lenovo/code/TreeSeg/standalone/segmap_file/segmap.png"
    img = rstopen(segmap_path).read(1)
    init_h, init_w = img.shape
    h = int(init_h/2)
    w = int(init_w/2)
    img = cv.resize(img, dsize=(h,w), interpolation=cv.INTER_NEAREST)
    print(f'resize to {img.shape}')
    # par = img[20000:30000, 10000:20000]
    par = img

    mask_par = (par/255).astype(float)
    conn31 = pipe(mask_par, 31, 15, 31)
    conn51 = pipe(mask_par, 31, 15, 51)
    conn71 = pipe(mask_par, 31, 15, 71)
    conn101 = pipe(mask_par, 31, 15, 101)
    conn151 = pipe(mask_par, 31, 15, 151)

    t1 = time()
    conn201 = pipe(mask_par, 31, 15, 201)
    t2 = time()
    print(f'time: {t2-t1}')

    cv.imwrite(path.join(path.dirname(segmap_path), 'conn31.png'), conn31)
    cv.imwrite(path.join(path.dirname(segmap_path), 'conn51.png'), conn51)
    cv.imwrite(path.join(path.dirname(segmap_path), 'conn71.png'), conn71)
    cv.imwrite(path.join(path.dirname(segmap_path), 'conn101.png'), conn101)
    cv.imwrite(path.join(path.dirname(segmap_path), 'conn151.png'), conn151)
    cv.imwrite(path.join(path.dirname(segmap_path), 'conn201.png'), conn201)

    # fig = plt.figure(figsize=(9,3))

    # fig.add_subplot(1,3,1)
    # plt.imshow(conn33)
    # fig.add_subplot(1,3,2)
    # plt.imshow(conn51)
    # fig.add_subplot(1,3,3)
    # plt.imshow(conn71)

    # fig.savefig(path.join(path.dirname(segmap_path), 'conn.png'))
