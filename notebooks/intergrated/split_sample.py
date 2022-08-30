import sys
sys.path.append("/home/winter/TreeSegment/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel-v1.0.0/notebooks/")
import numpy as np
from PIL import Image
from config.Preprocessing import *
from core.visualize import display_images

config = Configuration(folder="test1")

sampleImage = '_0.png'
fn = os.path.join(config.path_to_write, config.extracted_ndvi_filename + sampleImage )
print(fn)

ndvi_img = Image.open(fn)
pan_img = Image.open(fn.replace(config.extracted_ndvi_filename ,config.extracted_pan_filename))
read_ndvi_img = np.array(ndvi_img)
read_pan_img = np.array(pan_img)
annotation_im = Image.open(fn.replace(config.extracted_ndvi_filename ,config.extracted_annotation_filename))
read_annotation = np.array(annotation_im)
weight_im = Image.open(fn.replace(config.extracted_ndvi_filename ,config.extracted_boundary_filename))
read_weight = np.array(weight_im)
all_images = np.array([read_ndvi_img, read_pan_img, read_annotation, read_weight ])

display_images(np.expand_dims(np.transpose(all_images, axes=(1,2,0)), axis=0))
# plt.imshow(read_weight)