import sys
sys.path.append("/home/lenovo/code/TreeSeg/notebooks")
import tensorflow as tf
import numpy as np
from PIL import Image
import rasterio
import imgaug as ia
from imgaug import augmenters as iaa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import imageio
import os
import time
import rasterio.warp             # Reproect raster samples
from functools import reduce
from tensorflow.keras.models import load_model
from configx import UNetTraining

from core.UNet import UNet
from core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity
from core.optimizers import adaDelta, adagrad, adam, nadam
from core.frame_info import FrameInfo_Input
from core.frame_info import image_normalize
from core.dataset_generator import DataGenerator_Input
from core.split_frames import split_dataset
from core.visualize import display_images

import json
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt  # plotting tools
import matplotlib.patches as patches
from matplotlib.patches import Polygon

import warnings                  # ignore annoying warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

config = UNetTraining.Configuration()

frames = []
dataset_dir = "/media/lenovo/Elements SE/predict/resize1/"
all_files = os.listdir(dataset_dir)
print(f"all files: {len(all_files)}")

# load all ndvi*.png
# all_files_ndvi = [fn for fn in all_files if fn.startswith(config.ndvi_fn) and fn.endswith(config.image_type)]
for ix in range(1000,len(all_files),1000):
    all_files_ndvi = [fn for fn in all_files if fn.__contains__(config.ndvi_fn) and fn.endswith(config.image_type)][ix-1000:ix]
    print(f"all ndvi: {len(all_files_ndvi)}")
    for i, fn in enumerate(all_files_ndvi):
        ndvi_img = np.load(os.path.join(dataset_dir, fn))
        ndvi_img[ndvi_img<=0.5]=0
        pan_img = np.load(os.path.join(dataset_dir, fn.replace(config.ndvi_fn,config.pan_fn)))
        comb_img = np.stack((ndvi_img, pan_img), axis=0)
        comb_img = np.transpose(comb_img, axes=(1,2,0)) #Channel at the end

        # annotation_img = np.load(os.path.join(config.base_dir, fn.replace(config.ndvi_fn,config.annotation_fn)))
        # weight_img = np.load(os.path.join(config.base_dir, fn.replace(config.ndvi_fn,config.weight_fn)))
        f = FrameInfo_Input(comb_img)
        frames.append(f)

    training_frames, validation_frames, testing_frames  = split_dataset(frames, config.frames_json, config.patch_dir)
    # training_frames = validation_frames = testing_frames  = list(range(len(frames)))
    # annotation_channels = config.input_label_channel + config.input_weight_channel
    # train_generator = DataGenerator(config.input_image_channel, config.patch_size, training_frames, frames, annotation_channels, augmenter = 'iaa').random_generator(config.BATCH_SIZE, normalize = config.normalize)
    # val_generator = DataGenerator(config.input_image_channel, config.patch_size, validation_frames, frames, annotation_channels, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize)
    test_generator = DataGenerator_Input(config.input_image_channel, config.patch_size_input, testing_frames, frames, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize)

    OPTIMIZER = adaDelta
    LOSS = tversky 

    #Only for the name of the model in the very end
    OPTIMIZER_NAME = 'AdaDelta'
    LOSS_NAME = 'weightmap_tversky'

    # Declare the path to the final model
    # If you want to retrain an exising model then change the cell where model is declared. 
    # This path is for storing a model after training.

    timestr = time.strftime("%Y%m%d-%H%M")
    chf = config.input_image_channel + config.input_label_channel
    chs = reduce(lambda a,b: a+str(b), chf, '')


    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    model_path = os.path.join(config.model_path,'trees_{}_{}_{}_{}_{}.h5'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs,config.input_shape[0]))
    #print(model_path)

    # The weights without the model architecture can also be saved. Just saving the weights is more efficent.

    #     weight_path="./saved_weights/UNet/{}/".format(timestr)
    # if not os.path.exists(weight_path):
    #     os.makedirs(weight_path)
    # weight_path=weight_path + "{}_weights.best.hdf5".format('UNet_model')
    # print(weight_path)

    model = UNet([config.BATCH_SIZE, *config.input_shape],config.input_label_channel)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy])

    # Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing
    from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard


    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = False)
 
    #reduceonplatea; It can be useful when using adam as optimizer
    #Reduce learning rate when a metric has stopped improving (after some patience,reduce by a factor of 0.33, new_lr = lr * factor).
    #cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=4, verbose=1, mode='min', min_delta=0.0001, cooldown=4, min_lr=1e-16)

    #early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=15)

    log_dir = os.path.join('./logs','UNet_{}_{}_{}_{}_{}'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs, config.input_shape[0]))
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    callbacks_list = [checkpoint, tensorboard] #reduceLROnPlat is not required with adaDelta

    # Load model after training
    #    If you load a model with different python version, than you may run into a problem: https://github.com/keras-team/keras/issues/9595#issue-303471777

    model_path = "/home/lenovo/code/TreeSeg/notebooks/saved_models/UNet/model.h5"

    model = load_model(model_path, custom_objects={'tversky': LOSS, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy , 'specificity': specificity, 'sensitivity':sensitivity}, compile=False)

    # In case you want to use multiple GPU you can uncomment the following lines.
    # from tensorflow.python.keras.utils import multi_gpu_model
    # model = multi_gpu_model(model, gpus=2, cpu_merge=False)

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[dice_coef, dice_loss, accuracy, specificity, sensitivity])


    # Print one batch on the training/test data!
    for i in range(len(all_files_ndvi)):
        test_images = next(test_generator)
        #5 images per row: pan, ndvi, label, weight, prediction
        print(test_images.shape)
        print(test_images[0,:,:,1])
        print(f"max: {np.max(test_images[0,:,:,1])}")
        print(f"min: {np.min(test_images[0,:,:,1])}")
        prediction = model.predict(test_images, steps=1)
        prediction[prediction>0.5]=1
        prediction[prediction<=0.5]=0
        # real_label[real_label>0.5]=1
        # real_label[real_label<=0.5]=0
        display_images(np.concatenate((test_images, prediction), axis = -1),cmap=i,save=True,dir=f"/media/lenovo/Elements SE/predict/result1/")