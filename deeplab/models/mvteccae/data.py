#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, time
import random
import numpy as np
import cv2
from PIL import Image, ImageOps
from sklearn.utils import class_weight
from tensorflow.keras.utils import Sequence

from deeplab.src.common.data_utils import random_horizontal_flip, random_vertical_flip, random_brightness, random_grayscale, random_chroma, random_contrast, random_sharpness, random_blur, random_zoom_rotate, random_gridmask, random_crop, random_histeq, normalize_image


class CAEGenerator(Sequence):
    def __init__(self, dataset_path, data_list,
                 batch_size=1,
                 input_shape=(256, 256),
                 color_mode=1,
                 is_eval=False,
                 augment=True):
        # get real path for dataset
        dataset_realpath = os.path.realpath(dataset_path)
        self.image_path_list = [os.path.join(dataset_realpath, 'images', image_id.strip() + '.png') for image_id in data_list]
        # initialize random seed
        np.random.seed(int(time.time()))

        self.batch_size = batch_size
        self.input_shape = input_shape
        self.augment = augment
        self.is_eval = is_eval
        self.color_mode = color_mode

        # Preallocate memory for batch input/output data
        self.batch_images = np.zeros((batch_size, input_shape[0], input_shape[1], self.color_mode), dtype='float32')

    def get_batch_image_path(self, i):
        return self.image_path_list[i*self.batch_size:(i+1)*self.batch_size]

    def __len__(self):
        return len(self.image_path_list) // self.batch_size

    def __getitem__(self, i):
        
        for n, (image_path) in enumerate(zip(self.image_path_list[i*self.batch_size:(i+1)*self.batch_size])):

            # Load image and label array
            img = Image.open(image_path).convert('RGB')
            image = np.array(img)
            img.close()

            label = np.zeros(img.shape, np.uint8)
            
            # Do augmentation
            if self.augment:

                # random adjust brightness
                image, label = random_brightness(image, label)

                # random adjust color level
                image, label = random_chroma(image, label)

                # random adjust contrast
                image, label = random_contrast(image, label)

                # random adjust sharpness
                image, label = random_sharpness(image, label)

                # random convert image to grayscale
                image, label = random_grayscale(image, label)

                # random do gaussian blur to image
                image, label = random_blur(image, label)

                # random do histogram equalization using CLAHE
                image, label = random_histeq(image, label)


            # Resize image & label mask to model input shape
            image = cv2.resize(image, self.input_shape[::-1])

            # normalize image as input
            image = normalize_image(image)

            # append input image and label array
            self.batch_images[n] = image


        return self.batch_images


    def on_epoch_end(self):
        # Shuffle dataset for next epoch
        c = list(zip(self.image_path_list, self.label_path_list))
        random.shuffle(c)
        self.image_path_list, self.label_path_list = zip(*c)

