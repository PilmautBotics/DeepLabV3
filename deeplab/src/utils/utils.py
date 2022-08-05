import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import cv2

def check_mask_labels(masks):
    """
    Check mask labels
    Args:
        masks: list of file to check
    Return:
        print the number of label present in all masks
    """
    unique_labels_len = set()
    
    for mask in masks:
        test_mask = cv2.imread(mask)
        unique_labels = np.unique(test_mask)
        len_unique_labels = len(unique_labels)
        unique_labels_len.add(len_unique_labels)

    max_label_len = max(unique_labels_len)
    unique_labels_len = list(unique_labels_len)
    unique_labels_len.sort()
    
    print(f" Number of labels across all masks: {unique_labels_len} \n Maximum number of masks: {max_label_len}")


def choose_data(root, action, image_type):
    """
    Choose the data to train neural network
    Args: 
        root: path of the dataset
        action: train or validated action data
        image_type: left_rgb or semantic images
    Return:
        data: list of files through action dir
    """
    data = []
    print("-------------------------")
    action_dir = os.path.join('', '/{}/{}/{}/'.format(root, action, image_type))
    sub_dir = os.listdir(action_dir)
    sub_dir = [(action_dir + i) for i in sub_dir]

    for i in sub_dir:
        if image_type == 'left_seg':
            i = i + '/label/'
        image_path = os.listdir(i)
        for j in image_path:
            path = i + '/' + j
            data.append(path)
    return data


def process_path(image_path, mask_path):
    """
    Get and process path for image and relative mask
    Args: 
        image_path: path of the dataset
        mask_path: train or validated action data
    Return:
        img: image content
        mask: mask content
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    #img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    #mask = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    #mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask


def preprocess(image, mask):
    """
    Preprocess image and mask to feed the neural network
    Args:
        image: image to preprocess
        mask: image to preprocess
    Return:
        input_image: preprocessed image
        input_mask: preprocessed mask
    """
    input_image = tf.image.resize(image, (512, 512), method='nearest')
    input_mask = tf.image.resize(mask, (512, 512), method='nearest')

    return input_image, input_mask