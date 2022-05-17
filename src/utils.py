import os

import tensorflow as tf


def choose_data(root, action, image_type, label_type):
    data = []
    train_left_rgb_dir = os.path.join('', '/{}/{}/{}/'.format(root, action, image_type))
    sub_train_dir = os.listdir(train_left_rgb_dir)
    sub_train_dir = [(train_left_rgb_dir + i) for i in sub_train_dir]
    for i in sub_train_dir:
        if label_type:
            i = i + '/{}/'.format(label_type)
        image_path = os.listdir(i)
        image_path = [data.append((i + '/' + j)) for j in image_path]
    return data


def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask


def preprocess(image, mask):
    input_image = tf.image.resize(image, (512, 512), method='nearest')
    input_mask = tf.image.resize(mask, (512, 512), method='nearest')

    return input_image, input_mask
