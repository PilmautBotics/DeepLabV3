#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create deeplabv3p models
"""
from functools import partial
from tensorflow.keras.layers import Conv2D, Reshape, Activation, Softmax, Lambda, Input
from tensorflow.keras.models import Model


def build_model(color_mode,input_shape):
    # set channels
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3

    # define model
    input_img = tensorflow.keras.layers.Input(shape=(input_shape, channels))
    # Encode-----------------------------------------------------------
    x = tensorflow.keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(
        input_img
    )
    x = tensorflow.keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = tensorflow.keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = tensorflow.keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = tensorflow.keras.layers.Conv2D(64, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = tensorflow.keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = tensorflow.keras.layers.Conv2D(128, (4, 4), strides=2, activation="relu", padding="same")(
        x
    )
    x = tensorflow.keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = tensorflow.keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    encoded = tensorflow.keras.layers.Conv2D(1, (8, 8), strides=1, padding="same")(x)

    # Decode---------------------------------------------------------------------
    x = tensorflow.keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(
        encoded
    )
    x = tensorflow.keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = tensorflow.keras.layers.UpSampling2D((2, 2))(x)
    x = tensorflow.keras.layers.Conv2D(128, (4, 4), strides=2, activation="relu", padding="same")(
        x
    )
    x = tensorflow.keras.layers.UpSampling2D((2, 2))(x)
    x = tensorflow.keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = tensorflow.keras.layers.UpSampling2D((2, 2))(x)
    x = tensorflow.keras.layers.Conv2D(64, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = tensorflow.keras.layers.UpSampling2D((2, 2))(x)
    x = tensorflow.keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = tensorflow.keras.layers.UpSampling2D((2, 2))(x)
    x = tensorflow.keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = tensorflow.keras.layers.UpSampling2D((4, 4))(x)
    x = tensorflow.keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = tensorflow.keras.layers.UpSampling2D((2, 2))(x)
    x = tensorflow.keras.layers.Conv2D(32, (8, 8), activation="relu", padding="same")(x)

    x = tensorflow.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tensorflow.keras.layers.Conv2D(
        channels, (8, 8), activation="sigmoid", padding="same"
    )(x)

    model = tensorflow.keras.models.Model(input_img, decoded)

    return model


def get_mvteccae_model(color_mode, model_input_shape):

    model = build_model(color_mode, input_shape)
    
    return model



