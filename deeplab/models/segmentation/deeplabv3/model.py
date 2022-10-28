#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create deeplabv3p models
"""
from functools import partial

from tensorflow.keras.layers import Input, Lambda, Reshape, Softmax
from tensorflow.keras.models import Model

from deeplab.models.segmentation.deeplabv3.deeplabv3p_mobilenetv2 import (
    Deeplabv3pLiteMobileNetV2,
    Deeplabv3pMobileNetV2,
)
from deeplab.models.segmentation.deeplabv3.layers import DeeplabConv2D, img_resize

#
# A map of model type to construction function for DeepLabv3+
#
deeplab_model_map = {
    'mobilenetv2': partial(Deeplabv3pMobileNetV2, alpha=1.0),
    'mobilenetv2_lite': partial(Deeplabv3pLiteMobileNetV2, alpha=1.0),
}


def get_deeplabv3p_model(
    model_type,
    num_classes,
    model_input_shape,
    output_stride,
    freeze_level=0,
    weights_path=None,
    training=True,
    use_subpixel=False,
):
    # check if model type is valid
    if model_type not in deeplab_model_map.keys():
        raise ValueError('This model type is not supported now')

    model_function = deeplab_model_map[model_type]

    input_tensor = Input(shape=model_input_shape + (3,), name='image_input')
    model, backbone_len = model_function(
        input_tensor=input_tensor,
        input_shape=model_input_shape + (3,),
        num_classes=3,
        OS=output_stride,
    )

    base_model = Model(model.input, model.layers[-5].output)
    print('backbone layers number: {}'.format(backbone_len))

    x = DeeplabConv2D(num_classes, (1, 1), padding='same', name='conv_upsample')(base_model.output)
    x = Lambda(
        img_resize,
        arguments={'size': (model_input_shape[0], model_input_shape[1]), 'mode': 'bilinear'},
        name='pred_resize',
    )(x)

    # for training model, we need to flatten mask to calculate loss
    if training:
        x = Reshape((model_input_shape[0] * model_input_shape[1], num_classes))(x)

    x = Softmax(name='pred_mask')(x)
    model = Model(base_model.input, x, name='deeplabv3p_' + model_type)

    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    return model
