#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert keras model files to frozen pb tensorflow weight file. The resultant TensorFlow model
holds both the model architecture and its associated weights.
"""
import os, sys, argparse, logging
from pathlib import Path
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json, model_from_yaml, load_model

from deeplab.src.common.utils import get_custom_objects

import tf2onnx
import onnxruntime as rt


def keras_to_tensorflow(args):
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model', required=True, type=str, help='Path to the input model.')
    parser.add_argument('--output_model', required=True, type=str, help='Path where the converted model will be stored.')

    keras_to_onnx()


if __name__ == '__main__':
    main()
