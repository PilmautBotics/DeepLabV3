#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the deeplabv3p model for your own dataset.
"""
import argparse
import os
import warnings
import time

# Import tensorflow settings
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    TerminateOnNaN,
)

# Import common tools
from deeplab.src.common.callbacks import CheckpointCleanCallBack, EvalCallBack
from deeplab.src.common.model_utils import get_optimizer
from deeplab.src.common.utils import calculate_weigths_labels, get_classes, get_data_list, load_class_weights

# Import model settings
from deeplab.models.mvteccae.data import CAEGenerator
from deeplab.model.mvteccae.metrics import ssim_metric
from deeplab.model.mvteccae.loss import ssim_metric

from deeplab.models.mvteccae.model import get_mvteccae_model

# Try to enable Auto Mixed Precision on TF 2.0
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# configure which gpus to use
num_threads = 2
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)


def main(args):
    # create log_dir regarding the model type used
    log_nb = 0
    log_id = str(log_nb).zfill(3)
    log_dir = "../../results/logs/{}_{}".format(log_id, args.model_type)
    while os.path.exists(log_dir):
        log_nb += 1
        log_id = str(log_nb).zfill(3)
        log_dir = "../../results/logs/{}_{}".format(log_id, args.model_type)

    # -----------------  CALLBACKS FOR TRAINING PROCESS ----------------- #
    monitor = 'Jaccard'

    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_grads=False,
        write_images=False,
        update_freq='batch',
    )
    checkpoint = ModelCheckpoint(
        os.path.join(
            log_dir,
            'ep{epoch:03d}-loss{loss:.3f}-Jaccard{Jaccard:.3f}-val_loss{val_loss:.3f}-val_Jaccard{val_Jaccard:.3f}.h5',
        ),
        monitor='val_{}'.format(monitor),
        mode='max',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_{}'.format(monitor),
        factor=0.5,
        mode='max',
        patience=5,
        verbose=1,
        cooldown=0,
        min_lr=1e-6,
    )
    early_stopping = EarlyStopping(
        monitor='val_{}'.format(monitor), min_delta=0, patience=100, verbose=1, mode='max'
    )
    checkpoint_clean = CheckpointCleanCallBack(log_dir, max_val_keep=5, max_eval_keep=2)
    terminate_on_nan = TerminateOnNaN()

    callbacks = [
        tensorboard,
        checkpoint,
        reduce_lr,
        early_stopping,
        terminate_on_nan,
        checkpoint_clean,
    ]

    # ----------------- GET TRAIN AND VALIDATION DATASET -----------------#
    dataset = get_data_list(args.dataset_file, shuffle=True)
    if args.val_dataset_file:
        val_dataset = get_data_list(args.val_dataset_file, shuffle=True)
        num_train = len(dataset)
        num_val = len(val_dataset)
        dataset.extend(val_dataset)
    else:
        val_split = args.val_split
        num_val = int(len(dataset) * val_split)
        num_train = len(dataset) - num_val

    # ----------------- PREPARE TRAIN ADN VALIDATION DATA GENERATOR ----------------- #
    train_generator = CAEGenerator(
        args.dataset_path,
        dataset[:num_train],
        args.batch_size,
        input_shape=args.model_input_shape,
        is_eval=False,
        augment=True,
        color_mode=1,
    )

    valid_generator = CAEGenerator(
        args.dataset_path,
        dataset[num_train:],
        args.batch_size,
        input_shape=args.model_input_shape,
        is_eval=False,
        augment=False,
        color_mode=1,
    )

    # ----------------- PREPARE ONLINE EVALUATION CALLBACK -----------------#
    if args.eval_online:
        eval_callback = EvalCallBack(
            args.dataset_path,
            dataset[num_train:],
            class_names,
            args.model_input_shape,
            False,
            log_dir,
            eval_epoch_interval=args.eval_epoch_interval,
            save_eval_checkpoint=args.save_eval_checkpoint,
        )
        callbacks.insert(-1, eval_callback)  # add before checkpoint clean

    # ----------------- PREPARE OPTIMIZER -----------------#
    optimizer = "adam"

    # ----------------- PREPARE METRIC -----------------#
    metrics = ssim_metric(1.0)
    loss = ssim_loss(1.0)

    # ----------------- GET NORMAL TRAIN MODEL -----------------#
    model = get_mvteccae_model(color_mode, args.model_input_shape)

    # ----------------- COMPILE MODEL -----------------# 
    model.compile(loss=loss, optimizer="adam", metrics=metrics)
    model.summary()

    # Transfer training some epochs with frozen layers first if needed, to get a stable loss.
    initial_epoch = args.init_epoch

    # Wait 2 seconds for next stage
    time.sleep(2)

    print(
        'Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(
            num_train, num_val, args.batch_size, args.model_input_shape
        )
    )
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        epochs=args.total_epoch,
        initial_epoch=initial_epoch,
        verbose=1,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
        callbacks=callbacks,
    )

    # Finally store model
    tf.keras.models.save_model(model, os.path.join(log_dir, 'trained_final.h5'))

    # saving the model to a YAML file
    yaml_model = model.to_yaml()
    with open(os.path.join(log_dir, 'trained_yaml_final.yaml'), 'w') as yaml_file:
        yaml_file.write(yaml_model)

    # saving the model to a JSONs file
    model_json = model.to_json()
    with open(os.path.join(log_dir, 'trained_json_final.yaml'), "w") as json_file:
        json_file.write(model_json)

    # tf.saved_model.save(model, log_dir)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    # Save the model.
    print(
        "path to save tflite mode : {}".format(
            os.path.join(log_dir, 'trained_tflite_final.tflite')
        )
    )
    with open(os.path.join(log_dir, 'trained_tflite_final.tflite'), "wb") as tflite_writer:
        tflite_writer.write(tflite_model)

    interpreter = tf.lite.Interpreter(
        model_path=os.path.join(log_dir, 'trained_tflite_final.tflite')
    )
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Input Shape:", input_details[0]['shape'])
    print("Input Type:", input_details[0]['dtype'])
    print("Output Shape:", output_details[0]['shape'])
    print("Output Type:", output_details[0]['dtype'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model definition options
    parser.add_argument(
        '--model_type',
        type=str,
        required=False,
        default='mvteccae',
        help='MVTECCAE model type: mvteccae, default=%(default)s',
    )

    # Data options
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=False,
        default='VOC2012/',
        help='dataset path containing images and label png file, default=%(default)s',
    )
    parser.add_argument(
        '--dataset_file',
        type=str,
        required=False,
        default='VOC2012/ImageSets/Segmentation/trainval.txt',
        help='train samples txt file, default=%(default)s',
    )
    parser.add_argument(
        '--val_dataset_file',
        type=str,
        required=False,
        default=None,
        help='val samples txt file, default=%(default)s',
    )
    parser.add_argument(
        '--val_split',
        type=float,
        required=False,
        default=0.1,
        help="validation data persentage in dataset if no val dataset provide, default=%(default)s",
    )

    # Training options
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=16,
        help='batch size for training, default=%(default)s',
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        required=False,
        default='sgd',
        choices=['adam', 'rmsprop', 'sgd'],
        help="optimizer for training (adam/rmsprop/sgd), default=%(default)s",
    )
    parser.add_argument(
        '--loss',
        type=str,
        required=False,
        default='crossentropy',
        choices=['crossentropy', 'focal'],
        help="loss type for training (crossentropy/focal), default=%(default)s",
    )
    parser.add_argument(
        "--init_epoch",
        type=int,
        required=False,
        default=0,
        help="initial training epochs for fine tune training, default=%(default)s",
    )
    parser.add_argument(
        "--total_epoch",
        type=int,
        required=False,
        default=2,
        help="total training epochs, default=%(default)s",
    )

    # Evaluation options
    parser.add_argument(
        '--eval_online',
        default=False,
        action="store_true",
        help='Whether to do evaluation on validation dataset during training',
    )
    parser.add_argument(
        '--eval_epoch_interval',
        type=int,
        required=False,
        default=10,
        help="Number of iteration(epochs) interval to do evaluation, default=%(default)s",
    )
    parser.add_argument(
        '--save_eval_checkpoint',
        default=False,
        action="store_true",
        help='Whether to save checkpoint with best evaluation result',
    )

    args = parser.parse_args()
    args.model_input_shape = (int(256), int(256))

    main(args)
