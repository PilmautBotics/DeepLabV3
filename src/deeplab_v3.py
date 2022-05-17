import argparse
import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import *
from utils import *


def parse_args():
    """
    Necessary arguments to test the script
    """
    parser = argparse.ArgumentParser(description='DeepLabV3 training pipe')
    parser.add_argument(
        '-id',
        '--id_gpu',
        required=True,
        help='choose the gpu to use',
    )

    return parser.parse_args()


def main(args=None):
    print('Hi from DeepLab V3.')

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

    log_dir = 'tf_log'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    root_path = 'home/pguillemaut/deep_ws/data/Cityscapes'
    image_list = choose_data(root_path, 'train', 'left_rgb', '')
    mask_list = choose_data(root_path, 'train', 'left_seg', 'label')

    # val images
    image_val_list = choose_data(root_path, 'val', 'left_rgb', '')
    mask_val_list = choose_data(root_path, 'val', 'left_seg', 'label')

    # sorted image list
    image_list = sorted(image_list)
    mask_list = sorted(mask_list)
    image_val_list = sorted(image_val_list)
    mask_val_list = sorted(mask_val_list)

    print("number of train images is : {} ".format(len(image_list)))
    print("number of train mask is : {} ".format(len(mask_list)))
    print("number of val images is : {} ".format(len(image_val_list)))
    print("number of val mask is : {} ".format(len(mask_val_list)))

    # Check out the some of the unmasked and masked images from the dataset #
    N = 2
    img = imageio.imread(image_list[N])
    mask = imageio.imread(mask_list[N])

    # Split Dataset into unmasked and masked image #
    image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
    mask_list_ds = tf.data.Dataset.list_files(mask_list, shuffle=False)
    image_val_list_ds = tf.data.Dataset.list_files(image_val_list, shuffle=False)
    mask_val_list_ds = tf.data.Dataset.list_files(mask_val_list, shuffle=False)

    image_filenames = tf.constant(image_list)
    masks_filenames = tf.constant(mask_list)

    image_val_filenames = tf.constant(image_val_list)
    masks_val_filenames = tf.constant(mask_val_list)

    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))
    val_dataset = tf.data.Dataset.from_tensor_slices((image_val_filenames, masks_val_filenames))

    image_ds = dataset.map(process_path)
    processed_image_ds = image_ds.map(preprocess)
    image_val_ds = val_dataset.map(process_path)
    processed_val_image_ds = image_val_ds.map(preprocess)

    # Create the model
    IMAGE_SIZE = 512
    NUM_CLASSES = 129

    deeplab_v3plus = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    deeplab_v3plus.summary()

    # Todo: add metrcis mean IoU for semantic image segmentation
    # Create deeplabv3 model

    # Learning rate decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    deeplab_v3plus.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    # Define training parameters #
    EPOCHS = 150
    BUFFER_SIZE = 500
    BATCH_SIZE = 4  # multiple of 8
    processed_image_ds.batch(BATCH_SIZE)
    processed_val_image_ds.batch(BATCH_SIZE)
    train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    val_dataset = processed_val_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # Define callbacks
    logging = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_grads=False,
        write_images=False,
        update_freq='batch',
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='min'
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, mode='min', patience=10, verbose=1, cooldown=0, min_lr=1e-10
    )
    callbacks = [logging, checkpoint, early_stopping, reduce_lr]

    # Train deepLabV3
    model_history = deeplab_v3plus.fit(
        train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks
    )

    # Saved model trained
    deeplab_v3plus.save("deeplab_v3plus_v1")

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(model_history.history["loss"], label="loss")
    plt.plot(model_history.history["val_loss"], label="val_loss")
    plt.plot(
        np.argmin(model_history.history["val_loss"]),
        np.min(model_history.history["val_loss"]),
        marker="x",
        color="r",
        label="best model",
    )
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig('epochs_logloss_curve.png')


if __name__ == '__main__':
    main()
