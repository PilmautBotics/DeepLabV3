import argparse
import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from datetime import datetime

from deeplab.models.enet import *
from deeplab.src.utils.utils import *
from deeplab.models.metrics import *
from deeplab.src.utils.coco_dataset import CocoSegDatasetReader

# Parameters used
ROOT_CITYSCAPE_DIR_PATH = 'home/pguillemaut/deep_ws/data/Cityscapes'
ROOT_COCOSEG_DIR_PATH = '/home/pguillemaut/deep_ws/data/project-1-at-2022-08-04-12-07-29486ded/'

IMAGE_SIZE = 512
NUM_CLASSES = 2
EPOCHS = 50
BUFFER_SIZE = 500
BATCH_SIZE = 4  # multiple of 8

def parse_args():
    """
    Necessary arguments to test the script
    """
    parser = argparse.ArgumentParser(description='Enet training pipe')
    parser.add_argument(
        '-id',
        '--id_gpu',
        required=True,
        help='choose the gpu to use',
    )
    return parser.parse_args()


def main(args=None):
    print('Hi from Enet.')

    args = parse_args()

    # configure which gpus to use
    num_threads = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
    os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    tf.config.threading.set_inter_op_parallelism_threads(
        num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(
        num_threads)
    tf.config.set_soft_device_placement(True)
    
    # create log directory
    today_date_time = str(datetime.now())
    today_date_time = today_date_time.replace(' ', '_')
    today_date_time = today_date_time.replace('.', '_')
    log_dir = '../../results/enet/enet-{}'.format(str(today_date_time))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    coco_dataset = CocoSegDatasetReader(ROOT_COCOSEG_DIR_PATH)
    imgs_list = coco_dataset.images
    msks_list = coco_dataset.masks



    middle_idx = int(len(imgs_list) / 2.0)
    print(middle_idx)
    images_list = imgs_list[:middle_idx]
    masks_list = msks_list[:middle_idx]

    image_val_list = imgs_list[middle_idx:]
    mask_val_list = msks_list[middle_idx:]


    #print(image_val_list)
    # print(mask_val_list)
    



    # defined which data is going to be used for train and val dataset
    #image_list = choose_data(ROOT_CITYSCAPE_DIR_PATH, 'train', 'left_rgb')
    #mask_list = choose_data(ROOT_CITYSCAPE_DIR_PATH, 'train', 'left_seg')

    # val images
    #image_val_list = choose_data(ROOT_CITYSCAPE_DIR_PATH, 'val', 'left_rgb')
    #mask_val_list = choose_data(ROOT_CITYSCAPE_DIR_PATH, 'val', 'left_seg')

    # sorted image list
    #image_list = sorted(image_list)
    #mask_list = sorted(mask_list)
    #image_val_list = sorted(image_val_list)
    #mask_val_list = sorted(mask_val_list)

    #print("number of train images/masks are: {} // {}".format(len(image_list), len(mask_list)))
    #print("number of val images/masks are: {} // {} ".format(len(image_val_list), len(mask_val_list)))

    # Split Dataset into unmasked and masked image #
    image_list_ds = tf.data.Dataset.list_files(images_list, shuffle=False)
    mask_list_ds = tf.data.Dataset.list_files(masks_list, shuffle=False)

    image_val_list_ds = tf.data.Dataset.list_files(image_val_list, shuffle=False)
    mask_val_list_ds = tf.data.Dataset.list_files(mask_val_list, shuffle=False)

    image_filenames = tf.constant(images_list)
    masks_filenames = tf.constant(masks_list)

    print("image_filenames", image_filenames)
    print("masks_filenames", masks_filenames)

    image_val_filenames = tf.constant(image_val_list)
    masks_val_filenames = tf.constant(mask_val_list)

    print("image_val_list", image_val_list)
    print("mask_val_list", mask_val_list)

    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))
    val_dataset = tf.data.Dataset.from_tensor_slices((image_val_filenames, masks_val_filenames))

    image_ds = dataset.map(process_path)
    processed_image_ds = image_ds.map(preprocess)
    
    image_val_ds = val_dataset.map(process_path)
    processed_val_image_ds = image_val_ds.map(preprocess)

    # Create the model
    enet = ENet(IMAGE_SIZE, NUM_CLASSES)
    enet.summary()

    # Learning rate decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    enet.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[Mean_IOU]
    )

    # Define training parameters #
    processed_image_ds.batch(BATCH_SIZE)
    processed_val_image_ds.batch(BATCH_SIZE)
    train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    val_dataset = processed_val_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # Define callbacks
    logging = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir + '/tf_logs',
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

    # Train enet
    model_history = enet.fit(
        train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks
    )


    # Saved model trained
    enet_dir = log_dir + "/enet_saved_model"
    enet.save(enet_dir)

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
    plt.savefig("{}/{}".format(log_dir,'deeplabv3_epochs_logloss_curve.png'))


if __name__ == '__main__':
    main()
