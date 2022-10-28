import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

colormap = \
   {0:             (0,  0,  0), # black
    1:             (0, 0,  255), # blue
    2:             (255,  0, 0), # red (rgb ?)
    3:             (0, 255, 0), # green : 
    4:             (125, 125, 125), # yellow
    5:             (255, 255, 0), # yellow
    -1:            (255,  255, 255)} # white

city_colormap = \
   {0:             (0,  0,  0),
    1:             (0,  0,  0),
    2:             (0,  0,  0),
    3:             (0,  0,  0),
    4:             (0,  0,  0),
    5:             (111, 74,  0),
    6:             (81,  0, 81),
    7:             (128, 64, 128),
    8:             (244, 35, 232),
    9:             (250, 170, 160),
    10:            (230, 150, 140),
    11:            (70, 70, 70),
    12:            (102, 102, 156),
    13:            (190, 153, 153),
    14:            (180, 165, 180),
    15:            (150, 100, 100),
    16:            (150, 120, 90),
    17:            (153, 153, 153),
    18:            (153, 153, 153),
    19:            (250, 170, 30),
    20:            (220, 220,  0),
    21:            (107, 142, 35),
    22:            (152, 251, 152),
    23:            (70, 130, 180),
    24:            (220, 20, 60),
    25:            (255,  0,  0),
    26:            (0,  0, 142),
    27:            (0,  0, 70),
    28:            (0, 60, 100),
    29:            (0,  0, 90),
    30:            (0,  0, 110),
    31:            (0, 80, 100),
    32:            (0,  0, 230),
    33:           (119, 11, 32),
    34:           (255, 255, 255),
    -1:            (0,  0, 142)}

## --------- Images callback tensorboard --------- ##
class TensorBoardImage(tf.keras.callbacks.Callback):
    def __init__(self, seg_mask_tag, train_image_tag, img_path, logdir, model, num_classes, is_city):
        super().__init__() 
        self.train_image_tag = train_image_tag
        self.seg_mask_tag = seg_mask_tag
        # read image
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        self.img = tf.image.resize(images=image, size=[512, 512])
        self.logdir = logdir
        self.model = model
        self.num_classes = num_classes
        if is_city:
            self.colormap = city_colormap
        else:
            self.colormap = colormap 


    def decode_segmentation_masks(self, mask, n_classes):
        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        for l in range(0, n_classes):
            idx = mask == l
            r[idx] = self.colormap[l][0]
            g[idx] = self.colormap[l][1]
            b[idx] = self.colormap[l][2]
        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def get_overlay(self, image, colored_mask):
        image = tf.keras.preprocessing.image.array_to_img(image)
        image = np.array(image).astype(np.uint8)
        overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
        return overlay


    def on_epoch_end(self, epoch, logs={}):
        pred_image = self.model.predict(np.expand_dims((self.img), axis=0))
        pred_image = np.squeeze(pred_image)
        pred_image = np.argmax(pred_image, axis=2)

        prediction_colormap = self.decode_segmentation_masks(pred_image, self.num_classes)
        overlay = self.get_overlay(self.img, prediction_colormap)

        plot_list = [self.img, overlay, prediction_colormap]
        if prediction_colormap.shape[-1] == 3:
            img = tf.keras.preprocessing.image.array_to_img(prediction_colormap)
            img_overlay = tf.keras.preprocessing.image.array_to_img(overlay)
        else:
            img = prediction_colormap
            img_overlay = overlay

        img = np.reshape(img, (-1, 512, 512, 3))
        img_overlay = np.reshape(img_overlay, (-1, 512, 512, 3))
        writer = tf.summary.create_file_writer(self.logdir)
        with writer.as_default():
            tf.summary.image(
                    self.seg_mask_tag,
                    img,
                    step=epoch)
            tf.summary.image(
                    self.train_image_tag,
                    img_overlay,
                    step=epoch)

        writer.close()

        return