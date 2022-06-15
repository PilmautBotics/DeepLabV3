import matplotlib.pyplot as plt
import tensorflow as tf

# from DeeplabV3.src.model import *
# from DeeplabV3.src.utils import *


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def main(args=None):
    IMAGE_SIZE = 512
    NUM_CLASSES = 129

    # deeplab_v3 = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    model = tf.keras.models.load_model('/home/pguillemaut/deep_ws/DeepLabV3/src/deeplab_v3plus_v1/')

    image_path = '/home/pguillemaut/deep_ws/data/Cityscapes/test/left_rgb/mainz/mainz_000003_017171_leftImg8bit.png'
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    input_image = tf.image.resize(img, (512, 512), method='nearest')

    mask_path = '/home/pguillemaut/deep_ws/data/Cityscapes/test/left_seg/mainz/label/mainz_000003_017171_gtFine_labelIds.png'
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.image.resize(mask, (512, 512), method='nearest')

    display([input_image, mask, create_mask(model.predict(input_image[tf.newaxis, ...]))])


if __name__ == '__main__':
    main()
