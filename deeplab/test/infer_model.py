import argparse
from distutils.sysconfig import customize_compiler
import matplotlib.pyplot as plt
import tensorflow as tf

# from deeplab.models.metrics import Mean_IOU, ground_iou

def parse_args():
    """
    Necessary arguments to test the script
    """
    parser = argparse.ArgumentParser(description='Infer model')
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-i', '--image_path', type=str)
    return parser.parse_args()

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def main(args=None):
    IMAGE_SIZE = 512
    NUM_CLASSES = 4

    args = parse_args()
    model = tf.keras.models.load_model(args.model_path, compile=False)

    image_path = args.image_path
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    input_image = tf.image.resize(img, (512, 512), method='nearest')

    """mask_path = '/home/pguillemaut/deep_ws/data/Cityscapes/test/left_seg/mainz/label/mainz_000003_017171_gtFine_labelIds.png'
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.image.resize(mask, (512, 512), method='nearest')"""

    display([input_image, create_mask(model.predict(input_image[tf.newaxis, ...]))])


if __name__ == '__main__':
    main()
