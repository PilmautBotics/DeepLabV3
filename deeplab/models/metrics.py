from tensorflow.keras import backend as K

import numpy as np 
import tensorflow as tf

def create_mask(pred_mask):
    """
    Args:
        pred_mask: predicted mask from a semantic neural network
    Returns:
        pred_mask[0]: return the image mask predicted
    """
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def ground_iou(y_true, y_pred):
    """
    Compute ground iou 
    Args: 
        - y_true: 
        - y_pred: 
    Returns: 
    """
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = tf.cast(K.reshape(y_true, (-1, 1))[:,0], tf.int32)
    y_true = K.one_hot(y_true, nb_classes)
    true_pixels = K.argmax(y_true, axis=-1) # exclude background
    pred_pixels = K.argmax(y_pred, axis=-1)
    iou = []
    flag = tf.convert_to_tensor(-1, dtype='float64')
    for i in range(nb_classes-1):
        # ground metrics 
        if i in [1]:
            true_labels = K.equal(true_pixels, i)
            pred_labels = K.equal(pred_pixels, i)
            inter = tf.cast(true_labels & pred_labels, tf.int32)
            union = tf.cast(true_labels | pred_labels, tf.int32)
            cond = (K.sum(union) > 0) & (K.sum(tf.cast(true_labels, tf.int32)) > 0)
            res = tf.cond(cond, lambda: K.sum(inter)/K.sum(union), lambda: flag)
            iou.append(res)
    iou = tf.stack(iou)
    legal_labels = tf.greater(iou, flag)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)


def Mean_IOU(y_true, y_pred):
    """
    Compute for individual class the iou metric
    Args:
        y_true: predicted ground truth
        y_pred: estimated prediction
    Returns:
        K.mean(iou): mean intersection over union metric
    """
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = tf.cast(K.reshape(y_true, (-1, 1))[:,0], tf.int32)
    y_true = K.one_hot(y_true, nb_classes)
    true_pixels = K.argmax(y_true, axis=-1) # exclude background
    pred_pixels = K.argmax(y_pred, axis=-1)
    iou = []
    flag = tf.convert_to_tensor(-1, dtype='float64')

    iou_ground = []
    for i in range(nb_classes-1):
        true_labels = K.equal(true_pixels, i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.cast(true_labels & pred_labels, tf.int32)
        union = tf.cast(true_labels | pred_labels, tf.int32)
        cond = (K.sum(union) > 0) & (K.sum(tf.cast(true_labels, tf.int32)) > 0)
        res = tf.cond(cond, lambda: K.sum(inter)/K.sum(union), lambda: flag)
        iou.append(res)
    iou = tf.stack(iou)
    legal_labels = tf.greater(iou, flag)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)