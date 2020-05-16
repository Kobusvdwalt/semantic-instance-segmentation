"""An implementation of the Intersection over Union (IoU) metric for Keras."""
from tensorflow.keras import backend as K
import tensorflow as tf
# Jarracrd : https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
# DICE : https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
# IOU : https://www.kaggle.com/vbookshelf/keras-iou-metric-implemented-without-tensor-drama
# F1 : 2*((precision*recall)/(precision+recall+K.epsilon()))

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2))

    return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

def F1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))

def weightedBCE(y_true, y_pred): 
    y_pred = tf.clip_by_value(y_pred, 10e-8, 1. - 10e-8)
    beta1 = 1.0
    beta2 = 1.2
    beta3 = 1.0
    loss1 = -(beta1 * y_true[:, :, :, 0] * tf.math.log(y_pred[:, :, :, 0]) + (1-y_true[:, :, :, 0]) * tf.math.log(1- y_pred[:, :, :, 0]))
    loss2 = -(beta2 * y_true[:, :, :, 1] * tf.math.log(y_pred[:, :, :, 1]) + (1-y_true[:, :, :, 1]) * tf.math.log(1- y_pred[:, :, :, 1]))
    loss3 = -(beta3 * y_true[:, :, :, 2] * tf.math.log(y_pred[:, :, :, 2]) + (1-y_true[:, :, :, 2]) * tf.math.log(1- y_pred[:, :, :, 2]))

    alpha1 = 0.7
    alpha2 = 0.7
    alpha3 = 0.7
    loss1 = alpha1*loss1 + (1-alpha1)*(dice_loss(y_true[:, :, :, 0], y_pred[:, :, :, 0]))
    loss2 = alpha2*loss2 + (1-alpha2)*(dice_loss(y_true[:, :, :, 1], y_pred[:, :, :, 1]))
    loss3 = alpha3*loss3 + (1-alpha3)*(dice_loss(y_true[:, :, :, 2], y_pred[:, :, :, 2]))
    
    t = (loss1*0.2 + loss2*0.7 + loss3*0.1)
    return t
    
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def build_iou_for(label: int, name: str=None):
    """
    Build an Intersection over Union (IoU) metric for a label.
    Args:
        label: the label to build the IoU metric for
        name: an optional name for debugging the built method
    Returns:
        a keras metric to evaluate IoU for the given label
        
    Note:
        label and name support list inputs for multiple labels
    """
    # handle recursive inputs (e.g. a list of labels and names)
    if isinstance(label, list):
        if isinstance(name, list):
            return [build_iou_for(l, n) for (l, n) in zip(label, name)]
        return [build_iou_for(l) for l in label]

    # build the method for returning the IoU of the given label
    def label_iou(y_true, y_pred):
        """
        Return the Intersection over Union (IoU) score for {0}.
        Args:
            y_true: the expected y values as a one-hot
            y_pred: the predicted y values as a one-hot or softmax output
        Returns:
            the scalar IoU value for the given label ({0})
        """.format(label)
        return iou(y_true, y_pred, label)

    # if no name is provided, us the label
    if name is None:
        name = label
    # change the name of the method for debugging
    label_iou.__name__ = 'iou_{}'.format(name)

    return label_iou
        

def mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1]
    # initialize a variable to store total IoU in
    total_iou = K.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        total_iou = total_iou + iou(y_true, y_pred, label)
    # divide total IoU by number of labels to get mean IoU
    return total_iou / num_labels